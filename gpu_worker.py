#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import glob
import json
import os
import shutil
import time
from functools import partial

from multiprocessing import cpu_count

import sys
import absl.app
import pandas
import progressbar
import tensorflow as tf
import tensorflow.compat.v1 as tfv1

from ds_ctcdecoder import ctc_beam_search_decoder_batch, Scorer

from DeepSpeech import try_loading, create_model
from util.config import Config, initialize_globals
from util.feeding import create_dataset, to_sparse_tuple, entry_to_features
from util.flags import create_flags, FLAGS
from util.logging import log_error, log_progress, create_progressbar
from util.text import text_to_char_array


class Worker:
    def __init__(self):
        self.session = self.initialise_session()
        self.logits = None
        self.transposed = None
        self.worker_path = os.path.join(FLAGS.worker_path, str(FLAGS.cpu_no))

        # create output_dir if does not exist
        self.output_dir = os.path.join(self.worker_path, 'result')
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)


    def create_dataset(self, file_list, batch_size, enable_cache=False, cache_path=None, train_phase=False):
        df = pandas.concat(file_list, join='inner', ignore_index=True)
        df.sort_values(by='wav_filesize', inplace=True)

        df['transcript'] = df.apply(text_to_char_array, alphabet=Config.alphabet, result_type='reduce', axis=1)

        def generate_values():
            for _, row in df.iterrows():
                yield row.wav_filename, to_sparse_tuple(row.transcript)

        # Batching a dataset of 2D SparseTensors creates 3D batches, which fail
        # when passed to tf.nn.ctc_loss, so we reshape them to remove the extra
        # dimension here.
        def sparse_reshape(sparse):
            shape = sparse.dense_shape
            return tf.sparse.reshape(sparse, [shape[0], shape[2]])

        def batch_fn(wav_filenames, features, features_len, transcripts):
            features = tf.data.Dataset.zip((features, features_len))
            features = features.padded_batch(batch_size,
                                             padded_shapes=([None, Config.n_input], []))
            transcripts = transcripts.batch(batch_size).map(sparse_reshape)
            wav_filenames = wav_filenames.batch(batch_size)
            return tf.data.Dataset.zip((wav_filenames, features, transcripts))

        num_gpus = len(Config.available_devices)
        process_fn = partial(entry_to_features, train_phase=train_phase)

        dataset = (tf.data.Dataset.from_generator(generate_values,
                                                  output_types=(tf.string, (tf.int64, tf.int32, tf.int64)))
                   .map(process_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE))

        if enable_cache:
            dataset = dataset.cache(cache_path)

        dataset = (dataset.window(batch_size, drop_remainder=True).flat_map(batch_fn)
                   .prefetch(num_gpus))

        return dataset

    @staticmethod
    def sparse_tensor_value_to_texts(value, alphabet):
        r"""
        Given a :class:`tf.SparseTensor` ``value``, return an array of Python strings
        representing its values, converting tokens to strings using ``alphabet``.
        """
        return Worker.sparse_tuple_to_texts((value.indices, value.values, value.dense_shape), alphabet)

    @staticmethod
    def sparse_tuple_to_texts(sp_tuple, alphabet):
        indices = sp_tuple[0]
        values = sp_tuple[1]
        results = [[] for _ in range(sp_tuple[2][0])]
        for i, index in enumerate(indices):
            results[index[0]].append(values[i])
        # List of strings
        return [alphabet.decode(res) for res in results]

    @staticmethod
    def initialise_session():
        tfv1.train.get_or_create_global_step()
        session = tfv1.Session(config=Config.session_config)
        # Create a saver using variables from the above newly created graph
        saver = tfv1.train.Saver()
        # Restore variables from training checkpoint
        loaded = try_loading(session, saver, 'best_dev_checkpoint', 'best validation')
        if not loaded:
            print('Could not load checkpoint from {}'.format(FLAGS.checkpoint_dir))
            sys.exit(1)
        return session

    def json_output(self, file_id, ):
        pass

    def worker_loop(self):
        ids = os.listdir(self.worker_path)
        if not ids:
            return
        for id in ids:
            with open(os.path.join(self.worker_path, id, "files.json"), 'r') as f:
                file_dict = json.load(f)
            files = list(filter(os.path.isfile, glob.glob(os.path.join(self.worker_path, id, 'voicefile', "*"))))
            files.sort(key=lambda x: os.path.getmtime(x))
            from DeepSpeech import create_model, try_loading  # pylint: disable=cyclic-import
            files_to_take = len(files)
            while files_to_take % FLAGS.worker_batch_size != 0:
                files_to_take -= 1
            files_to_process = files[:files_to_take]
            for index, file in enumerate(files_to_process):
                new_filename = file + '.processing'
                os.rename(file, new_filename)
                files_to_process[index] = new_filename
            predictions = self.evaluate(files_to_process)
            for prediction, file_to_process in zip(predictions, files_to_process):
                file_dict[file_to_process]["text"] = prediction
            with open(os.path.join(self.output_dir, id + ".json"), 'w') as f:
                json.dump(file_dict, f)
            shutil.rmtree(os.path.join(self.worker_path, id))

    def evaluate(self, voicefile_list):
        if FLAGS.lm_binary_path:
            scorer = Scorer(FLAGS.lm_alpha, FLAGS.lm_beta,
                            FLAGS.lm_binary_path, FLAGS.lm_trie_path,
                            Config.alphabet)
        else:
            scorer = None

        transcribe_set = create_dataset(voicefile_list, batch_size=FLAGS.worker_batch_size, train_phase=False)
        iterator = tfv1.data.Iterator.from_structure(tfv1.data.get_output_types(transcribe_set),
                                                     tfv1.data.get_output_shapes(transcribe_set),
                                                     output_classes=tfv1.data.get_output_classes(transcribe_set))
        transcribe_init_op = iterator.make_initializer(transcribe_set)

        batch_wav_filename, (batch_x, batch_x_len), batch_y = iterator.get_next()

        # One rate per layer
        no_dropout = [None] * 6
        if self.logits is None:
            self.logits, _ = create_model(batch_x=batch_x,
                                     batch_size=FLAGS.worker_batch_size,
                                     seq_length=batch_x_len,
                                     dropout=no_dropout)

        # Transpose to batch major and apply softmax for decoder
        if self.transposed is None:
            self.transposed = tf.nn.softmax(tf.transpose(a=self.logits, perm=[1, 0, 2]))

        tfv1.train.get_or_create_global_step()

        # Get number of accessible CPU cores for this process
        try:
            num_processes = cpu_count() / len(Config.available_devices)
        except NotImplementedError:
            num_processes = 1

        def run_test(init_op):
            wav_filenames = []
            predictions = []
            ground_truths = []

            bar = create_progressbar(prefix='Test epoch | ',
                                     widgets=['Steps: ', progressbar.Counter(), ' | ', progressbar.Timer()]).start()
            log_progress('Test epoch...')

            step_count = 0

            # Initialize iterator to the appropriate dataset
            self.session.run(init_op)

            # First pass, compute losses and transposed logits for decoding
            while True:
                try:
                    batch_wav_filenames, batch_logits, batch_lengths, batch_transcripts = \
                        self.session.run([batch_wav_filename, self.transposed, batch_x_len, batch_y])
                except tf.errors.OutOfRangeError:
                    break

                decoded = ctc_beam_search_decoder_batch(batch_logits, batch_lengths, Config.alphabet, FLAGS.beam_width,
                                                        num_processes=num_processes, scorer=scorer,
                                                        cutoff_prob=FLAGS.cutoff_prob, cutoff_top_n=FLAGS.cutoff_top_n)
                predictions.extend(d[0][1] for d in decoded)
                ground_truths.extend(self.sparse_tensor_value_to_texts(batch_transcripts, Config.alphabet))
                wav_filenames.extend(wav_filename.decode('UTF-8') for wav_filename in batch_wav_filenames)

                step_count += 1
                bar.update(step_count)

            bar.finish()

            return predictions

        print('Running infrenece on files')
        predictions = run_test(transcribe_init_op)
        return predictions


def main(_):
    initialize_globals()

    if FLAGS.gpu_no is None:
        log_error('flag --gpu_no has to be specified for worker.')
        sys.exit(1)

    if not FLAGS.worker_path:
        log_error('flag --worker_path has to be specified. Tell which root path should be used.')
        sys.exit(1)

    if FLAGS.gpu_no >= len(Config.available_devices):
        log_error("gpu_no " + str(FLAGS.gpu_no) + " is to high. Available devices " + str(len(Config.available_devices)))
    if FLAGS.gpu_no >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu_no)

    worker = Worker()

    while True:
        worker.worker_loop()
        time.sleep(20)


if __name__ == '__main__':
    create_flags()
    absl.app.run(main)
