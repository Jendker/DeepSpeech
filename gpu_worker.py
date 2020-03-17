#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import csv
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
from util.helpers import check_ctcdecoder_version

from util.checkpoints import load_or_init_graph
from util.config import Config, initialize_globals
from util.feeding import create_dataset
from util.flags import create_flags, FLAGS
from util.logging import log_error, log_progress, create_progressbar


class Worker:
    def __init__(self):
        check_ctcdecoder_version()
        self.session = None
        self.logits = None
        self.transposed = None
        self.iterator = None
        self.worker_path = os.path.join(FLAGS.worker_path, str(FLAGS.gpu_no))

        # create output_dir if does not exist
        self.output_dir = os.path.join(self.worker_path, 'result')
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

        if FLAGS.scorer_path:
            self.scorer = Scorer(FLAGS.lm_alpha, FLAGS.lm_beta,
                                 FLAGS.scorer_path, Config.alphabet)
        else:
            self.scorer = None

        # Get number of accessible CPU cores for this process
        try:
            available_devices = len(Config.available_devices)
            if available_devices:
                num_processes = cpu_count() / available_devices
            else:
                num_processes = cpu_count()
        except NotImplementedError:
            num_processes = 1
        self.num_processes = int(num_processes)

    @staticmethod
    def initialise_session():
        tfv1.train.get_or_create_global_step()
        session = tfv1.Session(config=Config.session_config)
        # Restore variables from training checkpoint
        load_or_init_graph(session, method_order=['best'])
        return session

    def worker_loop(self, create_model):
        ids = os.listdir(os.path.join(self.worker_path, "voicefile"))
        if not ids:
            return
        for id in ids:
            if id == ".DS_Store":
                continue
            with open(os.path.join(self.worker_path, "voicefile", id, "files.json"), 'r') as f:
                file_dict = json.load(f)
            file_dict_segments = file_dict['segments']
            dataset_list = []
            for key, value in file_dict_segments.items():
                value["wav_filename"] = os.path.join(self.worker_path, "voicefile", id, key)
                value["transcript"] = "a"
                dataset_list.append(value)
            print("evaluating ID", id)
            with open('../temp/' + id + '.csv', 'w', newline='') as csvfile:
                fieldnames = ['wav_filename', 'wav_filesize', 'transcript']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for key, value in file_dict_segments.items():
                    writer.writerow({'wav_filename': value['wav_filename'], 'wav_filesize': value['wav_filesize'], 'transcript': value['transcript']})
            predictions = self.evaluate(dataset_list, create_model, file_dict_segments)
            if not os.path.isdir(self.output_dir):
                os.mkdir(self.output_dir)
            with open(os.path.join(self.output_dir, id + "_text.txt"), 'w') as f:
                f.write("\n".join(predictions))
            with open(os.path.join(self.output_dir, id + ".json"), 'w') as f:
                json.dump(file_dict, f)
            shutil.rmtree(os.path.join(self.worker_path, "voicefile", id))

    def evaluate(self, voicefile_list, create_model, file_dict_segments):
        dataset = create_dataset(voicefile_list, batch_size=FLAGS.worker_batch_size, train_phase=False)
        if self.iterator is None:
            self.iterator = tfv1.data.Iterator.from_structure(tfv1.data.get_output_types(dataset),
                                                              tfv1.data.get_output_shapes(dataset),
                                                              output_classes=tfv1.data.get_output_classes(dataset))
        transcribe_init_op = self.iterator.make_initializer(dataset)

        # One rate per layer
        if self.logits is None:
            self.batch_wav_filename, (self.batch_x, self.batch_x_len), self.batch_y = self.iterator.get_next()
            no_dropout = [None] * 6
            self.logits, _ = create_model(batch_x=self.batch_x,
                                          batch_size=FLAGS.worker_batch_size,
                                          seq_length=self.batch_x_len,
                                          dropout=no_dropout)

        # Transpose to batch major and apply softmax for decoder
        if self.transposed is None:
            self.transposed = tf.nn.softmax(tf.transpose(a=self.logits, perm=[1, 0, 2]))

        if self.session is None:
            self.session = self.initialise_session()

        predictions = []

        def run_transcribe(init_op):
            bar = create_progressbar(prefix='Inference epoch | ',
                                     widgets=['Steps: ', progressbar.Counter(), ' | ', progressbar.Timer()]).start()
            log_progress('Inference epoch...')
            step_count = 0

            # Initialize iterator to the appropriate dataset
            self.session.run(init_op)

            # First pass transposed logits for decoding
            while True:
                try:
                    batch_wav_filenames, batch_logits, batch_lengths = \
                        self.session.run([self.batch_wav_filename, self.transposed, self.batch_x_len])
                except tf.errors.OutOfRangeError:
                    break

                decoded = ctc_beam_search_decoder_batch(batch_logits, batch_lengths, Config.alphabet, FLAGS.beam_width,
                                                        num_processes=self.num_processes, scorer=self.scorer,
                                                        cutoff_prob=FLAGS.cutoff_prob, cutoff_top_n=FLAGS.cutoff_top_n)
                for wav_filename, d in zip(batch_wav_filenames, decoded):
                    filename = wav_filename.decode('UTF-8').split("/")[-1]
                    prediction = d[0][1]
                    if not prediction:
                        del file_dict_segments[filename]
                        continue
                    predictions.append(prediction)
                    file_dict_segments[filename]["text"] = prediction
                    del file_dict_segments[filename]["transcript"]  # remove unneeded dummy key
                step_count += 1
                bar.update(step_count)
            bar.finish()

        print('Running inference on files')
        run_transcribe(transcribe_init_op)
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

    from DeepSpeech import create_model  # pylint: disable=cyclic-import,import-outside-toplevel

    worker = Worker()

    while True:
        print("Loop")
        worker.worker_loop(create_model)
        print("Loop finished")
        time.sleep(20)


if __name__ == '__main__':
    create_flags()
    absl.app.run(main)
