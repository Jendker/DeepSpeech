#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import collections
import json
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import shutil
import time
import wave
import numpy as np

from multiprocessing import JoinableQueue, Process, cpu_count, Manager
from mozilla_voice_stt import Model

import sys
import absl.app
import progressbar

import tensorflow as tf
import tensorflow.compat.v1 as tfv1

import tensorflow.compat.v1.logging as tflogging
tflogging.set_verbosity(tflogging.ERROR)

from mvs_ctcdecoder import ctc_beam_search_decoder_batch, Scorer
from mozilla_voice_stt_training.util.helpers import check_ctcdecoder_version

from mozilla_voice_stt_training.util.checkpoints import load_graph_for_evaluation
from mozilla_voice_stt_training.util.config import Config, initialize_globals
from mozilla_voice_stt_training.util.feeding import create_dataset
from mozilla_voice_stt_training.util.flags import create_flags, FLAGS
from mozilla_voice_stt_training.util.logging import log_error, log_progress, create_progressbar


def sentence_from_candidate_transcript(metadata):
    word = ""
    words = []
    # Loop through each character
    for i, token in enumerate(metadata.tokens):
        # Append character to word if it's not a space
        if token.text != " ":
            word = word + token.text
        # Word boundary is either a space or the last character in the array
        if token.text == " " or i == len(metadata.tokens) - 1:
            words.append(word)
            # Reset
            word = ""
    return " ".join(words)


def predictions_from_json_output(json_output):
    return [sentence_from_candidate_transcript(transcript) for transcript in json_output.transcripts]


class Worker:
    def __init__(self):
        check_ctcdecoder_version()
        self.worker_path = os.path.join(FLAGS.worker_path, str(FLAGS.gpu_no))
        self.files_from_last_run = []
        self.results_to_save = {}

        # create output_dir if does not exist
        self.output_dir = os.path.join(self.worker_path, 'result')
        self.input_dir = os.path.join(self.worker_path, 'voicefile')
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        if FLAGS.scorer_path:
            self.scorer = Scorer(FLAGS.lm_alpha, FLAGS.lm_beta,
                                 FLAGS.scorer_path, Config.alphabet)
        else:
            self.scorer = None

        # Get number of accessible CPU cores for this process
        if FLAGS.cpus_no is not None:
            num_processes = FLAGS.cpus_no
        else:
            try:
                available_devices = len(Config.available_devices)
                if available_devices:
                    num_processes = cpu_count() / available_devices
                else:
                    num_processes = cpu_count()
            except NotImplementedError:
                num_processes = 1
        self.num_processes = int(num_processes)
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

    @staticmethod
    def wav_filename_to_id(wav_filename):
        return wav_filename.split('/')[-2]

    @staticmethod
    def wav_filename_to_filename(wav_filename):
        return wav_filename.split('/')[-1]

    @staticmethod
    def alternatives_to_words(alternatives):
        all_words = {}
        for alternative in alternatives:
            words = alternative.split(' ')
            words = set(words)
            for word in words:
                if word in all_words:
                    all_words[word] += 1
                else:
                    all_words[word] = 1
        for word, count in all_words.items():
            all_words[word] = count / len(alternatives)
        return all_words

    def get_prediction_and_save_json(self, predictions, wav_filenames):
        for prediction_alternatives, wav_filename in zip(predictions, wav_filenames):
            id = self.wav_filename_to_id(wav_filename)
            id_result = self.results_to_save[id]
            filename = self.wav_filename_to_filename(wav_filename)
            if prediction_alternatives:
                best_text = prediction_alternatives[0]
                if best_text:
                    words = self.alternatives_to_words(prediction_alternatives)
                else:
                    words = {}
                id_result['segments'][filename]['text'] = best_text
                id_result['segments'][filename]['words'] = words
            else:
                del id_result['segments'][filename]
        ids_saved = []
        for id, id_result in self.results_to_save.items():
            if all('text' in x for x in id_result['segments'].values()):
                # sort by filename
                id_result['segments'] = collections.OrderedDict(sorted(id_result['segments'].items()))
                with open(os.path.join(self.output_dir, id + ".json"), 'w') as f:
                    json.dump(id_result, f)
                ids_saved.append(id)
        for id in ids_saved:
            del self.results_to_save[id]
            shutil.rmtree(os.path.join(self.input_dir, id))

    @staticmethod
    def initialise_session():
        tfv1.train.get_or_create_global_step()
        session = tfv1.Session(config=Config.session_config)
        # Restore variables from training checkpoint
        load_graph_for_evaluation(session)
        return session

    def loop(self, create_model):
        files_processed = False
        if not os.path.exists(self.input_dir):
            return files_processed
        ids = os.listdir(self.input_dir)
        if not ids:
            return files_processed
        for id in ids:
            if not os.path.exists(os.path.join(self.input_dir, id)):
                # file was processed in the last loop and already finished and removed
                continue
            if id in self.results_to_save.keys():
                # file was processed in the last loop
                continue
            try:
                with open(os.path.join(self.input_dir, id, "files.json"), 'r') as f:
                    file_dict = json.load(f)
            except:
                print("File processing failed. Deleting. Path:", os.path.join(self.input_dir, id))
                shutil.rmtree(os.path.join(self.input_dir, id))
                continue
            if file_dict['incidentId'] not in self.results_to_save:
                self.results_to_save[file_dict['incidentId']] = file_dict
            dataset_list_to_predict = []
            for key, segment in file_dict['segments'].items():
                wav_filename = os.path.join(self.input_dir, id, key)
                wav_filesize = segment['wav_filesize']
                wav_transcript = 'dummy'
                new_value = [wav_filename, wav_filesize, wav_transcript]
                dataset_list_to_predict.append(new_value)
            package_processing_dataset, fast_processing_dataset = self.split_processing_method(dataset_list_to_predict)
            if package_processing_dataset:
                print('processing with package len:', len(package_processing_dataset))
                self.predict_with_package(package_processing_dataset)
                if FLAGS.process_in_sequence:
                    predictions, wav_filenames = self.get_package_processing_results()
                    self.get_prediction_and_save_json(predictions, wav_filenames)
                files_processed = True
            new_dataset_length = len(self.files_from_last_run) + len(fast_processing_dataset)
            if new_dataset_length < FLAGS.worker_batch_size:
                self.files_from_last_run = self.files_from_last_run + fast_processing_dataset
            else:
                too_much_for_batch = new_dataset_length % FLAGS.worker_batch_size
                complete_dataset_list_to_predict = self.files_from_last_run + fast_processing_dataset[too_much_for_batch:]
                self.files_from_last_run = fast_processing_dataset[:too_much_for_batch]
                predictions, wav_filenames = self.predict_fast(complete_dataset_list_to_predict, create_model)
                self.get_prediction_and_save_json(predictions, wav_filenames)
                files_processed = True
            if package_processing_dataset and not FLAGS.process_in_sequence:
                predictions, wav_filenames = self.get_package_processing_results()
                self.get_prediction_and_save_json(predictions, wav_filenames)
        return files_processed

    @staticmethod
    def split_processing_method(dataset_list):
        to_process_with_package = []
        fast_process = []
        for x in dataset_list:
            wav_filesize = x[1]
            if wav_filesize > FLAGS.package_larger_than:
                to_process_with_package.append(x)
            else:
                fast_process.append(x)
        return to_process_with_package, fast_process

    def predict_with_package(self, dataset_list):
        tfv1.reset_default_graph()
        def tflite_worker(model, scorer, beam_width, lm_alpha, lm_beta, queue_in, queue_out, gpu_mask):
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_mask)
            ds = Model(model)
            ds.enableExternalScorer(scorer)
            ds.setBeamWidth(beam_width)
            ds.setScorerAlphaBeta(lm_alpha, lm_beta)

            while True:
                try:
                    msg = queue_in.get()

                    filename = msg['filename']
                    target_transcript = msg['transcript']  # dummy
                    fin = wave.open(filename, 'rb')
                    audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)
                    fin.close()

                    prediction_alternatives = predictions_from_json_output(ds.sttWithMetadata(audio, num_results=beam_width))

                    queue_out.put({'wav': filename, 'prediction_alternatives': prediction_alternatives, 'ground_truth': target_transcript})
                except FileNotFoundError as ex:
                    print('FileNotFoundError: ', ex)

                print(queue_out.qsize(), end='\r')  # Update the current progress
                queue_in.task_done()

        self.manager = Manager()
        self.work_todo = JoinableQueue()  # this is where we are going to store input data
        self.work_done = self.manager.Queue()  # this where we are gonna push them out

        self.processes = []
        for i in range(1 if FLAGS.process_in_sequence else min(len(dataset_list), self.num_processes)):  # just use 1 if in sequence - single GPU, otherwise cpus_no
            worker_process = Process(target=tflite_worker, args=(FLAGS.export_dir, FLAGS.scorer_path, FLAGS.beam_width,
                                                                 FLAGS.lm_alpha, FLAGS.lm_beta, self.work_todo,
                                                                 self.work_done, FLAGS.gpu_no),
                                     daemon=True, name='tflite_process_{}'.format(i))
            worker_process.start()  # Launch reader() as a separate python process
            self.processes.append(worker_process)

        for row in dataset_list:
            self.work_todo.put({'filename': row[0], 'transcript': row[2]})  # row[2] - dummy target transcript

    def get_package_processing_results(self):
        wavlist = []
        predictions = []
        self.work_todo.join()
        for process in self.processes:
            process.terminate()

        while not self.work_done.empty():
            msg = self.work_done.get()
            predictions.append(msg['prediction_alternatives'])
            wavlist.append(msg['wav'])
        return predictions, wavlist

    def predict_fast(self, voicefile_list, create_model):
        tfv1.reset_default_graph()

        dataset = create_dataset(voicefile_list, batch_size=FLAGS.worker_batch_size, train_phase=False,
                                 sample_list=True)
        iterator = tfv1.data.Iterator.from_structure(tfv1.data.get_output_types(dataset),
                                                     tfv1.data.get_output_shapes(dataset),
                                                     output_classes=tfv1.data.get_output_classes(dataset))
        transcribe_init_op = iterator.make_initializer(dataset)

        # One rate per layer
        batch_wav_filename, (batch_x, batch_x_len), _ = iterator.get_next()
        no_dropout = [None] * 6
        logits, _ = create_model(batch_x=batch_x,
                                 batch_size=FLAGS.worker_batch_size,
                                 seq_length=batch_x_len,
                                 dropout=no_dropout)

        # Transpose to batch major and apply softmax for decoder
        transposed = tf.nn.softmax(tf.transpose(a=logits, perm=[1, 0, 2]))

        with self.initialise_session() as session:
            predictions = []
            wav_filenames = []

            def run_transcribe(init_op):
                bar = create_progressbar(prefix='Inference epoch | ',
                                         widgets=['Steps: ', progressbar.Counter(), ' | ', progressbar.Timer()]).start()
                log_progress('Inference epoch...')
                step_count = 0

                # Initialize iterator to the appropriate dataset
                session.run(init_op)

                # First pass transposed logits for decoding
                while True:
                    try:
                        batch_wav_filenames, batch_logits, batch_lengths = \
                            session.run([batch_wav_filename, transposed, batch_x_len])
                    except tf.errors.OutOfRangeError:
                        break

                    decoded = ctc_beam_search_decoder_batch(batch_logits, batch_lengths, Config.alphabet, FLAGS.beam_width,
                                                            num_processes=self.num_processes, scorer=self.scorer,
                                                            cutoff_prob=FLAGS.cutoff_prob, cutoff_top_n=FLAGS.cutoff_top_n,
                                                            num_results=FLAGS.beam_width)
                    for wav_filename, d in zip(batch_wav_filenames, decoded):
                        wav_filename = wav_filename.decode('UTF-8')
                        prediction_alternatives = []
                        for prediction_alternative in d:
                            prediction = prediction_alternative[1]
                            prediction_alternatives.append(prediction)
                        predictions.append(prediction_alternatives)
                        wav_filenames.append(wav_filename)
                    step_count += 1
                    bar.update(step_count)
                bar.finish()

            run_transcribe(transcribe_init_op)
            return predictions, wav_filenames


def main(_):
    initialize_globals()

    if not FLAGS.export_dir:
        log_error('flag --export_dir has to be defined for processing with package')
        sys.exit(1)

    if FLAGS.gpu_no is None:
        log_error('flag --gpu_no has to be specified for worker. Tell which GPU is going to process data')
        sys.exit(1)

    if not FLAGS.worker_path:
        log_error('flag --worker_path has to be specified. Tell which root path should be used.')
        sys.exit(1)

    from mozilla_voice_stt_training.train import create_model  # pylint: disable=cyclic-import,import-outside-toplevel
    worker = Worker()

    while True:
        files_were_processed = worker.loop(create_model)
        if not files_were_processed:
            print("No files processed, sleep")
            time.sleep(20)


if __name__ == '__main__':
    create_flags()
    absl.app.run(main)
