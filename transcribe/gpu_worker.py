#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import collections
import json
import os
import shutil
import time
import wave
import numpy as np

from multiprocessing import JoinableQueue, Process, cpu_count, Manager
from deepspeech import Model

import sys
import absl.app
import progressbar
import tensorflow as tf
import tensorflow.compat.v1 as tfv1

from ds_ctcdecoder import ctc_beam_search_decoder_batch, Scorer
from deepspeech_training.util.helpers import check_ctcdecoder_version

from deepspeech_training.util.checkpoints import load_graph_for_evaluation
from deepspeech_training.util.config import Config, initialize_globals
from deepspeech_training.util.feeding import create_dataset
from deepspeech_training.util.flags import create_flags, FLAGS
from deepspeech_training.util.logging import log_error, log_progress, create_progressbar


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

    def get_prediction_and_save_json(self, predictions, wav_filenames):
        for prediction, wav_filename in zip(predictions, wav_filenames):
            id = self.wav_filename_to_id(wav_filename)
            id_result = self.results_to_save[id]
            filename = self.wav_filename_to_filename(wav_filename)
            if prediction:
                id_result['segments'][filename]['text'] = prediction
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
            with open(os.path.join(self.input_dir, id, "files.json"), 'r') as f:
                file_dict = json.load(f)
            if file_dict['incidentId'] not in self.results_to_save:
                self.results_to_save[file_dict['incidentId']] = file_dict
            dataset_list_to_predict = []
            for key, value in file_dict['segments'].items():
                new_value = value.copy()
                new_value["wav_filename"] = os.path.join(self.input_dir, id, key)
                new_value["transcript"] = "a"  # dummy value needed for evaluation
                dataset_list_to_predict.append(new_value)
            dataset_list_to_predict = sorted(dataset_list_to_predict, key=lambda x: x['wav_filesize'], reverse=False)
            package_processing_dataset, fast_processing_dataset = self.split_processing_method(dataset_list_to_predict)
            if package_processing_dataset:
                print('processing with package len:', len(package_processing_dataset))
                predictions, wav_filenames = self.predict_with_package(package_processing_dataset)
                self.get_prediction_and_save_json(predictions, wav_filenames)
                files_processed = True
            new_dataset_length = len(self.files_from_last_run) + len(fast_processing_dataset)
            if new_dataset_length < FLAGS.worker_batch_size:
                self.files_from_last_run = self.files_from_last_run + fast_processing_dataset
            else:
                too_much_for_batch = new_dataset_length % FLAGS.worker_batch_size
                ready_dataset_list_to_predict = self.files_from_last_run + fast_processing_dataset[too_much_for_batch:]
                self.files_from_last_run = fast_processing_dataset[:too_much_for_batch]
                predictions, wav_filenames = self.predict_fast(ready_dataset_list_to_predict, create_model)
                self.get_prediction_and_save_json(predictions, wav_filenames)
                files_processed = True
        return files_processed

    @staticmethod
    def split_processing_method(dataset_list):
        to_process_with_package = []
        fast_process = []
        for x in dataset_list:
            if x['wav_filesize'] > FLAGS.package_larger_than:
                to_process_with_package.append(x)
            else:
                fast_process.append(x)
        return to_process_with_package, fast_process

    def predict_with_package(self, dataset_list):
        tfv1.reset_default_graph()
        def tflite_worker(model, scorer, beam_width, queue_in, queue_out, gpu_mask):
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_mask)
            ds = Model(model)
            ds.enableExternalScorer(scorer)
            ds.setBeamWidth(beam_width)

            while True:
                try:
                    msg = queue_in.get()

                    filename = msg['filename']
                    fin = wave.open(filename, 'rb')
                    audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)
                    fin.close()

                    decoded = ds.stt(audio)

                    queue_out.put({'wav': filename, 'prediction': decoded, 'ground_truth': msg['transcript']})
                except FileNotFoundError as ex:
                    print('FileNotFoundError: ', ex)

                print(queue_out.qsize(), end='\r')  # Update the current progress
                queue_in.task_done()

        self.manager = Manager()
        self.work_todo = JoinableQueue()  # this is where we are going to store input data
        self.work_done = self.manager.Queue()  # this where we are gonna push them out

        self.processes = []
        for i in range(1):  # just use 1 - single GPU
            worker_process = Process(target=tflite_worker, args=(FLAGS.export_dir, FLAGS.scorer_path, FLAGS.beam_width,
                                                                 self.work_todo, self.work_done, FLAGS.gpu_no),
                                     daemon=True, name='tflite_process_{}'.format(i))
            worker_process.start()  # Launch reader() as a separate python process
            self.processes.append(worker_process)

        for row in dataset_list:
            self.work_todo.put({'filename': row['wav_filename'], 'transcript': row['transcript']})

        return self.get_package_processing_results()

    def get_package_processing_results(self):
        wavlist = []
        predictions = []
        self.work_todo.join()
        for process in self.processes:
            process.terminate()

        while not self.work_done.empty():
            msg = self.work_done.get()
            predictions.append(msg['prediction'])
            wavlist.append(msg['wav'])
        return predictions, wavlist

    def predict_fast(self, voicefile_list, create_model):
        tfv1.reset_default_graph()

        dataset = create_dataset(voicefile_list, batch_size=FLAGS.worker_batch_size, train_phase=False, file_dict=True)
        iterator = tfv1.data.Iterator.from_structure(tfv1.data.get_output_types(dataset),
                                                     tfv1.data.get_output_shapes(dataset),
                                                     output_classes=tfv1.data.get_output_classes(dataset))
        transcribe_init_op = iterator.make_initializer(dataset)

        # One rate per layer
        batch_wav_filename, (batch_x, batch_x_len), batch_y = iterator.get_next()
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
                                                            cutoff_prob=FLAGS.cutoff_prob, cutoff_top_n=FLAGS.cutoff_top_n)
                    for wav_filename, d in zip(batch_wav_filenames, decoded):
                        wav_filename = wav_filename.decode('UTF-8')
                        prediction = d[0][1]
                        predictions.append(prediction)
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

    from deepspeech_training.train import create_model  # pylint: disable=cyclic-import,import-outside-toplevel
    worker = Worker()

    while True:
        files_were_processed = worker.loop(create_model)
        if not files_were_processed:
            time.sleep(20)


if __name__ == '__main__':
    create_flags()
    absl.app.run(main)
