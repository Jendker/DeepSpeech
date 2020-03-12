import collections
import glob
import itertools
import json
import os
import shutil
import tempfile
import time
import wave
from functools import partial

from multiprocessing import cpu_count

import sys
import absl.app
import webrtcvad
import wget
import numpy as np
import pandas
import progressbar
import tensorflow as tf
import tensorflow.compat.v1 as tfv1

from ds_ctcdecoder import ctc_beam_search_decoder_batch, Scorer
from pydub import AudioSegment
import audioread

from DeepSpeech import try_loading, create_model
from util.config import Config, initialize_globals
from util.evaluate_tools import calculate_report
from util.feeding import create_dataset, to_sparse_tuple, entry_to_features
from util.flags import create_flags, FLAGS
from util.logging import log_error, log_progress, create_progressbar
from util.text import text_to_char_array
from native_client.python.client import convert_samplerate

SAMPLE_RATE = 8000

test_links_to_download = [['https://media.yoummday.com/1f9/vocRaJse4s60av.mp3', 2], ['https://media.yoummday.com/aa8/vocA4iKg1aynvi.mp3', 2],
                          ['https://media.yoummday.com/be8/voc6w23j4h2whf.mp3', 1]]

# -------------- Taken from DeepSpeech-examples repo --------------

class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.

    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.

    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n

def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    """Filters out non-voiced audio frames.

    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.

    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.

    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.

    Arguments:

    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).

    Returns: A generator that yields PCM audio data.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                triggered = False
                yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
    if triggered:
        pass
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames])

# -------------- End of files taken from DeepSpeech-examples repo --------------


def segment_file(path, output_dir, vad):
    fin = wave.open(path, 'rb')
    fs = fin.getframerate()
    if fs != SAMPLE_RATE:
        raise ValueError('Warning: original sample rate ({}) is different than required {}hz. Check required sample rate or check the file'.format(
                fs, SAMPLE_RATE))
    audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)

    def match_target_amplitude(sound, target_dBFS):
        change_in_dBFS = target_dBFS - sound.dBFS
        return sound.apply_gain(change_in_dBFS)

    sound = AudioSegment(audio.tobytes(), frame_rate=SAMPLE_RATE, channels=1, sample_width=2)
    normalized_sound = match_target_amplitude(sound, -18.0)
    audio = np.frombuffer(normalized_sound.raw_data, dtype=np.int16)

    frames = frame_generator(30, audio.tobytes(), SAMPLE_RATE)
    segments = vad_collector(SAMPLE_RATE, 30, 300, vad, frames)

    for i, segment in enumerate(segments):
        with wave.open(os.path.join(output_dir, "segment{}.wav".format(i)), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(segment)


def main(_):
    initialize_globals()

    if not FLAGS.worker_path:
        log_error('flag --worker_path has to be specified. Tell which root path should be used.')
        sys.exit(1)

    if not FLAGS.gpu_no:
        log_error('flag --gpu_no has to be specified. Tell which gpu is going to process data')
        sys.exit(1)

    if FLAGS.gpu_no >= len(Config.available_devices):
        log_error("gpu_no " + str(FLAGS.gpu_no) + " is to high. Available devices " + str(len(Config.available_devices)))
    if FLAGS.gpu_no >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu_no)

    aggressiveness = 1
    vad = webrtcvad.Vad(aggressiveness)

    output_dir = os.path.join(FLAGS.worker_path, str(FLAGS.gpu_no))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for link, channels in test_links_to_download:
        with tempfile.TemporaryDirectory(dir='/dev/shm/') as tmp:
            file_path = wget.download(link, tmp)
            file_id = os.path.split(file_path)[1].rstrip(".mp3")
            file_id_output_path = os.path.join(output_dir, file_id)
            if not os.path.isdir(file_id_output_path):
                os.mkdir(file_id_output_path)
            for channel in range(channels):
                channel_output_path = os.path.join(file_id_output_path, str(channel))
                if os.path.exists(channel_output_path):
                    shutil.rmtree(channel_output_path)
                os.mkdir(channel_output_path)
                channel_string = ''
                if channels > 1:
                    channel_string = ' remix ' + str(channel + 1)
                new_filename = file_path.replace(".mp3", "_" + str(channel) + ".wav")
                os.system(
                    "sox " + file_path + ' --bits 16 -V1 -c 1 --no-dither --encoding signed-integer --endian little ' +
                    '--compression 0.0 ' + new_filename + channel_string)
                segment_file(new_filename, channel_output_path, vad)
                os.remove(new_filename)
            os.remove(file_path)





if __name__ == '__main__':
    create_flags()
    absl.app.run(main)
