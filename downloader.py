import collections
import json
import os
import tempfile
import wave
import sys
import absl.app
import webrtcvad
import wget
import numpy as np
from pydub import AudioSegment

from util.config import Config, initialize_globals
from util.flags import create_flags, FLAGS
from util.logging import log_error

SAMPLE_RATE = 8000

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

# taken from DSAlign
def vad_split(audio_frames,
              num_padding_frames=10,
              threshold=0.5,
              aggressiveness=3):
    if aggressiveness not in [0, 1, 2, 3]:
        raise ValueError('VAD-splitting aggressiveness mode has to be one of 0, 1, 2, or 3')
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False
    vad = webrtcvad.Vad(int(aggressiveness))
    voiced_frames = []
    frame_duration_ms = 0
    frame_index = 0
    for frame_index, frame in enumerate(audio_frames):
        frame_duration_ms = frame.duration * 1000
        frame = frame.bytes
        if int(frame_duration_ms) not in [10, 20, 30]:
            raise ValueError('VAD-splitting only supported for frame durations 10, 20, or 30 ms')
        is_speech = vad.is_speech(frame, SAMPLE_RATE)
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > threshold * ring_buffer.maxlen:
                triggered = True
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > threshold * ring_buffer.maxlen:
                triggered = False
                yield b''.join(voiced_frames), \
                      frame_duration_ms * max(0, frame_index - len(voiced_frames)) / 1000, \
                      frame_duration_ms * frame_index / 1000
                ring_buffer.clear()
                voiced_frames = []
    if len(voiced_frames) > 0:
        yield b''.join(voiced_frames), \
              frame_duration_ms * (frame_index - len(voiced_frames)) / 1000, \
              frame_duration_ms * (frame_index + 1) / 1000

# -------------- End of files taken from DeepSpeech-examples repo --------------


def segment_file(path, output_dir, aggressiveness, channel, file_dict):
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
    segments = vad_split(frames, aggressiveness=aggressiveness)

    filtered_segments = []
    filter_shorter_than = 0.5  # seconds
    for segment in segments:
        _, time_start, time_end = segment
        duration = time_end - time_start  # in secs
        if duration > filter_shorter_than:
            filtered_segments.append(segment)
    for i, segment in enumerate(filtered_segments):
        filename = "segment{}_{}.wav".format(i, channel)
        with wave.open(os.path.join(output_dir, filename), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            segment_buffer, time_start, time_end = segment
            wf.writeframes(segment_buffer)
            duration = float("{:.2f}".format(time_end - time_start))
            file_dict['segments'][filename] = {"duration": duration, "start_time": time_start, "channel": channel,
                                               "wav_filesize": len(segment_buffer)}


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

    output_dir = os.path.join(FLAGS.worker_path, str(FLAGS.gpu_no), 'voicefile')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for link, channels, incident_id in test_links_to_download:
        audio_length = 123
        file_dict = {'segments': {}, 'incidentId': incident_id, 'numChannels': channels, 'audioLength': audio_length}
        with tempfile.TemporaryDirectory() as tmp:
            file_path = wget.download(link, tmp)
            file_id_output_path = os.path.join(output_dir, incident_id)
            if os.path.isdir(file_id_output_path):
                # don't touch ids which already have been downloaded
                continue
            os.mkdir(file_id_output_path)
            for channel in range(channels):
                channel_string = ''
                if channels > 1:
                    channel_string = ' remix ' + str(channel + 1)
                new_filename = file_path.replace(".mp3", "_" + str(channel) + ".wav")
                os.system(
                    "sox " + file_path + ' --bits 16 -V1 -c 1 --no-dither --encoding signed-integer --endian little ' +
                    '--compression 0.0 ' + new_filename + channel_string)
                segment_file(new_filename, file_id_output_path, aggressiveness, channel, file_dict)
                os.remove(new_filename)
            with open(os.path.join(file_id_output_path, "files.json"), 'w') as f:
                json.dump(file_dict, f)
            os.remove(file_path)


if __name__ == '__main__':
    create_flags()
    absl.app.run(main)
