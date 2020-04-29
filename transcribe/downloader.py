import ast
import collections
import json
import os
import shutil
import subprocess
import tempfile
import time
import wave
import sys
import absl.app
import requests
import webrtcvad
import numpy as np
from pydub import AudioSegment

from deepspeech_training.util.flags import create_flags, FLAGS
from deepspeech_training.util.logging import log_error

SAMPLE_RATE = 8000
INITIAL_AGGRESSIVENESS = 1  # for VAD
FILES_TO_FILL = 100
MINIMAL_AUDIO_DURATION = 0.75  # in seconds
BASE_ADDRESS = 'https://api.yoummday.com/files/'


class Frame(object):
    """Represents a "frame" of audio data."""

    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


class Downloader:
    def __init__(self, base_address):
        self.output_dir = os.path.join(FLAGS.worker_path, str(FLAGS.gpu_no), 'voicefile')
        self.base_address = base_address
        auth_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'auth')
        if not os.path.exists(auth_path):
            print('auth file missing. place it with auth key in the transcribe folder')
            sys.exit(1)
        with open(auth_path, 'r') as f:
            self.auth = f.read()

    # -------------- Taken from DeepSpeech-examples repo --------------
    @staticmethod
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
    @staticmethod
    def vad_split(audio_frames,
                  num_padding_frames=10,
                  threshold=0.5,
                  aggressiveness=None):
        if aggressiveness not in [0, 1, 2, 3]:
            raise ValueError('VAD-splitting aggressiveness mode has to be one of 0, 1, 2, or 3. Given', aggressiveness)
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        triggered = False
        vad = webrtcvad.Vad(aggressiveness)
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
                  frame_duration_ms * max(0, frame_index - len(voiced_frames)) / 1000, \
                  frame_duration_ms * (frame_index + 1) / 1000

    # -------------- End of files taken from DeepSpeech-examples repo --------------

    def segment_file(self, path, output_dir, channel, file_dict, aggressiveness=INITIAL_AGGRESSIVENESS):
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

        frames = self.frame_generator(30, audio.tobytes(), SAMPLE_RATE)
        segments = self.vad_split(frames, aggressiveness=aggressiveness)

        filtered_segments = []
        for segment in segments:
            _, time_start, time_end = segment
            duration = time_end - time_start  # in secs
            if duration > MINIMAL_AUDIO_DURATION:
                filtered_segments.append(segment)
        for i, segment in enumerate(filtered_segments):
            filename = "segment{}_{}.wav".format(len(file_dict['segments']), channel)
            output_file_path = os.path.join(output_dir, filename)
            with wave.open(output_file_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                segment_buffer, time_start, time_end = segment
                wf.writeframes(segment_buffer)
                duration = float("{:.2f}".format(time_end - time_start))
            if duration > FLAGS.retry_split_duration and aggressiveness <= 2:
                self.segment_file(output_file_path, output_dir, channel, file_dict, aggressiveness + 1)
            else:
                file_dict['segments'][filename] = {"duration": duration, "start_time": time_start, "channel": channel,
                                                   "wav_filesize": len(segment_buffer)}

    def download_files(self, to_download):
        success = False
        return_list = None
        data = {'auth': self.auth, 'limit': to_download}
        try:
            response = requests.post(self.base_address + 'mozdeepspeechvoice', data=data)
        except Exception as e:
            print("Exception at download_files", e)
            return success, return_list
        response_content = ast.literal_eval(response.content.decode('utf-8'))
        success = bool(response_content)
        return_list = []
        if not success:
            print("Download failed.")
            print("Download data:", data)
            print("Response:", response_content)
        else:
            print("Download success")
            return_list = response_content['list']
        return success, return_list

    def download_and_process_link(self, incident_id, link, audio_length, channels):
        link = link.replace('\\', '')
        file_dict = {'segments': {}, 'incidentId': incident_id, 'numChannels': channels,
                     'audioLength': audio_length}
        with tempfile.TemporaryDirectory(dir='/dev/shm/') as tmp:
            file_path = os.path.join(tmp, "download_file.mp3")
            try:
                subprocess.check_output(['wget', link, '-O', file_path, '-q'], stderr=subprocess.STDOUT)
            except Exception as e:
                print("file download failed with error", e)
                return
            temp_id_output_path = os.path.join(tmp, incident_id)
            file_id_final_output_path = os.path.join(self.output_dir, incident_id)
            if os.path.isdir(file_id_final_output_path):
                # don't touch ids which already have been downloaded
                return
            os.makedirs(temp_id_output_path)
            for channel in range(channels):
                channel_string = ''
                if channels > 1:
                    channel_string = ' remix ' + str(channel + 1)
                new_filename = file_path.replace(".mp3", "_" + str(channel) + ".wav")
                os.system(
                    "sox " + file_path + ' --bits 16 -V1 -c 1 --no-dither --encoding signed-integer --endian little ' +
                    '--compression 0.0 ' + new_filename + channel_string)
                self.segment_file(new_filename, temp_id_output_path, channel, file_dict,
                                  INITIAL_AGGRESSIVENESS)
                os.remove(new_filename)
            with open(os.path.join(temp_id_output_path, "files.json"), 'w') as f:
                json.dump(file_dict, f)
            shutil.copytree(temp_id_output_path, file_id_final_output_path)

    def loop(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        incidents = [f for f in os.listdir(self.output_dir) if os.path.isdir(os.path.join(self.output_dir, f))]
        to_download = FILES_TO_FILL - len(incidents)
        if not to_download:
            print("No new files to download. Continuing.")
            return
        success, files_list = self.download_files(to_download)
        if not success:
            print("Download file list failed.")
            return
        for incident_id, link, audio_length, channels in files_list:
            if FILES_TO_FILL <= len(incidents):
                return
            incidents.append(incident_id)
            self.download_and_process_link(incident_id, link, audio_length, channels)


def main(_):
    if not FLAGS.worker_path:
        log_error('flag --worker_path has to be specified. Tell which root path should be used.')
        sys.exit(1)

    if FLAGS.gpu_no is None:
        log_error('flag --gpu_no has to be specified. Tell which GPU is going to process data.')
        sys.exit(1)

    downloader = Downloader(BASE_ADDRESS)
    # for debugging - use list of files from API and save it as files_list
    # for line in files_list:
    #     downloader.download_and_process_link(*line)
    while True:
        downloader.loop()
        time.sleep(20)


if __name__ == '__main__':
    create_flags()
    absl.app.run(main)
