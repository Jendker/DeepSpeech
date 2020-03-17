import ast
import json
import os
import sys
import requests
import datetime
import argparse

base_address = 'https://apiqa.yoummday.com/files/'


def results_to_tsv(result_dict):
    tsv_data = ''
    for _, value in result_dict['segments'].items():
        if 'text' not in value:
            print("continuing")
            continue
        tsv_data += str(value["channel"]) + '\t' + str(int(value['start_time'] * 1000)) + '\t' + str(int(value['duration'] * 1000)) + \
                    value['text'] + '\n'
    return tsv_data


def upload_single_result(auth, result_dict):
    data = {'auth': auth}
    data['incidentId'] = result_dict['incidentId']
    data['audioLength'] = result_dict['audioLength']
    data['numChannels'] = result_dict['numChannels']
    data['data'] = results_to_tsv(result_dict)
    response = requests.post(base_address + 'mozdeepspeechtext', data=data)
    response_content = ast.literal_eval(response.content.decode('utf-8'))
    success = bool(response_content)
    if not success:
        print("Upload failed.")
        print("Upload data:", data)
        print("Response:", response_content)
    else:
        print("Upload success")
    return success


def upload_results(auth, worker_path):
    now = datetime.datetime.now()
    YYYYMMDD = "{:04d}{:02d}{:02d}".format(now.year, now.month, now.day)
    archive_folder = os.path.join(worker_path, 'archive', YYYYMMDD)
    if not os.path.isdir(archive_folder):
        os.makedirs(archive_folder)

    for folder_name in [os.path.join(worker_path, o) for o in os.listdir(worker_path) if os.path.isdir(os.path.join(worker_path,o))]:
        if folder_name == 'archive':
            continue
        gpu_no = folder_name
        gpu_results_path = os.path.join(worker_path, gpu_no, 'result')
        if not os.path.isdir(gpu_results_path):
            continue
        for result_json_file_name in [o for o in os.listdir(gpu_results_path) if
                       '.json' in o]:
            result_json_path = os.path.join(gpu_results_path, result_json_file_name)
            with open(result_json_path) as f:
                result_dict = json.load(f)
                upload_success = upload_single_result(auth, result_dict)
            if upload_success:
                os.rename(result_json_path, os.path.join(archive_folder, result_json_file_name))


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--worker_path', type=str)
    args = parser.parse_args()

    if not args.worker_path:
        print('flag --worker_path has to be specified. Tell which root path should be used.')
        sys.exit(1)

    with open('auth', 'r') as f:
        auth = f.read()

    upload_results(auth, args.worker_path)


if __name__ == '__main__':
    main()