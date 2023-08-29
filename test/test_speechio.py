# -*- coding:utf-8 -*-
# @FileName  :test_speechio.py
# @Time      :2023/8/29 16:27
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com
import argparse
import json

from jiwer import cer
import logging.handlers
import os.path
import sys

from tqdm import tqdm

from paraformerOnline import AudioReader, ParaformerOffline

_LOG_FILE_DIR = "logs"

DEFAULT_FORMAT = (
    "[%(levelname)s] %(asctime)s " "[%(pathname)s:%(lineno)d] %(funcName)s: %(message)s"
)

parse = argparse.ArgumentParser()
parse.add_argument('-i', '--input', required=True, help='input wav file or dir of wav files')
parse.add_argument('-n', '--name', required=True)
args = parse.parse_args()

logger = logging.getLogger('paraformer cer test')
logger.parent = None
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_formatter = logging.Formatter(DEFAULT_FORMAT)
stream_handler.setFormatter(stream_formatter)
logger.addHandler(stream_handler)

file_formatter = logging.Formatter(DEFAULT_FORMAT)
file_handler = logging.handlers.RotatingFileHandler(filename=os.path.join(_LOG_FILE_DIR, f"{args.name}.txt"))
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)


model = ParaformerOffline()


def cal_cer(config):
    input = config.input

    transcripts = dict()
    logger.info("reading trans.txt")
    with open(os.path.join(input, 'trans.txt'), 'r', encoding='utf-8') as file:
        for line in file:
            id, text = line.split('\t')
            transcripts[id] = {'text': text.strip()}

    logger.info("reading wav.scp")
    with open(os.path.join(input, 'wav.scp'), 'r', encoding='utf-8') as file:
        for line in file:
            id, path = line.split('\t')
            transcripts[id]['path'] = path.strip()

    _cer_all = 0
    for id, info in tqdm(transcripts.items()):
        audio, sample_rate = AudioReader.read_wav_file(os.path.join(input, info['path']))
        text = info['text'].replace(' ', '')
        hypothesis = model.infer_offline(audio)
        error = cer(text, hypothesis)
        transcripts[id]['hypothesis'] = hypothesis
        transcripts[id]['error'] = error

        logger.info(f"id:{id} path: {info['path']}, ground_truth: {text} , hypothesis: {hypothesis}, cer:{error}")
        _cer_all += error

    logger.info(f"all cer is :{_cer_all/len(transcripts.keys())}")

    with open(f'{config.name}.json', 'w', encoding='utf-8') as file:
        json.dump(transcripts, file)


if __name__ == '__main__':
    cal_cer(args)
