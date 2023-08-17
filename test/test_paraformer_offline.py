# -*- coding:utf-8 -*-
# @FileName  :test_paraformer_offline.py
# @Time      :2023/8/10 14:48
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com
import logging

from paraformerOnline import AudioReader, ParaformerOffline

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(levelname)s] [%(filename)s:%(lineno)d %(module)s.%(funcName)s] %(message)s",
)
if __name__ == "__main__":
    logging.info("Testing offline asr")
    wav_path = "test/P9_0002.wav"
    speech, sample_rate = AudioReader.read_wav_file(wav_path)
    model = ParaformerOffline()
    result = model.infer_offline(speech)
    logging.info(result)
