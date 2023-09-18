# -*- coding:utf-8 -*-
# @FileName  :test_file_transcript.py
# @Time      :2023/9/18 16:38
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com
import logging

from paraformerOnline import AsrAllInOne, AudioReader

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(levelname)s] [%(filename)s:%(lineno)d %(module)s.%(funcName)s] %(message)s",
)

if __name__ == "__main__":
    logging.info("Testing online asr")
    wav_path = "test/vad_example.wav"
    speech, sample_rate = AudioReader.read_wav_file(wav_path)

    model = AsrAllInOne(
        mode="file_transcription",
        speaker_verification=True,
        sv_threshold=0.75,
        sv_model_name="cam++",
        hot_words="任意热词 空格隔开",
    )

    results = model.file_transcript(speech)

    for i in results:
        logging.info(i)
