# -*- coding:utf-8 -*-
# @FileName  :test_asr_all_in_one.py
# @Time      :2023/8/14 10:27
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
    wav_path = "test/P9_0002.wav"
    speech, sample_rate = AudioReader.read_wav_file(wav_path)
    speech_length = speech.shape[0]
    sample_offset = 0
    step = 9600
    model = AsrAllInOne(
        mode="2pass",
        speaker_verification=True,
        sv_threshold=0.75,
        sv_model_name="cam++",
        hot_words='任意热词 空格隔开'
    )

    final_result = ""
    for sample_offset in range(
        0, speech_length, min(step, speech_length - sample_offset)
    ):
        if sample_offset + step >= speech_length - 1:
            step = speech_length - sample_offset
            is_final = True
        else:
            is_final = False
        rec_result = model.two_pass_asr(
            speech[sample_offset : sample_offset + step], is_final=is_final
        )

        logging.info(rec_result)
