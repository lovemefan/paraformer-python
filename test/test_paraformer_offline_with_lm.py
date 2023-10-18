# -*- coding:utf-8 -*-
# @FileName  :test_paraformer_offline_with_lm.py
# @Time      :2023/10/18 17:17
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com
# -*- coding:utf-8 -*-
# @FileName  :test_paraformer_online.py
# @Time      :2023/8/8 21:03
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com
import logging

from paraformer import AudioReader, CttPunctuator, FSMNVad, ParaformerOffline

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(levelname)s] [%(filename)s:%(lineno)d %(module)s.%(funcName)s] %(message)s",
)

if __name__ == "__main__":
    logging.info("Testing offline asr")
    wav_path = "test/vad_example.wav"
    speech, sample_rate = AudioReader.read_wav_file(wav_path)
    model = ParaformerOffline(use_lm=True)
    vad = FSMNVad()
    punc = CttPunctuator()

    segments = vad.segments_offline(speech)
    results = ""
    for part in segments:
        _result = model.infer_offline(
            speech[part[0] * 16 : part[1] * 16],
            hot_words="",
            beam_search=True,
            beam_size=5,
            lm_weight=0.1,
        )
        results += punc.punctuate(_result)[0]
    logging.info(results)
