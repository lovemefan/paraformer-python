# -*- coding:utf-8 -*-
# @FileName  :test_vad_offline.py
# @Time      :2023/9/6 16:19
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com
from paraformer import AudioReader, FSMNVad

if __name__ == "__main__":
    speech, sample_rate = AudioReader.read_wav_file("test/vad_example.wav")
    speech_length = speech.shape[0]
    vad = FSMNVad()

    result = vad.segments_offline(speech)
    print(result)
