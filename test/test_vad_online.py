# -*- coding:utf-8 -*-
# @FileName  :test_vad_online.py
# @Time      :2023/8/9 09:36
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com

from paraformerOnline import FSMNVadOnline
from paraformerOnline.runtime.python.utils.audioHelper import AudioReader

# online
in_cache = []
speech, sample_rate = AudioReader.read_wav_file("test/vad_example.wav")
speech_length = speech.shape[0]

sample_offset = 0
step = 1600
vad_online = FSMNVadOnline()

for sample_offset in range(0, speech_length, min(step, speech_length - sample_offset)):
    if sample_offset + step >= speech_length - 1:
        step = speech_length - sample_offset
        is_final = True
    else:
        is_final = False
    segments_result = vad_online.segments_online(
        speech[sample_offset : sample_offset + step], is_final=is_final
    )
    if segments_result:
        print(segments_result)
