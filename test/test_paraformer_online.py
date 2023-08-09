# -*- coding:utf-8 -*-
# @FileName  :test_paraformer_online.py
# @Time      :2023/8/8 21:03
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com
from paraformerOnline import AudioReader, ParaformerOnlineOrtInfer

if __name__ == '__main__':
    wav_path = 'test/P9_0002.wav'
    speech, sample_rate = AudioReader.read_wav_file(wav_path)
    speech_length = speech.shape[0]
    sample_offset = 0
    step = 10 * 960
    model = ParaformerOnlineOrtInfer()
    final_result = ""
    for sample_offset in range(0, speech_length, min(step, speech_length - sample_offset)):
        if sample_offset + step >= speech_length - 1:
            step = speech_length - sample_offset
            is_final = True
        else:
            is_final = False
        rec_result = model.infer_online(speech[sample_offset: sample_offset + step], is_final=is_final)
        if len(rec_result) > 0:
           final_result += rec_result
        print(rec_result.encode('utf-8').decode('utf-8'))
    print(final_result)