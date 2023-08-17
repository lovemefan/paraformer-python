# -*- coding:utf-8 -*-
# @FileName  :test_vad_online.py
# @Time      :2023/8/9 09:36
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com

from paraformerOnline import AudioReader, FSMNVadOnline

# online
in_cache = []
speech, sample_rate = AudioReader.read_wav_file("test/vad_example.wav")
speech_length = speech.shape[0]

sample_offset = 0
step = 1600
vad_online = FSMNVadOnline()
print(f"The audio totol has {len(speech)} frames")
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
        buffer = vad_online.vad.data_buf
        if buffer is not None:
            frame_start = vad_online.vad.data_buf_start_frame * int(
                vad_online.vad.vad_opts.frame_in_ms
                * vad_online.vad.vad_opts.sample_rate
                / 1000
            )
            print(frame_start, len(vad_online.vad.data_buf))
        print(segments_result)
