# -*- coding:utf-8 -*-
# @FileName  :asr_all_in_one.py
# @Time      :2023/8/14 09:31
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com
import numpy as np
from paraformerOnline import ParaformerOffline, FSMNVadOnline, ParaformerOnline, CttPunctuator
from paraformerOnline.runtime.python.svInfer import SpeakerVerificationInfer
from paraformerOnline.runtime.python.utils.logger import logger

mode_available = [
    'offline',
    'file_transcription',
    'online',
    '2pass'
]


class AsrAllInOne:
    def __init__(self,
                 mode: str,
                 *,
                 speaker_verification=False,
                 time_stamp=False,
                 sv_model_name='cam++',
                 sv_threshold=0.8
                 ):
        """
        Args:
          mode:
          speaker_verification:
          time_stamp:
        """
        assert mode in mode_available, f"{mode} is not support, only {mode_available} is available"
        self.mode = mode
        self.speaker_verification = speaker_verification
        self.time_stamp = time_stamp
        self.start_frame = 0
        self.end_frame = 0

        if mode == 'offline':
            self.asr_offline = ParaformerOffline()
        elif mode == 'online':
            self.asr_online = ParaformerOnline()
        elif mode == '2pass':
            self.asr_offline = ParaformerOffline()
            self.asr_online = ParaformerOnline()
            self.vad = FSMNVadOnline()
            self.punc = CttPunctuator(online=True)
        elif mode == 'file_transcription':
            pass

        if speaker_verification:
            self.sv = SpeakerVerificationInfer(model_name=sv_model_name,
                                               threshold=sv_threshold)

    def online(self, chunk: np.ndarray, is_final: bool = False):
        return self.asr_online.infer_online(chunk, is_final)

    def offline(self, audio_data: np.ndarray):
        return self.asr_offline.infer_offline(audio_data)

    def two_pass_asr(self, chunk: np.ndarray, is_final: bool = False):
        if len(chunk) != 9600:
            logger.warn(f"The recommended length of the chunk is 60 ms")
        partial = self.asr_online.infer_online(chunk, is_final)
        final = None
        segments_result = self.vad.segments_online(chunk, is_final=is_final)
        if segments_result:
            self.start_frame = self.end_frame
            self.end_frame = self.vad.vad.data_buf_start_frame * \
                             int(self.vad.vad.vad_opts.frame_in_ms *
                                 self.vad.vad.vad_opts.sample_rate / 1000)
            if self.end_frame is not None:
                buffer = self.vad.vad.data_buf_all[self.start_frame:self.end_frame]
            else:
                buffer = self.vad.vad.data_buf_all[self.start_frame:]

            if buffer is not None:
                final = self.punc.punctuate(
                    self.asr_offline.infer_offline(buffer)
                )[0]

        result = {
            'partial': partial,
        }
        if final is not None:
            result['final'] = final

        return result
