# -*- coding:utf-8 -*-
# @FileName  :asr_all_in_one.py
# @Time      :2023/8/14 09:31
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com
import time

import numpy as np

from paraformerOnline import (
    CttPunctuator,
    FSMNVadOnline,
    ParaformerOffline,
    ParaformerOnline,
)
from paraformerOnline.runtime.python.svInfer import SpeakerVerificationInfer
from paraformerOnline.runtime.python.utils.logger import logger

mode_available = ["offline", "file_transcription", "online", "2pass"]


class AsrAllInOne:
    def __init__(
        self,
        mode: str,
        *,
        speaker_verification=False,
        time_stamp=False,
        chunk_interval=10,
        sv_model_name="cam++",
        sv_threshold=0.6,
        vad_speech_max_length=20000,
    ):
        """
        Args:
          mode:
          speaker_verification:
          time_stamp:
        """
        assert (
            mode in mode_available
        ), f"{mode} is not support, only {mode_available} is available"
        self.mode = mode
        self.speaker_verification = speaker_verification
        self.time_stamp = time_stamp
        self.start_frame = -1
        self.end_frame = -1
        self.vad_pre_idx = 0
        self.mode = mode
        self.chunk_interval = chunk_interval
        self.speech_start = False
        self.frames = []
        self.offset = 0

        if mode == "offline":
            self.asr_offline = ParaformerOffline()
        elif mode == "online":
            self.asr_online = ParaformerOnline()
        elif mode == "2pass":
            self.frames_asr_offline = []
            self.frames_asr_online = []
            self.asr_offline = ParaformerOffline()
            self.asr_online = ParaformerOnline()
            self.vad = FSMNVadOnline()
            self.vad.vad.vad_opts.max_single_segment_time = vad_speech_max_length
            self.punc = CttPunctuator(online=True)
            self.text_cache = ""

        elif mode == "file_transcription":
            pass

        if speaker_verification:
            self.sv = SpeakerVerificationInfer(
                model_name=sv_model_name, threshold=sv_threshold
            )

    def reset_asr(self):
        self.frames = []
        self.frames_asr_offline = []
        self.frames_asr_online = []
        self.start_frame = -1
        self.end_frame = -1
        self.vad_pre_idx = 0
        self.vad.vad.all_reset_detection()

    def online(self, chunk: np.ndarray, is_final: bool = False):
        return self.asr_online.infer_online(chunk, is_final)

    def offline(self, audio_data: np.ndarray):
        return self.asr_offline.infer_offline(audio_data)

    def extract_endpoint_from_vad_result(self, segments_result):
        segments = []
        for _start, _end in segments_result:
            start = -1
            end = -1
            if _start != -1:
                start = _start
            if _end != -1:
                end = _end
            segments.append([start, end])
        return segments

    def two_pass_asr(self, chunk: np.ndarray, is_final: bool = False):
        self.frames.append(chunk)
        self.vad_pre_idx += len(chunk) // 16

        # paraformer online inference
        self.frames_asr_online.append(chunk)

        if len(self.frames_asr_online) > 0 or self.end_frame != -1:
            time_start = time.time()
            data = np.concatenate(self.frames_asr_online)
            partial = self.asr_online.infer_online(data, is_final)
            self.text_cache += partial
            # empty asr online buffer
            logger.debug(f"asr online inference use {time.time() - time_start} s")
            self.frames_asr_online = []

        if self.speech_start:
            self.frames_asr_offline.append(chunk)

        # parafprmer vad inference
        time_start = time.time()
        segments_result = self.vad.segments_online(chunk, is_final=is_final)
        logger.debug(f"vad online inference use {time.time() - time_start} s")

        segments = self.extract_endpoint_from_vad_result(segments_result)
        final = None
        for start, end in segments:
            self.start_frame = start
            self.end_frame = end
            # print(self.start_frame, self.end_frame)
            if self.start_frame != -1:
                self.speech_start = True
                beg_bias = (self.vad_pre_idx - self.start_frame) / (len(chunk) // 16)
                # print(beg_bias)
                end_idx = (beg_bias % 1) * len(self.frames[-int(beg_bias) - 1])
                frames_pre = [self.frames[-int(beg_bias) - 1][-int(end_idx) :]]
                if int(beg_bias) != 0:
                    frames_pre.extend(self.frames[-int(beg_bias) :])
                frames_pre = [np.concatenate(frames_pre)]
                # print(len(frames_pre[0]))
                self.frames_asr_offline = []
                self.frames_asr_offline.extend(frames_pre)
                # clear the frames queue
                # self.frames = self.frames[-10:]

            # parafprmer offline inference
            if self.end_frame != -1 and len(self.frames_asr_offline) > 0:
                time_start = time.time()
                if len(self.frames_asr_offline) > 1:
                    data = np.concatenate(self.frames_asr_offline[:-1])
                else:
                    data = np.concatenate(self.frames_asr_offline)
                asr_offline_final = self.asr_offline.infer_offline(data)
                logger.debug(f"asr offline inference use {time.time() - time_start} s")
                if self.speaker_verification:
                    time_start = time.time()
                    speaker_id = self.sv.recognize(data)
                    logger.debug(
                        f"asr offline inference use {time.time() - time_start} s"
                    )
                if len(self.frames_asr_offline) > 1:
                    self.frames_asr_offline = [self.frames_asr_offline[-1]]
                else:
                    self.frames_asr_offline = []
                self.speech_start = False
                time_start = time.time()
                _final = self.punc.punctuate(asr_offline_final)[0]
                if final is not None:
                    final += _final
                else:
                    final = _final
                logger.debug(f"punc online inference use {time.time() - time_start} s")

        result = {
            "partial": self.text_cache,
        }
        if final is not None:
            result["final"] = final
            result["partial"] = ""
            if self.speaker_verification:
                result["speaker_id"] = speaker_id
            self.text_cache = ""

        if is_final:
            self.reset_asr()

        return result

    def two_pass_for_dialogue(self, chunk, is_final=False):
        """
        asr for dialogue
        :return:
        """
        self.frames.append(chunk)
        self.vad_pre_idx += len(chunk) // 16

        # paraformer online inference
        self.frames_asr_online.append(chunk)

        if len(self.frames_asr_online) > 0 or self.end_frame != -1:
            time_start = time.time()
            data = np.concatenate(self.frames_asr_online)
            partial = self.asr_online.infer_online(data, is_final)
            self.text_cache += partial
            # empty asr online buffer
            self.frames_asr_online = []

            logger.debug(f"asr online inference use {time.time() - time_start} s")

        if self.speech_start:
            self.frames_asr_offline.append(chunk)

        # parafprmer vad inference

        time_start = time.time()
        segments_result = self.vad.segments_online(chunk, is_final=is_final)
        logger.debug(f"vad online inference use {time.time() - time_start} s")
        self.start_frame, self.end_frame = self.extract_endpoint_from_vad_result(
            segments_result
        )

        if self.start_frame != -1:
            self.speech_start = True
            # self.vad.vad.all_reset_detection()
            beg_bias = (self.vad_pre_idx - self.start_frame) / (len(chunk) // 16)
            end_idx = (beg_bias % 1) * len(self.frames[-int(beg_bias) - 1])
            frames_pre = [self.frames[-int(beg_bias) - 1][-int(end_idx) :]]
            if int(beg_bias) != 0:
                frames_pre.extend(self.frames[-int(beg_bias) :])
            frames_pre = [np.concatenate(frames_pre)]

            self.frames_asr_offline = []
            self.frames_asr_offline.extend(frames_pre)

        final = None
        # parafprmer offline inference
        if self.end_frame != -1 and len(self.frames_asr_offline) > 0 or is_final:
            time_start = time.time()
            data = np.concatenate(self.frames_asr_offline)
            # print(len(data))
            asr_offline_final = self.asr_offline.infer_offline(data)
            if self.speaker_verification:
                speaker_id = self.sv.recognize(data)
            logger.debug(f"asr offline inference use {time.time() - time_start} s")
            self.frames_asr_offline = []
            self.speech_start = False
            time_start = time.time()
            final = self.punc.punctuate(asr_offline_final)[0]
            logger.debug(f"punc online inference use {time.time() - time_start} s")

        result = {
            "partial": self.text_cache,
        }
        if final is not None:
            result["final"] = final
            result["partial"] = ""
            if self.speaker_verification:
                result["speaker_id"] = speaker_id
            self.text_cache = ""

        return result
