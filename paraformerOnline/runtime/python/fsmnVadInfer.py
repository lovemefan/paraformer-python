# -*- coding:utf-8 -*-
# @FileName  :fsmnVadInfer.py
# @Time      :2023/8/9 09:30
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com
# -*- coding:utf-8 -*-
# @FileName  :fsmnvad.py
# @Time      :2023/3/31 16:06
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com

__author__ = "lovemefan"
__copyright__ = "Copyright (C) 2016 lovemefan"
__license__ = "MIT"
__version__ = "v0.0.1"

import logging
import os.path
from pathlib import Path
from typing import Tuple, Union

import numpy as np

from paraformerOnline.runtime.python.model.vad.fsmnvad import E2EVadModel
from paraformerOnline.runtime.python.utils.asrOrtInferRuntimeSession import read_yaml
from paraformerOnline.runtime.python.utils.audioHelper import AudioReader
from paraformerOnline.runtime.python.utils.preprocess import (
    WavFrontend,
    WavFrontendOnline,
)

root_dir = Path(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


class FSMNVad(object):
    def __init__(self, config_path=root_dir / "onnx/vad/config.yaml"):
        self.config = read_yaml(config_path)
        self.frontend = WavFrontendOnline(
            cmvn_file=root_dir / "onnx/vad/am.mvn",
            **self.config["WavFrontend"]["frontend_conf"],
        )
        self.config["FSMN"]["model_path"] = root_dir / "onnx/vad/fsmnvad-online.onnx"

        self.vad = E2EVadModel(
            self.config["FSMN"], self.config["vadPostArgs"], root_dir
        )

    def set_parameters(self, mode):
        pass

    def extract_feature(self, waveform):
        fbank, _ = self.frontend.fbank(waveform)
        feats, feats_len = self.frontend.lfr_cmvn(fbank)
        return feats, feats_len

    def is_speech(self, buf, sample_rate=16000):
        assert sample_rate == 16000, "only support 16k sample rate"

    def segments_offline(self, waveform_path: Union[str, Path]):
        """get sements of audio"""

        logging.info(f"load audio {waveform_path}")
        if not os.path.exists(waveform_path):
            raise FileExistsError(f"{waveform_path} is not exist.")
        if os.path.isfile(waveform_path):
            waveform, _sample_rate = AudioReader.read_wav_file(waveform_path)
        else:
            raise FileNotFoundError(str(Path))
        assert (
            _sample_rate == 16000
        ), f"only support 16k sample rate, current sample rate is {_sample_rate}"

        feats, feats_len = self.extract_feature(waveform)
        waveform = waveform[None, ...]
        segments_part, in_cache = self.vad.infer_offline(
            feats[None, ...], waveform, is_final=True
        )
        return segments_part[0]


class FSMNVadOnline(FSMNVad):
    def __init__(self, config_path=None):
        project_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        config_path = config_path or os.path.join(
            project_dir, "onnx", "vad", "config.yaml"
        )
        super(FSMNVadOnline, self).__init__(config_path)
        self.in_cache = None

    def extract_feature(
        self, waveforms: np.ndarray, is_final: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        waveforms_lens = np.zeros(waveforms.shape[0]).astype(np.int32)
        for idx, waveform in enumerate(waveforms):
            waveforms_lens[idx] = waveform.shape[-1]

        feats, feats_len = self.frontend.extract_fbank(
            waveforms, waveforms_lens, is_final
        )
        return feats.astype(np.float32), feats_len.astype(np.int32)

    def is_speech(self, buf, sample_rate=16000):
        assert sample_rate == 16000, "only support 16k sample rate"

    def prepare_cache(self, in_cache: list):
        if len(in_cache) > 0:
            return in_cache
        fsmn_layers = self.config["FSMN"]["encoder_conf"]["fsmn_layers"]
        proj_dim = self.config["FSMN"]["encoder_conf"]["proj_dim"]
        lorder = self.config["FSMN"]["encoder_conf"]["lorder"]
        for i in range(fsmn_layers):
            cache = np.zeros((1, proj_dim, lorder - 1, 1)).astype(np.float32)
            in_cache.append(cache)
        return in_cache

    def segments_online(
        self, waveform: Union[str, np.ndarray], sample_rate=16000, is_final=False
    ):
        """get sements of audio"""

        if self.in_cache is None:
            self.in_cache = []

        if isinstance(waveform, str):
            waveform = AudioReader.read_pcm_byte(waveform)

        assert (
            sample_rate == 16000
        ), f"only support 16k sample rate, current sample rate is {sample_rate}"
        if waveform.ndim == 1:
            waveform = waveform[None, ...]
        feats, feats_len = self.extract_feature(waveform)
        waveform = self.frontend.get_waveforms()
        segments_part, self.in_cache = self.vad.infer_online(
            feats, waveform, self.prepare_cache(self.in_cache), is_final=is_final
        )
        return segments_part

    def get_current_state(
        self, waveform: Union[str, np.ndarray], sample_rate=16000, is_final=False
    ):
        if self.in_cache is None:
            self.in_cache = []

        if isinstance(waveform, str):
            waveform = AudioReader.read_pcm_byte(waveform)

        assert (
            sample_rate == 16000
        ), f"only support 16k sample rate, current sample rate is {sample_rate}"
        if waveform.ndim == 1:
            waveform = waveform[None, ...]
        feats, feats_len = self.extract_feature(waveform)
        waveform = self.frontend.get_waveforms()
        states = self.vad.get_frames_state(
            feats, waveform, self.prepare_cache(self.in_cache), is_final=is_final
        )
        return states
