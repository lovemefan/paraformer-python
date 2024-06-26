# -*- coding:utf-8 -*-
# @FileName  :paraformerInfer.py
# @Time      :2023/8/8 20:40
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com
import os.path

import numpy as np

from paraformer.runtime.python.model.asr.paraformer import (
    ParaformerOfflineModel,
    ParaformerOnlineModel,
)
from paraformer.runtime.python.utils.logger import logger


class ParaformerOnline:
    def __init__(self, model_dir=None, *, chunk_size=None, intra_op_num_threads=4):
        self.chunk_size = chunk_size or [5, 10, 5]
        project_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        model_dir = model_dir or os.path.join(project_dir, "onnx", "asr_online")
        logger.info(f"Load onnx model dir at {model_dir}")
        self.model = ParaformerOnlineModel(
            model_dir,
            batch_size=1,
            quantize=True,
            chunk_size=self.chunk_size,
            intra_op_num_threads=intra_op_num_threads,
        )
        self.param_dict = {"cache": dict()}

    def infer_online(self, chunk: np.ndarray, is_final=False):
        """
        Args:
            chunk: 300ms is best
            is_final: final flag of chunk

        Return:
            transcript of audio
        """
        self.param_dict["is_final"] = is_final
        result = self.model(audio_in=chunk, param_dict=self.param_dict)

        return result[0]["preds"][0] if len(result) > 0 else ""

    def reset_cache(self):
        self.param_dict = {"cache": dict()}


class ParaformerOffline:
    def __init__(self, model_dir=None, *, use_lm=False, intra_op_num_threads=4):
        project_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        model_dir = model_dir or os.path.join(project_dir, "onnx", "asr_offline")
        logger.info(f"Load onnx model dir at {model_dir}")
        self.model = ParaformerOfflineModel(
            model_dir, intra_op_num_threads=intra_op_num_threads, use_lm=use_lm
        )
        self.param_dict = {"cache": dict()}

    def infer_offline(
        self,
        audio: np.ndarray,
        hot_words: str = "",
        beam_search=False,
        beam_size=5,
        lm_weight=0.15,
    ):
        """
        Args:
            audio: 600ms is best
            hot_words: hot words split by space . eg `a b cc`
            beam_search
            beam_size

        Return:
            transcript of audio
        """
        result = self.model.infer(
            audio,
            hot_words,
            beam_search=beam_search,
            beam_size=beam_size,
            lm_weight=lm_weight,
        )

        return result[0][0]
