# -*- coding:utf-8 -*-
# @FileName  :eres2net.py
# @Time      :2023/8/12 15:52
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com
import os

import numpy as np
import onnxruntime

from paraformer.runtime.python.model.sv.campplus import Campplus
from paraformer.runtime.python.utils.singleton import singleton

campplus_class = Campplus().__class__


@singleton
class Eres2net(campplus_class):
    def __init__(self, onnx_path=None, threshold=0.5):
        self.onnx = onnx_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "onnx/campplus.onnx"
        )
        self.sess = onnxruntime.InferenceSession(self.onnx)
        self.output_name = [nd.name for nd in self.sess.get_outputs()]
        self.threshhold = threshold
        self.memory: np.ndarray = None
