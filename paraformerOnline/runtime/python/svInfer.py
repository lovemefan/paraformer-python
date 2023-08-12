# -*- coding:utf-8 -*-
# @FileName  :svInfer.py
# @Time      :2023/8/12 16:13
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com
import os
from pathlib import Path
from typing import Union

import numpy as np
from paraformerOnline.runtime.python.model.sv.campplus import Campplus
from paraformerOnline.runtime.python.model.sv.eres2net import Eres2net

model_names = {
    'cam++': (Campplus, 'campplus.onnx'),
    'eres2net': (Eres2net, 'eres2net-aug-sv.onnx'),
    'eres2net-quant': (Eres2net, 'eres2net-aug-sv-quant.onnx'),
}


class SpeakerVerificationInfer:
    def __init__(self, model_path=None, model_name='cam++', threshold=0.5):
        if model_name not in model_names:
            raise ValueError(f"model name {model_name} not in {model_names.keys()}")
        project_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        model_dir = os.path.join(project_dir, "onnx", "sv")
        model_path = model_path or os.path.join(model_dir, model_names[model_name][1])

        self.model = model_names[model_name][0](model_path, threshold)

    def register_speaker(self, emb: np.ndarray):
        self.model.recognize(emb)

    def recognize(self, waveform: Union[str, Path, bytes]):
        return self.model.recognize(waveform)
