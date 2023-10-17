# -*- coding:utf-8 -*-
# @FileName  :transformer_lm.py
# @Time      :2023/10/13 17:32
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com
import os
import pickle

import numpy as np

from paraformer.runtime.python.utils.asrOrtInferRuntimeSession import (
    TokenIDConverter,
    CharTokenizer,
)
from paraformer.runtime.python.utils.lmOrtInderRuntimeSession import (
    LMOrtInferRuntimeSession,
)
from paraformer.runtime.python.utils.singleton import singleton


@singleton
class TransformerLM:
    def __init__(self, model_dir: str = None, intra_op_num_threads=4):
        tokens_list_path = os.path.join(model_dir, "tokens.txt")
        segment_dict_path = os.path.join(model_dir, "seg_dict")
        self.tokens_list = []
        self.segment_dict = {}

        # load tokens
        with open(tokens_list_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line != "" or line != " ":
                    self.tokens_list.append(line)
        assert (
            len(self.tokens_list) == 8404
        ), f"checkout file {tokens_list_path} is complete"

        # load bpe segment dict for english word
        with open(segment_dict_path, "r", encoding="utf-8") as file:
            for line in file:
                word, seg = line.split("\t")
                self.segment_dict[word] = seg

        self.converter = TokenIDConverter(self.tokens_list)
        self.tokenizer = CharTokenizer()

        model_file = os.path.join(model_dir, "lm_quant.onnx")

        self.lm = LMOrtInferRuntimeSession(
            model_file,
            intra_op_num_threads=intra_op_num_threads,
        )

    def seg_tokenize_wo_pattern(self, txt, seg_dict):
        out_txt = ""
        for word in txt:
            if word in seg_dict:
                out_txt += seg_dict[word] + " "
            else:
                out_txt += "<unk>" + " "
        return out_txt.strip().split()

    def get_nll_and_ppl(self, text_ints):
        """
        Args:
             text_ints
        """
        # 1. Create a sentence pair like '<sos> w1 w2 w3' and 'w1 w2 w3 <eos>'
        # text: (Batch, Length) -> x, y: (Batch, Length + 1)
        x = np.pad(text_ints, [1, 0], "constant", constant_values=(1,))[None, ...]
        t = np.pad(text_ints, [0, 1], "constant", constant_values=(2,))

        # 2. Forward Language model
        # x: (Batch, Length) -> y: (Batch, Length, NVocab)
        y = self.lm(x)

        # 3. Calc negative log likelihood
        # cals negative log likelihood (Batch, Length)
        y = y[0]
        y_softmax = [np.exp(y[i]) / np.sum(np.exp(y[i])) for i in range(len(y))]
        negative_log_likelihood = -np.array(
            [np.log(y_softmax[i][t[i]]) for i in range(len(t))]
        )

        # 4. compute nll and ppl
        nll = negative_log_likelihood.sum()
        ppl = np.exp(negative_log_likelihood.mean())

        return nll, ppl

    def get_nll_and_ppl_from_text(self, text: str):
        tokens = text.strip().split(" ")
        if self.segment_dict is not None:
            tokens = self.seg_tokenize_wo_pattern(tokens, self.segment_dict)
        text_ints = np.array(self.converter.tokens2ids(tokens), dtype=np.int64)
        return self.get_nll_and_ppl(text_ints)
