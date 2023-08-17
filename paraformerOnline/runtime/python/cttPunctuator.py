# -*- coding:utf-8 -*-
# @FileName  :cttPuctuator.py
# @Time      :2023/8/9 21:47
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com
from paraformerOnline.runtime.python.model.punc.punctuator import (
    CT_Transformer,
    CT_Transformer_VadRealtime,
)
from paraformerOnline.runtime.python.utils.logger import logger


class CttPunctuator:
    def __init__(self, online: bool = False):
        """
        punctuator with singleton pattern
        :param online:
        """
        self.online = online

        if online:
            logger.info("Initializing punctuator model with online mode.")
            self.model = CT_Transformer_VadRealtime()
            self.param_dict = {"cache": []}
            logger.info("Online model initialized.")
        else:
            logger.info("Initializing punctuator model with offline mode.")
            self.model = CT_Transformer()
            logger.info("Offline model initialized.")

        logger.info("Punc Model initialized.")

    def punctuate(self, text: str, param_dict=None):
        if self.online:
            param_dict = param_dict or self.param_dict
            return self.model(text, param_dict)
        else:
            return self.model(text)
