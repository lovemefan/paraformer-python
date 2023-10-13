# -*- coding:utf-8 -*-
# @FileName  :cttPuctuator.py
# @Time      :2023/8/9 21:47
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com
from paraformer.runtime.python.model.punc.punctuator import CT_Transformer
from paraformer.runtime.python.utils.logger import logger


class CttPunctuator:
    def __init__(self, online: bool = False):
        """
        punctuator with singleton pattern
        :param online:
        """
        self.online = online

        if online:
            logger.info("Initializing punctuator instance with online mode.")
            self.model = CT_Transformer()
            self.param_dict = {"cache": []}
            logger.info("Online punctuator instance initialized.")
        else:
            logger.info("Initializing punctuator instance with offline mode.")
            self.model = CT_Transformer()
            logger.info("Offline punctuator instance initialized.")

    def punctuate(self, text: str, param_dict=None):
        if self.online:
            param_dict = param_dict or self.param_dict
            return self.model.online(text, param_dict)
        else:
            return self.model.offline(text)
