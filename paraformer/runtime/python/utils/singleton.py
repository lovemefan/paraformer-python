# -*- coding:utf-8 -*-
# @FileName  :singleton.py
# @Time      :2023/8/22 15:52
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com
import threading
from functools import wraps

from .logger import logger

lock = threading.RLock()
# instance container
instances = {}


def singleton(cls):
    """this is decorator to decorate class , make the class singleton(修饰器实现单例模式)"""

    @wraps(cls)
    def get_instance(*args, **kwargs):
        cls_name = cls.__name__

        if cls_name not in instances:
            with lock:
                if cls_name not in instances:
                    logger.info(f"creating {cls_name} instance")
                    instance = cls(*args, **kwargs)
                    instances[cls_name] = instance
                    logger.info(f"create {cls_name} instance finished")

        return instances[cls_name]

    return get_instance


def get_all_instance():
    """return all instance in the container"""
    return instances
