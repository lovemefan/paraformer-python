# -*- coding:utf-8 -*-
# @FileName  :logger.py
# @Time      :2023/8/8 20:17
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com

"""LOGGER Module"""
import logging
import logging.config
import logging.handlers
import os
import sys
from functools import wraps
from typing import List, Tuple, Union

logger_list = []
LEVEL = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
_LOG_FILE_DIR = "paraformer_logs/"
LOCAL_DEFAULT_LOG_FILE_DIR = os.path.join(
    os.getenv("LOCAL_DEFAULT_PATH", _LOG_FILE_DIR), "log"
)

DEFAULT_FILEHANDLER_FORMAT = (
    "[%(levelname)s] %(asctime)s " "[%(pathname)s:%(lineno)d] %(funcName)s: %(message)s"
)
DEFAULT_STDOUT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_REDIRECT_FILE_NAME = "paraformer-online.log"


class StreamRedirector:
    """Stream Re-director for Log."""

    def __init__(self, source_stream, target_stream):
        """Redirects the source stream to the target stream.

        Args:
            source_stream: Source stream.
            target_stream: Target stream.
        """
        super(StreamRedirector, self).__init__()

        self.source_stream = source_stream
        self.target_stream = target_stream

        self.save_source_stream_fd = os.dup(self.source_stream.fileno())

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.start()
            func(*args, **kwargs)
            self.stop()

        return wrapper

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        """start."""
        self.source_stream.flush()
        os.dup2(self.target_stream.fileno(), self.source_stream.fileno())

    def stop(self):
        """stop."""
        self.source_stream.flush()
        os.dup2(self.save_source_stream_fd, self.source_stream.fileno())
        self.target_stream.flush()


def validate_nodes_devices_input(var_name: str, var):
    """Check the list of nodes or devices.

    Args:
        var_name (str): Variable name.
        var: The name of the variable to be checked.

    Returns:
        None
    """
    if not (var is None or isinstance(var, (list, tuple, dict))):
        raise TypeError(
            "The value of {} can be None or a value of type tuple, "
            "list, or dict.".format(var_name)
        )
    if isinstance(var, (list, tuple)):
        for item in var:
            if not isinstance(item, int):
                raise TypeError(
                    "The elements of a variable of type list or "
                    "tuple must be of type int."
                )


def validate_level(var_name: str, var):
    """Verify that the log level is correct.

    Args:
        var_name (str): Variable name.
        var: The name of variable to be checked.

    Returns:
        None
    """
    if not isinstance(var, str):
        raise TypeError("The format of {} must be of type str.".format(var_name))
    if var not in LEVEL:
        raise ValueError("{}={} needs to be in {}".format(var_name, var, LEVEL))


def validate_std_input_format(
    to_std: bool,
    stdout_nodes: Union[List, Tuple, None],
    stdout_devices: Union[List, Tuple, None],
    stdout_level: str,
):
    """Validate the input about stdout of the get_logger function."""

    if not isinstance(to_std, bool):
        raise TypeError("The format of the to_std must be of type bool.")

    validate_nodes_devices_input("stdout_nodes", stdout_nodes)
    validate_nodes_devices_input("stdout_devices", stdout_devices)
    validate_level("stdout_level", stdout_level)


def validate_file_input_format(
    file_level: Union[List[str], Tuple[str]],
    file_save_dir: str,
    append_rank_dir: str,
    file_name: Union[List[str], Tuple[str]],
):
    """Validate the input about file of the get_logger function."""

    if not isinstance(file_level, (tuple, list)):
        raise TypeError("The value of file_level should be list or a tuple.")
    for level in file_level:
        validate_level("level in file_level", level)

    if not len(file_level) == len(file_name):
        raise ValueError("The length of file_level and file_name should be equal.")

    if not isinstance(file_save_dir, str):
        raise TypeError("The value of file_save_dir should be a value of type str.")

    if not isinstance(append_rank_dir, bool):
        raise TypeError("The value of append_rank_dir should be a value of type bool.")

    if not isinstance(file_name, (tuple, list)):
        raise TypeError("The value of file_name should be list or a tuple.")
    for name in file_name:
        if not isinstance(name, str):
            raise TypeError(
                "The value of name in file_name should be a value of type str."
            )


def _convert_level(level: str) -> int:
    """Convert the format of the log to logging level.

    Args:
        level (str): User log level.

    Returns:
        level (str): Logging level.
    """
    level_convert = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    level = level_convert.get(level, logging.INFO)

    return level


def get_logger(logger_name: str = "paraformer", **kwargs) -> logging.Logger:
    """Get the logger. Both computing centers and bare metal servers are
    available.

    Args:
        logger_name (str): Logger name.
        kwargs (dict): Other input.
            to_std (bool): If set to True, output the log to stdout.
            stdout_level (str): The level of the log output to stdout.
                If the type is str, the options are DEBUG, INFO, WARNING, ERROR, CRITICAL.
            stdout_format (str): Log format.
            file_level (list[str] or tuple[str]): The level of the log output to file.
                eg: ['INFO', 'ERROR'] Indicates that the logger will output logs above
                    the level INFO and ERROR in the list to the corresponding file.
                The length of the list needs to be the same as the length of file_name.
            file_save_dir (str): The folder where the log files are stored.
            append_rank_dir (bool): Whether to add a folder with the format rank{}.
            file_name (list[str] or list[tuple]): Store a list of output file names.
            max_file_size (int): The maximum size of a single log file. Unit: MB.
            max_num_of_files (int): The maximum number of files to save.

    Returns:
        logger (logging.Logger): Logger.
    """
    mf_logger = logging.getLogger(logger_name)
    if logger_name in logger_list:
        return mf_logger

    to_std = kwargs.get("to_std", True)
    stdout_nodes = kwargs.get("stdout_nodes", None)

    def get_stdout_devices():
        if os.getenv("STDOUT_DEVICES"):
            devices = os.getenv("STDOUT_DEVICES")
            if devices.startswith(("(", "[")) and devices.endswith((")", "]")):
                devices = devices[1:-1]
            devices = tuple(map(lambda x: int(x.strip()), devices.split(",")))
        else:
            devices = kwargs.get("stdout_devices", None)
        return devices

    stdout_devices = get_stdout_devices()
    stdout_level = kwargs.get("stdout_level", "INFO")
    stdout_format = kwargs.get("stdout_format", "")
    file_level = kwargs.get("file_level", ("INFO", "ERROR"))
    file_save_dir = kwargs.get("file_save_dir", "")
    append_rank_dir = kwargs.get("append_rank_dir", True)
    file_name = kwargs.get("file_name", ("info.log", "error.log"))
    max_file_size = kwargs.get("max_file_size", 50)
    max_num_of_files = kwargs.get("max_num_of_files", 5)

    validate_std_input_format(to_std, stdout_nodes, stdout_devices, stdout_level)
    validate_file_input_format(file_level, file_save_dir, append_rank_dir, file_name)

    if to_std:
        if not stdout_format:
            stdout_format = DEFAULT_STDOUT_FORMAT
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(_convert_level(stdout_level))
        stream_formatter = logging.Formatter(stdout_format)
        stream_handler.setFormatter(stream_formatter)
        mf_logger.addHandler(stream_handler)

    logging_level = []
    for level in file_level:
        logging_level.append(_convert_level(level))

    if not file_save_dir:
        file_save_dir = LOCAL_DEFAULT_LOG_FILE_DIR

    file_path = []
    for name in file_name:
        path = os.path.join(file_save_dir, name)
        path = os.path.realpath(path)
        base_dir = os.path.dirname(path)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir, exist_ok=True)
        file_path.append(path)

    max_file_size = max_file_size * 1024 * 1024

    file_formatter = logging.Formatter(DEFAULT_FILEHANDLER_FORMAT)
    for i, level in enumerate(file_level):
        file_handler = logging.handlers.RotatingFileHandler(
            filename=file_path[i], maxBytes=max_file_size, backupCount=max_num_of_files
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(file_formatter)
        mf_logger.addHandler(file_handler)

    mf_logger.setLevel(_convert_level("INFO"))

    mf_logger.propagate = False

    logger_list.append(logger_name)

    return mf_logger


class _LogActionOnce:
    """
    A wrapper for modify the warning logging to an empty function. This is used when we want to only log
    once to avoid the repeated logging.

    Args:
        logger (logging): The logger object.

    """

    is_logged = dict()

    def __init__(self, m_logger, key, no_warning=False):
        self.logger = m_logger
        self.key = key
        self.no_warning = no_warning

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            if not hasattr(self.logger, "warning"):
                return func(*args, **kwargs)

            old_func = self.logger.warning
            if self.no_warning or self.key in _LogActionOnce.is_logged:
                self.logger.warning = lambda x: x
            else:
                _LogActionOnce.is_logged[self.key] = True
            res = func(*args, **kwargs)
            if hasattr(self.logger, "warning"):
                self.logger.warning = old_func
            return res

        return wrapper


logger = get_logger(stdout_level="DEBUG")
