# -*- coding:utf-8 -*-
# @FileName  :ortruntimeSession.py
# @Time      :2023/8/8 20:20
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com

import logging
import re
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, NamedTuple, Set, Union

import jieba
import numpy as np
import yaml
from onnxruntime import (
    GraphOptimizationLevel,
    InferenceSession,
    SessionOptions,
    get_available_providers,
    get_device,
)

from paraformer.runtime.python.utils.singleton import singleton

root_dir = Path(__file__).resolve().parent


class TokenIDConverter:
    def __init__(
        self,
        token_list: Union[List, str],
    ):
        self.token_list = token_list
        self.unk_symbol = token_list[-1]
        self.token2id = {v: i for i, v in enumerate(self.token_list)}
        self.unk_id = self.token2id[self.unk_symbol]

    def get_num_vocabulary_size(self) -> int:
        return len(self.token_list)

    def ids2tokens(self, integers: Union[np.ndarray, Iterable[int]]) -> List[str]:
        if isinstance(integers, np.ndarray) and integers.ndim != 1:
            raise TokenIDConverterError(
                f"Must be 1 dim ndarray, but got {integers.ndim}"
            )
        return [self.token_list[i] for i in integers]

    def tokens2ids(self, tokens: Iterable[str]) -> List[int]:
        return [self.token2id.get(i, self.unk_id) for i in tokens]


class CharTokenizer:
    def __init__(
        self,
        symbol_value: Union[Path, str, Iterable[str]] = None,
        space_symbol: str = "<space>",
        remove_non_linguistic_symbols: bool = False,
    ):
        self.space_symbol = space_symbol
        self.non_linguistic_symbols = self.load_symbols(symbol_value)
        self.remove_non_linguistic_symbols = remove_non_linguistic_symbols

    @staticmethod
    def load_symbols(value: Union[Path, str, Iterable[str]] = None) -> Set:
        if value is None:
            return set()

        if isinstance(value, Iterable):
            return set(value)

        file_path = Path(value)
        if not file_path.exists():
            logging.warning("%s doesn't exist.", file_path)
            return set()

        with file_path.open("r", encoding="utf-8") as f:
            return set(line.rstrip() for line in f)

    def text2tokens(self, line: Union[str, list]) -> List[str]:
        tokens = []
        while len(line) != 0:
            for w in self.non_linguistic_symbols:
                if line.startswith(w):
                    if not self.remove_non_linguistic_symbols:
                        tokens.append(line[: len(w)])
                    line = line[len(w) :]
                    break
            else:
                t = line[0]
                if t == " ":
                    t = "<space>"
                tokens.append(t)
                line = line[1:]
        return tokens

    def tokens2text(self, tokens: Iterable[str]) -> str:
        tokens = [t if t != self.space_symbol else " " for t in tokens]
        return "".join(tokens)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f'space_symbol="{self.space_symbol}"'
            f'non_linguistic_symbols="{self.non_linguistic_symbols}"'
            f")"
        )


class Hypothesis(NamedTuple):
    """Hypothesis data type."""

    yseq: np.ndarray
    score: Union[float, np.ndarray] = 0
    scores: Dict[str, Union[float, np.ndarray]] = dict()
    states: Dict[str, Any] = dict()

    def asdict(self) -> dict:
        """Convert data to JSON-friendly dict."""
        return self._replace(
            yseq=self.yseq.tolist(),
            score=float(self.score),
            scores={k: float(v) for k, v in self.scores.items()},
        )._asdict()


class TokenIDConverterError(Exception):
    pass


class ONNXRuntimeError(Exception):
    pass


class AsrOnlineBaseOrtInferRuntimeSession:
    def __init__(self, model_file, device_id=-1, intra_op_num_threads=4):
        device_id = str(device_id)
        sess_opt = SessionOptions()
        sess_opt.intra_op_num_threads = intra_op_num_threads
        sess_opt.log_severity_level = 4
        sess_opt.enable_cpu_mem_arena = False
        sess_opt.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        cuda_ep = "CUDAExecutionProvider"
        cuda_provider_options = {
            "device_id": device_id,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "do_copy_in_default_stream": "true",
        }
        cpu_ep = "CPUExecutionProvider"
        cpu_provider_options = {
            "arena_extend_strategy": "kSameAsRequested",
        }

        EP_list = []
        if (
            device_id != "-1"
            and get_device() == "GPU"
            and cuda_ep in get_available_providers()
        ):
            EP_list = [(cuda_ep, cuda_provider_options)]
        EP_list.append((cpu_ep, cpu_provider_options))

        if isinstance(model_file, list):
            merged_model_file = b""
            for file in sorted(model_file):
                with open(file, "rb") as onnx_file:
                    merged_model_file += onnx_file.read()

            model_file = merged_model_file
        else:
            self._verify_model(model_file)
        self.session = InferenceSession(
            model_file, sess_options=sess_opt, providers=EP_list
        )

        # delete binary of model file to save memory
        del model_file

        if device_id != "-1" and cuda_ep not in self.session.get_providers():
            warnings.warn(
                f"{cuda_ep} is not avaiable for current env, the inference part is automatically shifted to be executed under {cpu_ep}.\n"
                "Please ensure the installed onnxruntime-gpu version matches your cuda and cudnn version, "
                "you can check their relations from the offical web site: "
                "https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html",
                RuntimeWarning,
            )

    def __call__(
        self, input_content: List[Union[np.ndarray, np.ndarray]]
    ) -> np.ndarray:
        input_dict = dict(zip(self.get_input_names(), input_content))
        try:
            result = self.session.run(self.get_output_names(), input_dict)
            return result
        except Exception as e:
            raise ONNXRuntimeError("ONNXRuntime inferece failed.") from e

    def get_input_names(
        self,
    ):
        return [v.name for v in self.session.get_inputs()]

    def get_output_names(
        self,
    ):
        return [v.name for v in self.session.get_outputs()]

    def get_character_list(self, key: str = "character"):
        return self.meta_dict[key].splitlines()

    def have_key(self, key: str = "character") -> bool:
        self.meta_dict = self.session.get_modelmeta().custom_metadata_map
        if key in self.meta_dict.keys():
            return True
        return False

    @staticmethod
    def _verify_model(model_path):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} does not exists.")
        if not model_path.is_file():
            raise FileExistsError(f"{model_path} is not a file.")


@singleton
class AsrOnlineEncoderOrtInferRuntimeSession(AsrOnlineBaseOrtInferRuntimeSession):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@singleton
class AsrOnlineDecoderOrtInferRuntimeSession(AsrOnlineBaseOrtInferRuntimeSession):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@singleton
class AsrOfflineOrtInferRuntimeSession:
    def __init__(
        self, model_file, contextual_model, device_id=-1, intra_op_num_threads=4
    ):
        sess_opt = SessionOptions()
        sess_opt.log_severity_level = 4
        sess_opt.intra_op_num_threads = intra_op_num_threads
        sess_opt.enable_cpu_mem_arena = False
        sess_opt.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        cuda_ep = "CUDAExecutionProvider"
        cuda_provider_options = {
            "device_id": device_id,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "do_copy_in_default_stream": "true",
        }
        cpu_ep = "CPUExecutionProvider"
        cpu_provider_options = {
            "arena_extend_strategy": "kSameAsRequested",
        }

        EP_list = []
        if (
            device_id != "-1"
            and get_device() == "GPU"
            and cuda_ep in get_available_providers()
        ):
            EP_list = [(cuda_ep, cuda_provider_options)]
        EP_list.append((cpu_ep, cpu_provider_options))

        if isinstance(model_file, list):
            merged_model_file = b""
            for file in sorted(model_file):
                with open(file, "rb") as onnx_file:
                    merged_model_file += onnx_file.read()

            model_file = merged_model_file
        else:
            self._verify_model(model_file)
        self.session = InferenceSession(
            model_file, sess_options=sess_opt, providers=EP_list
        )

        # delete binary of model file to save memory
        del model_file

        self.contextual_model = InferenceSession(
            contextual_model, sess_options=sess_opt, providers=EP_list
        )

        if device_id != "-1" and cuda_ep not in self.session.get_providers():
            logging.warning(
                f"{cuda_ep} is not avaiable for current env, the inference part is automatically shifted to be executed under {cpu_ep}.\n"
                "Please ensure the installed onnxruntime-gpu version matches your cuda and cudnn version, "
                "you can check their relations from the offical web site: "
                "https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html"
            )

    def __call__(
        self,
        feats: Union[np.ndarray],
        feats_length: Union[np.ndarray],
        bias_embed: np.ndarray = None,
    ) -> np.ndarray:
        """
        Args:
            feats: numpy.ndarray , [batch size , feats length, dim ] batch only support 1, dim is 560
            feats_length:  numpy.ndarray, [feats length]
            bias_embed: numpy.ndarray, [batch size, max string length, dim]
                batch only support 1, max string length is 10, dim is 512

        Returns:

        """

        input_dict = dict(
            zip(self.get_asr_input_names(), (feats, feats_length, bias_embed))
        )
        return self.session.run(None, input_dict)[0]

    def get_hot_words_embedding(self):
        pass

    def get_asr_input_names(
        self,
    ):
        return [v.name for v in self.session.get_inputs()]

    def get_contextual_model_input_names(
        self,
    ):
        return [v.name for v in self.contextual_model.get_inputs()]

    def get_output_names(
        self,
    ):
        return [v.name for v in self.session.get_outputs()]

    def get_character_list(self, key: str = "character"):
        return self.meta_dict[key].splitlines()

    def have_key(self, key: str = "character") -> bool:
        self.meta_dict = self.session.get_modelmeta().custom_metadata_map
        if key in self.meta_dict.keys():
            return True
        return False

    @staticmethod
    def _verify_model(model_path):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} does not exists.")
        if not model_path.is_file():
            raise FileExistsError(f"{model_path} is not a file.")

    def proc_hot_word(self, hot_words):
        hot_words_length = [len(i) - 1 for i in hot_words]
        hot_words_length.append(0)

        hot_words_length = np.array(hot_words_length)

        # hotwords.append('<s>')
        def word_map(word):
            return np.array([self.vocab[i] for i in word])

        hot_word_int = [word_map(i) for i in hot_words]
        hot_word_int.append(np.array([1]))
        n_batch = len(hot_word_int)

        hot_words = np.zeros((n_batch, 10, *hot_word_int[0].size()[1:]))

        for i in range(n_batch):
            hot_words[i, : hot_word_int[i].size(0)] = hot_word_int[i]

        return hot_words, hot_words_length


def split_to_mini_sentence(words: list, word_limit: int = 20):
    assert word_limit > 1
    if len(words) <= word_limit:
        return [words]
    sentences = []
    length = len(words)
    sentence_len = length // word_limit
    for i in range(sentence_len):
        sentences.append(words[i * word_limit : (i + 1) * word_limit])
    if length % word_limit > 0:
        sentences.append(words[sentence_len * word_limit :])
    return sentences


def code_mix_split_words(text: str):
    words = []
    segs = text.split()
    for seg in segs:
        # There is no space in seg.
        current_word = ""
        for c in seg:
            if len(c.encode()) == 1:
                # This is an ASCII char.
                current_word += c
            else:
                # This is a Chinese char.
                if len(current_word) > 0:
                    words.append(current_word)
                    current_word = ""
                words.append(c)
        if len(current_word) > 0:
            words.append(current_word)
    return words


def isEnglish(text: str):
    if re.search("^[a-zA-Z']+$", text):
        return True
    else:
        return False


def join_chinese_and_english(input_list):
    line = ""
    for token in input_list:
        if isEnglish(token):
            line = line + " " + token
        else:
            line = line + token

    line = line.strip()
    return line


def code_mix_split_words_jieba(seg_dict_file: str):
    jieba.load_userdict(seg_dict_file)

    def _fn(text: str):
        input_list = text.split()
        token_list_all = []
        langauge_list = []
        token_list_tmp = []
        language_flag = None
        for token in input_list:
            if isEnglish(token) and language_flag == "Chinese":
                token_list_all.append(token_list_tmp)
                langauge_list.append("Chinese")
                token_list_tmp = []
            elif not isEnglish(token) and language_flag == "English":
                token_list_all.append(token_list_tmp)
                langauge_list.append("English")
                token_list_tmp = []

            token_list_tmp.append(token)

            if isEnglish(token):
                language_flag = "English"
            else:
                language_flag = "Chinese"

        if token_list_tmp:
            token_list_all.append(token_list_tmp)
            langauge_list.append(language_flag)

        result_list = []
        for token_list_tmp, language_flag in zip(token_list_all, langauge_list):
            if language_flag == "English":
                result_list.extend(token_list_tmp)
            else:
                seg_list = jieba.cut(
                    join_chinese_and_english(token_list_tmp), HMM=False
                )
                result_list.extend(seg_list)

        return result_list

    return _fn


def read_yaml(yaml_path: Union[str, Path]) -> Dict:
    if not Path(yaml_path).exists():
        raise FileExistsError(f"The {yaml_path} does not exist.")

    with open(str(yaml_path), "rb") as f:
        data = yaml.load(f, Loader=yaml.Loader)
    return data
