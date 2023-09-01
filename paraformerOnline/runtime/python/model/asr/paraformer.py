# -*- coding:utf-8 -*-
# @FileName  :paraformer.py
# @Time      :2023/8/8 20:04
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com
import glob
import os
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np

from paraformerOnline.runtime.python.utils.asrOrtInferRuntimeSession import (
    AsrOfflineOrtInferRuntimeSession, AsrOnlineOrtInferRuntimeSession,
    CharTokenizer, Hypothesis, TokenIDConverter, read_yaml)
from paraformerOnline.runtime.python.utils.audioHelper import AudioReader
from paraformerOnline.runtime.python.utils.postprocess import \
    sentence_postprocess
from paraformerOnline.runtime.python.utils.preprocess import (
    SinusoidalPositionEncoderOnline, WavFrontend, WavFrontendOnline)
from paraformerOnline.runtime.python.utils.singleton import singleton


@singleton
class ParaformerOnlineModel:
    def __init__(
        self,
        model_dir: Union[str, Path] = None,
        batch_size: int = 1,
        chunk_size: List = [5, 10, 5],
        device_id: Union[str, int] = "-1",
        quantize: bool = False,
        intra_op_num_threads: int = 4,
    ):
        if not Path(model_dir).exists():
            raise FileNotFoundError(f"{model_dir} is not exist")

        encoder_model_file = os.path.join(model_dir, "model.onnx")
        decoder_model_file = os.path.join(model_dir, "decoder.onnx")
        if quantize:
            encoder_model_file = glob.glob(
                os.path.join(model_dir, "model_quant_*.onnx")
            )
            decoder_model_file = os.path.join(model_dir, "decoder_quant.onnx")

        config_file = os.path.join(model_dir, "config.yaml")
        cmvn_file = os.path.join(model_dir, "am.mvn")
        config = read_yaml(config_file)

        self.converter = TokenIDConverter(config["token_list"])
        self.tokenizer = CharTokenizer()
        self.frontend = WavFrontendOnline(
            cmvn_file=cmvn_file, **config["frontend_conf"]
        )
        self.pe = SinusoidalPositionEncoderOnline()
        self.ort_encoder_infer = AsrOnlineOrtInferRuntimeSession(
            encoder_model_file, device_id, intra_op_num_threads=intra_op_num_threads
        )
        self.ort_decoder_infer = AsrOnlineOrtInferRuntimeSession(
            decoder_model_file, device_id, intra_op_num_threads=intra_op_num_threads
        )
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.encoder_output_size = config["encoder_conf"]["output_size"]
        self.fsmn_layer = config["decoder_conf"]["num_blocks"]
        self.fsmn_lorder = config["decoder_conf"]["kernel_size"] - 1
        self.fsmn_dims = config["encoder_conf"]["output_size"]
        self.feats_dims = (
            config["frontend_conf"]["n_mels"] * config["frontend_conf"]["lfr_m"]
        )
        self.cif_threshold = config["predictor_conf"]["threshold"]
        self.tail_threshold = config["predictor_conf"]["tail_threshold"]

    def prepare_cache(self, cache: dict = {}, batch_size=1):
        if len(cache) > 0:
            return cache
        cache["start_idx"] = 0
        cache["cif_hidden"] = np.zeros(
            (batch_size, 1, self.encoder_output_size)
        ).astype(np.float32)
        cache["cif_alphas"] = np.zeros((batch_size, 1)).astype(np.float32)
        cache["chunk_size"] = self.chunk_size
        cache["last_chunk"] = False
        cache["feats"] = np.zeros(
            (batch_size, self.chunk_size[0] + self.chunk_size[2], self.feats_dims)
        ).astype(np.float32)
        cache["decoder_fsmn"] = []
        for i in range(self.fsmn_layer):
            fsmn_cache = np.zeros(
                (batch_size, self.fsmn_dims, self.fsmn_lorder)
            ).astype(np.float32)
            cache["decoder_fsmn"].append(fsmn_cache)
        return cache

    def add_overlap_chunk(self, feats: np.ndarray, cache: dict = {}):
        if len(cache) == 0:
            return feats
        # process last chunk
        overlap_feats = np.concatenate((cache["feats"], feats), axis=1)
        if cache["is_final"]:
            cache["feats"] = overlap_feats[:, -self.chunk_size[0] :, :]
            if not cache["last_chunk"]:
                padding_length = sum(self.chunk_size) - overlap_feats.shape[1]
                overlap_feats = np.pad(
                    overlap_feats, ((0, 0), (0, padding_length), (0, 0))
                )
        else:
            cache["feats"] = overlap_feats[
                :, -(self.chunk_size[0] + self.chunk_size[2]) :, :
            ]
        return overlap_feats

    def __call__(self, audio_in: np.ndarray, **kwargs):
        waveforms = np.expand_dims(audio_in, axis=0)
        param_dict = kwargs.get("param_dict", dict())
        is_final = param_dict.get("is_final", False)
        cache = param_dict.get("cache", dict())
        asr_res = []

        if waveforms.shape[1] < 16 * 60 and is_final and len(cache) > 0:
            cache["last_chunk"] = True
            feats = cache["feats"]
            feats_len = np.array([feats.shape[1]]).astype(np.int32)
            asr_res = self.infer(feats, feats_len, cache)
            return asr_res

        feats, feats_len = self.extract_feat(waveforms, is_final)
        if feats.shape[1] != 0:
            feats *= self.encoder_output_size**0.5
            cache = self.prepare_cache(cache)
            cache["is_final"] = is_final

            # fbank -> position encoding -> overlap chunk
            feats = self.pe.forward(feats, cache["start_idx"])
            cache["start_idx"] += feats.shape[1]
            if is_final:
                if feats.shape[1] + self.chunk_size[2] <= self.chunk_size[1]:
                    cache["last_chunk"] = True
                    feats = self.add_overlap_chunk(feats, cache)
                else:
                    # first chunk
                    feats_chunk1 = self.add_overlap_chunk(
                        feats[:, : self.chunk_size[1], :], cache
                    )
                    feats_len = np.array([feats_chunk1.shape[1]]).astype(np.int32)
                    asr_res_chunk1 = self.infer(feats_chunk1, feats_len, cache)

                    # last chunk
                    cache["last_chunk"] = True
                    feats_chunk2 = self.add_overlap_chunk(
                        feats[
                            :,
                            -(
                                feats.shape[1] + self.chunk_size[2] - self.chunk_size[1]
                            ) :,
                            :,
                        ],
                        cache,
                    )
                    feats_len = np.array([feats_chunk2.shape[1]]).astype(np.int32)
                    asr_res_chunk2 = self.infer(feats_chunk2, feats_len, cache)

                    asr_res_chunk = asr_res_chunk1 + asr_res_chunk2
                    res = {}
                    for pred in asr_res_chunk:
                        for key, value in pred.items():
                            if key in res:
                                res[key][0] += value[0]
                                res[key][1].extend(value[1])
                            else:
                                res[key] = [value[0], value[1]]
                    return [res]
            else:
                feats = self.add_overlap_chunk(feats, cache)

            feats_len = np.array([feats.shape[1]]).astype(np.int32)
            asr_res = self.infer(feats, feats_len, cache)

        return asr_res

    def infer(self, feats: np.ndarray, feats_len: np.ndarray, cache):
        # encoder forward
        enc_input = [feats, feats_len]
        enc, enc_lens, cif_alphas = self.ort_encoder_infer(enc_input)

        # predictor forward
        acoustic_embeds, acoustic_embeds_len = self.cif_search(enc, cif_alphas, cache)

        # decoder forward
        asr_res = []
        if acoustic_embeds.shape[1] > 0:
            dec_input = [enc, enc_lens, acoustic_embeds, acoustic_embeds_len]
            dec_input.extend(cache["decoder_fsmn"])
            dec_output = self.ort_decoder_infer(dec_input)
            logits, _, cache["decoder_fsmn"] = (
                dec_output[0],
                dec_output[1],
                dec_output[2:],
            )
            cache["decoder_fsmn"] = [
                item[:, :, -self.fsmn_lorder :] for item in cache["decoder_fsmn"]
            ]

            preds = self.decode(logits, acoustic_embeds_len)
            for pred in preds:
                pred = sentence_postprocess(pred)
                asr_res.append({"preds": pred})

        return asr_res

    def load_data(self, wav_content: Union[str, np.ndarray, List[str]]) -> List:
        def load_wav(path: str) -> np.ndarray:
            waveform, _ = AudioReader.read_wav_file(path)
            return waveform

        if isinstance(wav_content, np.ndarray):
            return [wav_content]

        if isinstance(wav_content, str):
            return [load_wav(wav_content)]

        if isinstance(wav_content, list):
            return [load_wav(path) for path in wav_content]

        raise TypeError(f"The type of {wav_content} is not in [str, np.ndarray, list]")

    def extract_feat(
        self, waveforms: np.ndarray, is_final: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        waveforms_lens = np.zeros(waveforms.shape[0]).astype(np.int32)
        for idx, waveform in enumerate(waveforms):
            waveforms_lens[idx] = waveform.shape[-1]

        feats, feats_len = self.frontend.extract_fbank(
            waveforms, waveforms_lens, is_final
        )
        return feats.astype(np.float32), feats_len.astype(np.int32)

    def decode(self, am_scores: np.ndarray, token_nums: int) -> List[str]:
        return [
            self.decode_one(am_score, token_num)
            for am_score, token_num in zip(am_scores, token_nums)
        ]

    def decode_one(self, am_score: np.ndarray, valid_token_num: int) -> List[str]:
        yseq = am_score.argmax(axis=-1)
        score = am_score.max(axis=-1)
        score = np.sum(score, axis=-1)

        # pad with mask tokens to ensure compatibility with sos/eos tokens
        # asr_model.sos:1  asr_model.eos:2
        yseq = np.array([1] + yseq.tolist() + [2])
        hyp = Hypothesis(yseq=yseq, score=score)

        # remove sos/eos and get results
        last_pos = -1
        token_int = hyp.yseq[1:last_pos].tolist()

        # remove blank symbol id, which is assumed to be 0
        token_int = list(filter(lambda x: x not in (0, 2), token_int))

        # Change integer-ids to tokens
        token = self.converter.ids2tokens(token_int)
        token = token[:valid_token_num]
        # texts = sentence_postprocess(token)
        return token

    def cif_search(self, hidden, alphas, cache=None):
        batch_size, len_time, hidden_size = hidden.shape
        token_length = []
        list_fires = []
        list_frames = []
        cache_alphas = []
        cache_hiddens = []
        alphas[:, : self.chunk_size[0]] = 0.0
        alphas[:, sum(self.chunk_size[:2]) :] = 0.0
        if cache is not None and "cif_alphas" in cache and "cif_hidden" in cache:
            hidden = np.concatenate((cache["cif_hidden"], hidden), axis=1)
            alphas = np.concatenate((cache["cif_alphas"], alphas), axis=1)
        if cache is not None and "last_chunk" in cache and cache["last_chunk"]:
            tail_hidden = np.zeros((batch_size, 1, hidden_size)).astype(np.float32)
            tail_alphas = np.array([[self.tail_threshold]]).astype(np.float32)
            tail_alphas = np.tile(tail_alphas, (batch_size, 1))
            hidden = np.concatenate((hidden, tail_hidden), axis=1)
            alphas = np.concatenate((alphas, tail_alphas), axis=1)

        len_time = alphas.shape[1]
        for b in range(batch_size):
            integrate = 0.0
            frames = np.zeros(hidden_size).astype(np.float32)
            list_frame = []
            list_fire = []
            for t in range(len_time):
                alpha = alphas[b][t]
                if alpha + integrate < self.cif_threshold:
                    integrate += alpha
                    list_fire.append(integrate)
                    frames += alpha * hidden[b][t]
                else:
                    frames += (self.cif_threshold - integrate) * hidden[b][t]
                    list_frame.append(frames)
                    integrate += alpha
                    list_fire.append(integrate)
                    integrate -= self.cif_threshold
                    frames = integrate * hidden[b][t]

            cache_alphas.append(integrate)
            if integrate > 0.0:
                cache_hiddens.append(frames / integrate)
            else:
                cache_hiddens.append(frames)

            token_length.append(len(list_frame))
            list_fires.append(list_fire)
            list_frames.append(list_frame)

        max_token_len = max(token_length)
        list_ls = []
        for b in range(batch_size):
            pad_frames = np.zeros(
                (max_token_len - token_length[b], hidden_size)
            ).astype(np.float32)
            if token_length[b] == 0:
                list_ls.append(pad_frames)
            else:
                list_ls.append(np.concatenate((list_frames[b], pad_frames), axis=0))

        cache["cif_alphas"] = np.stack(cache_alphas, axis=0)
        cache["cif_alphas"] = np.expand_dims(cache["cif_alphas"], axis=0)
        cache["cif_hidden"] = np.stack(cache_hiddens, axis=0)
        cache["cif_hidden"] = np.expand_dims(cache["cif_hidden"], axis=0)

        return np.stack(list_ls, axis=0).astype(np.float32), np.stack(
            token_length, axis=0
        ).astype(np.int32)


@singleton
class ParaformerOfflineModel:
    def __init__(self, model_dir: str = None, intra_op_num_threads=4) -> None:
        config_path = os.path.join(model_dir, "config.yaml")

        config = read_yaml(config_path)

        self.converter = TokenIDConverter(config["token_list"])
        self.tokenizer = CharTokenizer(**config["CharTokenizer"])
        self.frontend = WavFrontend(
            cmvn_file=os.path.join(model_dir, "am.mvn"), **config["frontend_conf"]
        )
        if os.path.exists(os.path.join(model_dir, "model_quant.onnx")):
            model_file = os.path.join(model_dir, "model_quant.onnx")
        else:
            model_file = glob.glob(os.path.join(model_dir, "model_quant_*.onnx"))

        self.ort_infer = AsrOfflineOrtInferRuntimeSession(
            model_file, intra_op_num_threads=intra_op_num_threads
        )

    def extract_feat(self, waveforms: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        fbank, fbank_len = self.frontend.fbank(waveforms)
        feats, feats_len = self.frontend.lfr_cmvn(fbank)

        return feats.astype(np.float32), feats_len.astype(np.int32)

    async def transcribe(self, audio: Union[np.ndarray]):
        waveform = audio[None, ...]
        feats, feats_len = self.extract_feat(waveform)
        am_scores = self.ort_infer(input_content=[feats, feats_len])

        results = []
        for am_score in am_scores:
            pred_res = await self.infer_one_feat(am_score)
            results.append(pred_res)
        return results

    def infer(self, audio: Union[str, np.ndarray, bytes]):
        def infer_one_feat_without_async(am_score):
            yseq = am_score.argmax(axis=-1)
            score = am_score.max(axis=-1)
            score = np.sum(score, axis=-1)

            # pad with mask tokens to ensure compatibility with sos/eos tokens
            # asr_model.sos:1  asr_model.eos:2
            yseq = np.array([1] + yseq.tolist() + [2])
            hyp = Hypothesis(yseq=yseq, score=score)
            # remove sos/eos and get results
            last_pos = -1
            token_int = hyp.yseq[1:last_pos].tolist()

            # remove blank symbol id, which is assumed to be 0
            token_int = list(filter(lambda x: x not in (0, 2), token_int))

            # Change integer-ids to tokens
            token = self.converter.ids2tokens(token_int)
            texts = sentence_postprocess(token)
            return texts

        if isinstance(audio, str):
            audio, _ = AudioReader.read_wav_file(audio)
        elif isinstance(audio, bytes):
            audio, _ = AudioReader.read_wav_bytes(audio)

        feats, feats_len = self.extract_feat(audio)
        if feats_len > 0:
            am_scores = self.ort_infer(
                input_content=[feats[None, ...], feats_len[None, ...]]
            )
        else:
            am_scores = []

        results = []
        for am_score in am_scores:
            pred_res = infer_one_feat_without_async(am_score)
            results.append(pred_res)
        return results if len(results) != 0 else [[""]]
