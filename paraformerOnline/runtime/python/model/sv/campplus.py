import os
from pathlib import Path
from typing import Union

import kaldi_native_fbank as knf
import numpy as np
import onnxruntime

from paraformerOnline.runtime.python.utils.audioHelper import AudioReader
from paraformerOnline.runtime.python.utils.singleton import singleton


@singleton
class Campplus:
    def __init__(self, onnx_path=None, threshold=0.5):
        """
        :param onnx_path: onnx model file path
        :param threshold: threshold of speaker embedding similarity
        """
        self.onnx = onnx_path or os.path.join(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                )
            ),
            "onnx/sv/campplus.onnx",
        )
        self.sess = onnxruntime.InferenceSession(self.onnx)
        self.output_name = [nd.name for nd in self.sess.get_outputs()]
        self.threshhold = threshold
        self.memory: np.ndarray = None

    def compute_cos_similarity(self, emb):
        assert len(emb.shape) == 2, "emb must be length * 80"
        cos_sim = emb.dot(self.memory.T) / (
            np.linalg.norm(emb) * np.linalg.norm(self.memory, axis=1)
        )
        cos_sim[np.isneginf(cos_sim)] = 0

        return 0.5 + 0.5 * cos_sim

    def register_speaker(self, emb: np.ndarray):
        """
        register speaker with embedding and name
        :param emb:
        :param name: speaker name
        :return:
        """
        assert len(emb.shape) == 2, "emb must be length * 80"
        self.memory = np.concatenate(
            (
                self.memory,
                emb,
            )
        )

    def extract_feature(self, audio: Union[str, Path, bytes], sample_rate=16000):
        if isinstance(audio, str) or isinstance(audio, Path):
            waveform, sample_rate = AudioReader.read_wav_file(audio)
        elif isinstance(audio, np.ndarray):
            waveform = audio
        opts = knf.FbankOptions()
        opts.frame_opts.samp_freq = float(sample_rate)
        opts.frame_opts.dither = 0.0
        opts.energy_floor = 1.0
        opts.mel_opts.num_bins = 80
        fbank_fn = knf.OnlineFbank(opts)
        fbank_fn.accept_waveform(sample_rate, waveform.tolist())
        frames = fbank_fn.num_frames_ready
        mat = np.empty([frames, opts.mel_opts.num_bins])
        for i in range(frames):
            mat[i, :] = fbank_fn.get_frame(i)
        feature = mat.astype(np.float32)

        feature = feature - feature.mean()
        feature = feature[None, ...]
        return feature

    def embedding(self, feature: np.ndarray):
        feed_dict = {"fbank": feature}
        output = self.sess.run(self.output_name, input_feed=feed_dict)
        return output

    def recognize(self, waveform: Union[str, Path, bytes], threshold=0.65):
        """
        auto register speaker with input waveform。
        input waveform, output speaker id , id in range 0,1,2,....,n
        :param waveform:
        :return index: if max similarity less than threshold, it will add current emb into memory
        """
        feature = self.extract_feature(waveform)
        emb = self.embedding(feature)[0]

        if self.memory is None:
            self.memory = emb / np.linalg.norm(emb)
            return 0
        sim = self.compute_cos_similarity(emb)[0]
        max_sim_index = np.argmax(sim)

        if sim[max_sim_index] <= threshold:
            self.register_speaker(emb)

        return max_sim_index
