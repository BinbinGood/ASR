import os
import sys

import numpy as np
import torch

from masr.data_utils.audio import AudioSegment
from masr.data_utils.featurizer.audio_featurizer import AudioFeaturizer
from masr.data_utils.featurizer.text_featurizer import TextFeaturizer
from masr.data_utils.ctc_greedy_decoder import greedy_decoder


class Predictor:
    def __init__(self,
                 model_path='models/deepspeech2/inference.pt',
                 vocab_path='dataset/vocabulary.txt',
                 use_model='deepspeech2',
                 feature_method='linear',
                 use_gpu=True):
        """
        语音识别预测工具
        :param model_path: 导出的预测模型文件夹路径
        :param vocab_path: 数据集的词汇表文件路径
        :param use_model: 所使用的模型
        :param feature_method: 所使用的预处理方法
        :param use_gpu: 是否使用GPU预测
        """
        self.use_model = use_model
        self.use_gpu = use_gpu
        self.lac = None
        self.last_audio_data = []
        self._text_featurizer = TextFeaturizer(vocab_filepath=vocab_path)
        self._audio_featurizer = AudioFeaturizer(feature_method=feature_method)

        # 创建模型
        if not os.path.exists(model_path):
            raise Exception("模型文件不存在，请检查{}是否存在！".format(model_path))
        # 根据 config 创建 predictor
        if self.use_gpu:
            self.predictor = torch.load(model_path)
            self.predictor.to('cuda')
        else:
            self.predictor = torch.load(model_path, map_location='cpu')
        self.predictor.eval()
        # 预热
        warmup_audio_path = 'dataset/test.wav'
        if os.path.exists(warmup_audio_path):
            self.predict(warmup_audio_path)
        else:
            print('预热文件不存在，忽略预热！', file=sys.stderr)

    # 解码模型输出结果
    def decode(self, output_data):
        # 执行解码
        result = greedy_decoder(probs_seq=output_data, vocabulary=self._text_featurizer.vocab_list)
        score, text = result[0], result[1]
        return score, text

    # 预测音频
    def predict(self,
                audio_path=None,
                audio_bytes=None,
                audio_ndarray=None):
        """
        预测函数，只预测完整的一句话。
        :param audio_path: 需要预测音频的路径
        :param audio_bytes: 需要预测的音频wave读取的字节流
        :param audio_ndarray: 需要预测的音频未预处理的numpy值
        :return: 识别的文本结果和解码的得分数
        """
        assert audio_path is not None or audio_bytes is not None or audio_ndarray is not None, \
            'audio_path，audio_bytes和audio_ndarray至少有一个不为None！'
        # 加载音频文件，并进行预处理
        if audio_path is not None:
            audio_data = AudioSegment.from_file(audio_path)
        elif audio_ndarray is not None:
            audio_data = AudioSegment.from_ndarray(audio_ndarray)
        else:
            audio_data = AudioSegment.from_wave_bytes(audio_bytes)
        audio_feature = self._audio_featurizer.featurize(audio_data)
        audio_data = np.array(audio_feature).astype('float32')[np.newaxis, :]
        audio_len = np.array([audio_data.shape[2]]).astype('int64')

        audio_data = torch.from_numpy(audio_data).float()
        audio_len = torch.from_numpy(audio_len)
        init_state_h_box = None
        init_state_c_box = None

        if self.use_gpu:
            audio_data = audio_data.cuda()

        # 运行predictor
        output_data, _, _ = self.predictor(audio_data, audio_len, init_state_h_box, init_state_c_box)
        output_data = output_data.cpu().detach().numpy()[0]

        # 解码
        score, text = self.decode(output_data=output_data)
        return score, text
