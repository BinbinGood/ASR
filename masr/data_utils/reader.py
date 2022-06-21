import json
import sys
from datetime import datetime

import numpy as np
from torch.utils.data import Dataset

from masr.data_utils.featurizer.speech_featurizer import SpeechFeaturizer
from masr.data_utils.normalizer import FeatureNormalizer
from masr.data_utils.speech import SpeechSegment


# 音频数据加载器
class MASRDataset(Dataset):
    def __init__(self, data_list, vocab_filepath, mean_std_filepath, feature_method='linear',
                 min_duration=0, max_duration=20):
        super(MASRDataset, self).__init__()
        self._normalizer = FeatureNormalizer(mean_std_filepath, feature_method=feature_method)
        self._speech_featurizer = SpeechFeaturizer(vocab_filepath=vocab_filepath, feature_method=feature_method)
        # 获取数据列表
        with open(data_list, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.data_list = []
        for line in lines:
            line = json.loads(line)
            # 跳过超出长度限制的音频
            if line["duration"] < min_duration:
                continue
            if max_duration != -1 and line["duration"] > max_duration:
                continue
            self.data_list.append([line["audio_filepath"], line["text"]])

    def __getitem__(self, idx):
        try:
            # 音频路径--标签
            audio_file, transcript = self.data_list[idx]
            speech_segment = SpeechSegment.from_file(audio_file, transcript)
            feature, transcript = self._speech_featurizer.featurize(speech_segment)
            feature = self._normalizer.apply(feature)
            transcript = np.array(transcript, dtype='int32')
            return feature, transcript
        except Exception as ex:
            print("[{}] 数据: {} 出错，错误信息: {}".format(datetime.now(), self.data_list[idx], ex), file=sys.stderr)
            rnd_idx = np.random.randint(self.__len__())
            return self.__getitem__(rnd_idx)

    def __len__(self):
        return len(self.data_list)

    @property
    def feature_dim(self):
        """返回词汇表大小

        :return: 词汇表大小
        :rtype: int
        """
        return self._speech_featurizer.feature_dim

    @property
    def vocab_size(self):
        """返回词汇表大小

        :return: 词汇表大小
        :rtype: int
        """
        return self._speech_featurizer.vocab_size

    @property
    def vocab_list(self):
        """返回词汇表列表

        :return: 词汇表列表
        :rtype: list
        """
        return self._speech_featurizer.vocab_list
