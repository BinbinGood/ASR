import soundfile
import matplotlib.pyplot as plt
import numpy as np
from masr.data_utils.audio import AudioSegment
from masr.data_utils.featurizer.audio_featurizer import AudioFeaturizer


# 音频数据可视化
audio_path = "./dataset/test.wav"
audio_data, samplerate = soundfile.read(audio_path)
print("len of audio:", float(len(audio_data)) / samplerate)
# print(audio_data)
# print(samplerate)
plt.plot(np.arange(len(audio_data)), audio_data)
plt.show()


audio = AudioSegment.from_file(audio_path)
audio_featurizer = AudioFeaturizer(feature_method='linear')
feature = audio_featurizer.featurize(audio)
