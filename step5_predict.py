import argparse
import functools
import time
import wave

from masr.predict import Predictor
from masr.utils.audio_vad import crop_audio_vad
from masr.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('wav_path',        str,      './dataset/test_long.wav', "预测音频的路径")
add_arg('is_long_audio',  bool,                     False, "是否为长语音")
add_arg('use_gpu',        bool,                     False, "是否使用GPU预测")
add_arg('use_model',       str,             'deepspeech2', "所使用的模型",  choices=['deepspeech2', 'deepspeech2_big'])
add_arg('vocab_path',      str,             './models/vocabulary.txt', "数据集的词汇表文件路径")
add_arg('model_path',      str,   './models/deepspeech2_inference.pt', "导出的预测模型文件路径")
add_arg('feature_method',  str,                  'linear', "音频预处理方法", choices=['linear', 'mfcc', 'fbank'])
args = parser.parse_args()
print_arguments(args)

# 获取识别器
predictor = Predictor(model_path=args.model_path, vocab_path=args.vocab_path, use_model=args.use_model,
                      use_gpu=args.use_gpu, feature_method=args.feature_method)


# 长语音识别
def predict_long_audio():
    start = time.time()
    # 分割长音频
    audios_bytes = crop_audio_vad(args.wav_path)
    texts = ''
    scores = []
    # 执行识别
    for i, audio_bytes in enumerate(audios_bytes):
        score, text = predictor.predict(audio_bytes=audio_bytes)
        texts = texts + '，' + text
        scores.append(score)
        print("第%d个分割音频, 得分: %d, 识别结果: %s" % (i, score, text))
    print("最终结果，消耗时间：%d, 得分: %d, 识别结果: %s" % (round((time.time() - start) * 1000), sum(scores) / len(scores), texts))


# 短语音识别
def predict_audio():
    start = time.time()
    score, text = predictor.predict(audio_path=args.wav_path)
    print("消耗时间：%dms, 识别结果: %s, 得分: %d" % (round((time.time() - start) * 1000), text, score))


if __name__ == "__main__":
    if args.is_long_audio:
        predict_long_audio()
    else:
        predict_audio()
