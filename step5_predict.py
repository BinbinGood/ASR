import argparse
import functools
import time
import wave

from masr.predict import Predictor
from masr.utils.audio_vad import crop_audio_vad
from masr.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('wav_path',        str,      './dataset/123.m4a', "预测音频的路径")
add_arg('is_long_audio',  bool,                     False, "是否为长语音")
add_arg('real_time_demo', bool,                     False, "是否使用实时语音识别演示")
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


# 实时识别模拟
def real_time_predict_demo():
    state_h, state_c = None, None
    result = []
    # 识别间隔时间
    interval_time = 1
    CHUNK = 16000 * interval_time
    all_data = []
    # 读取数据
    wf = wave.open(args.wav_path, 'rb')
    data = wf.readframes(CHUNK)
    # 播放
    while data != b'':
        all_data.append(data)
        start = time.time()
        score, text, state_h, state_c = predictor.predict_stream(audio_bytes=data,
                                                                 init_state_h_box=state_h, init_state_c_box=state_c)
        result.append(text)
        print("分段结果：消耗时间：%dms, 识别结果: %s, 得分: %d" % ((time.time() - start) * 1000, ''.join(result), score))
        data = wf.readframes(CHUNK)
    all_data = b''.join(all_data)
    start = time.time()
    score, text, _, _ = predictor.predict_stream(audio_bytes=all_data, is_end=True)
    print("整一句结果：消耗时间：%dms, 识别结果: %s, 得分: %d" % ((time.time() - start) * 1000, text, score))


if __name__ == "__main__":
    if args.real_time_demo:
        real_time_predict_demo()
    else:
        if args.is_long_audio:
            predict_long_audio()
        else:
            predict_audio()
