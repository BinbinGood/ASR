import argparse
import functools
import time

from masr.trainer import MASRTrainer
from masr.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('batch_size',       int,    32,                       '评估的批量大小')
add_arg('min_duration',     int,    0.5,                      '过滤最短的音频长度')
add_arg('max_duration',     int,    -1,                       '过滤最长的音频长度，当为-1的时候不限制长度')
add_arg('num_workers',      int,    6,                        '读取数据的线程数量')
add_arg('use_model',        str,   'deepspeech2',             '所使用的模型', choices=['deepspeech2', 'deepspeech2_big'])
add_arg('test_manifest',    str,   '/home/featurize/dataset/manifest.test',   '测试数据的数据列表路径')
add_arg('dataset_vocab',    str,   '/home/featurize/dataset/vocabulary.txt',  '数据字典的路径')
add_arg('mean_std_path',    str,   '/home/featurize/dataset/mean_std.npz',    '数据集的均值和标准值的npy文件路径')
add_arg('metrics_type',     str,   'cer',                     '计算错误率方法', choices=['cer', 'wer'])
add_arg('feature_method',   str,   'linear',                  '音频预处理方法', choices=['linear', 'mfcc', 'fbank'])
add_arg('resume_model',     str,   'models/deepspeech2/best_model/', '模型的路径')
args = parser.parse_args()
print_arguments(args)


trainer = MASRTrainer(use_model=args.use_model,
                      feature_method=args.feature_method,
                      mean_std_path=args.mean_std_path,
                      test_manifest=args.test_manifest,
                      dataset_vocab=args.dataset_vocab,
                      num_workers=args.num_workers,
                      metrics_type=args.metrics_type)

start = time.time()
error_rate = trainer.evaluate(batch_size=args.batch_size,
                              min_duration=args.min_duration,
                              max_duration=args.max_duration,
                              resume_model=args.resume_model)
end = time.time()
print('评估消耗时间：{}s，{}：{:.5f}'.format(int(end - start), args.metrics_type, error_rate))
# 使用了SortaGrad方法:
# 100%|█████████████████████████████████████████████| 1/1 [00:00<00:00,  1.22it/s]
# 评估消耗时间：6s，cer：0.08374
# 未使用SortaGrad方法:
# 100%|█████████████████████████████████████████████| 1/1 [00:01<00:00,  1.02s/it]
# 评估消耗时间：8s，cer：0.07617
