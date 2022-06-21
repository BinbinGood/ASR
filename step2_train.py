import argparse
import functools

from masr.trainer import MASRTrainer
from masr.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('batch_size',       int,    16,                       '训练的批量大小')
add_arg('local_rank',       int,    -1,                       '多卡训练的节点')
add_arg('num_workers',      int,    6,                        '读取数据的线程数量')
add_arg('num_epoch',        int,    30,                       '训练的轮数')
add_arg('learning_rate',    int,    5e-5,                     '初始学习率的大小')
add_arg('min_duration',     int,    0.5,                      '过滤最短的音频长度')
add_arg('max_duration',     int,    -1,                       '过滤最长的音频长度，当为-1的时候不限制长度')
add_arg('use_model',        str,    'deepspeech2',              '所使用的模型', choices=['deepspeech2', 'deepspeech2_big'])
add_arg('train_manifest',   str,    '/home/featurize/dataset/manifest.train',   '训练数据的数据列表路径')
add_arg('test_manifest',    str,    '/home/featurize/dataset/manifest.test',    '测试数据的数据列表路径')
add_arg('dataset_vocab',    str,    '/home/featurize/dataset/vocabulary.txt',   '数据字典的路径')
add_arg('mean_std_path',    str,    '/home/featurize/dataset/mean_std.npz',     '数据集的均值和标准值的npy文件路径')
add_arg('save_model_path',  str,    './models/',                  '模型保存的路径')
add_arg('metrics_type',     str,    'cer',                      '计算错误率方法', choices=['cer', 'wer'])
add_arg('feature_method',   str,    'linear',                   '音频预处理方法', choices=['linear', 'mfcc', 'fbank'])
add_arg('resume_model',     str,    None,                       '恢复训练，当为None则不使用预训练模型')
add_arg('pretrained_model', str,    None,                       '预训练模型的路径，当为None则不使用预训练模型')
args = parser.parse_args()
print_arguments(args)

trainer = MASRTrainer(use_model=args.use_model,
                      feature_method=args.feature_method,
                      mean_std_path=args.mean_std_path,
                      train_manifest=args.train_manifest,
                      test_manifest=args.test_manifest,
                      dataset_vocab=args.dataset_vocab,
                      num_workers=args.num_workers,
                      metrics_type=args.metrics_type)

trainer.train(batch_size=args.batch_size,
              min_duration=args.min_duration,
              max_duration=args.max_duration,
              num_epoch=args.num_epoch,
              learning_rate=args.learning_rate,
              save_model_path=args.save_model_path,
              resume_model=args.resume_model,
              pretrained_model=args.pretrained_model)
