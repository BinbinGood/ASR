import argparse
import functools
import os

from masr.trainer import MASRTrainer
from masr.utils.utils import add_arguments, print_arguments, prepare_dataset

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("target_dir", str, "/home/featurize/data/", "存放音频文件的目录 (默认: %(default)s)")
add_arg('annotation_path', str, '/home/featurize/dataset/annotation/', '标注文件的路径')
add_arg('train_manifest', str, '/home/featurize/dataset/manifest.train', '训练数据的数据列表路径')
add_arg('test_manifest', str, '/home/featurize/dataset/manifest.test', '测试数据的数据列表路径')
add_arg('is_change_frame_rate', bool, True, '是否统一改变音频为16000Hz，这会消耗大量的时间')
add_arg('max_test_manifest', int, 10000, '生成测试数据列表的最大数量，如果annotation_path包含了test.txt，就全部使用test.txt的数据')
add_arg('count_threshold', int, 2, '字符计数的截断阈值，0为不做限制')
add_arg('dataset_vocab', str, '/home/featurize/dataset/vocabulary.txt', '生成的数据字典文件')
add_arg('num_workers', int, 8, '读取数据的线程数量')
add_arg('num_samples', int, -1, '用于计算均值和标准值得音频数量，当为-1使用全部数据')
add_arg('mean_std_path', str, '/home/featurize/dataset/mean_std.npz', '保存均值和标准值得numpy文件路径，后缀 (.npz).')
add_arg('feature_method', str, 'linear', '音频预处理方法', choices=['linear', 'mfcc', 'fbank'])
args = parser.parse_args()
print_arguments(args)

# 首先读取数据集的标签文件，得到每一条语音和对应的标签
if args.target_dir.startswith('~'):
    args.target_dir = os.path.expanduser(args.target_dir)

prepare_dataset(target_dir=args.target_dir,
                annotation_path=args.annotation_path)

# 得到所有数据的标签文件后，对数据进行划分、创建字典、计算均值和标准值
trainer = MASRTrainer(mean_std_path=args.mean_std_path,
                      feature_method=args.feature_method,
                      train_manifest=args.train_manifest,
                      test_manifest=args.test_manifest,
                      dataset_vocab=args.dataset_vocab,
                      num_workers=args.num_workers)

trainer.create_data(annotation_path=args.annotation_path,
                    num_samples=args.num_samples,
                    count_threshold=args.count_threshold,
                    is_change_frame_rate=args.is_change_frame_rate,
                    max_test_manifest=args.max_test_manifest)
