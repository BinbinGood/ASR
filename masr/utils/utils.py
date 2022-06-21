import distutils.util
import json
import os
import wave

import librosa
import numpy as np
import soundfile
from tqdm import tqdm
from zhconv import convert

from masr.data_utils.normalizer import FeatureNormalizer


def print_arguments(args):
    print("-----------  Configuration Arguments -----------")
    for arg, value in sorted(vars(args).items()):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")


def add_arguments(argname, type, default, help, argparser, **kwargs):
    type = distutils.util.strtobool if type == bool else type
    argparser.add_argument("--" + argname,
                           default=default,
                           type=type,
                           help=help + ' 默认: %(default)s.',
                           **kwargs)


def create_annotation_text(data_dir, annotation_path):
    if not os.path.exists(annotation_path):
        os.makedirs(annotation_path)
    print('Create THCHS-30 annotation text ...')
    f_a = open(os.path.join(annotation_path, 'thchs_30.txt'), 'w', encoding='utf-8')
    data_path = 'data'
    for file in os.listdir(os.path.join(data_dir, data_path)):
        if '.trn' in file:
            file = os.path.join(data_dir, data_path, file)
            with open(file, 'r', encoding='utf-8') as f:
                line = f.readline()
                line = ''.join(line.split())
            f_a.write(file[:-4] + '\t' + line + '\n')
    f_a.close()


# 加载数据，然后将每条数据的路径和对应的标签保存
def prepare_dataset(target_dir, annotation_path):
    """Download, unpack and create manifest file."""
    data_dir = os.path.join(target_dir, 'data_thchs30')
    if not os.path.exists(data_dir):
        print("Dataset THCHS-30 not ready!")
    create_annotation_text(data_dir, annotation_path)
    print("提取数据路径和标签完成！")


# 根据字的索引查询词典返回对应的句子
def labels_to_string(label, vocabulary, blank_index=0):
    labels = []
    for l in label:
        index_list = [index for index in l if index != blank_index and index != -1]
        labels.append((''.join([vocabulary[index] for index in index_list])).replace('<space>', ' '))
    return labels


# 使用模糊删除方式删除文件
def fuzzy_delete(dir, fuzzy_str):
    if os.path.exists(dir):
        for file in os.listdir(dir):
            if fuzzy_str in file:
                path = os.path.join(dir, file)
                os.remove(path)


# 创建数据列表
def create_manifest(annotation_path, train_manifest_path, test_manifest_path, is_change_frame_rate=True,
                    max_test_manifest=10000):
    data_list = []
    test_list = []
    durations = []
    for annotation_text in os.listdir(annotation_path):
        if annotation_text == ".ipynb_checkpoints":
            continue
        annotation_text_path = os.path.join(annotation_path, annotation_text)
        with open(annotation_text_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in tqdm(lines):
            audio_path = line.split('\t')[0]
            # 重新调整音频采样频率并保存
            if is_change_frame_rate:
                change_rate(audio_path)
            # 获取音频长度
            audio_data, samplerate = soundfile.read(audio_path)
            duration = float(len(audio_data)) / samplerate
            durations.append(duration)
            # 过滤非法的字符
            text = is_ustr(line.split('\t')[1].replace('\n', '').replace('\r', '')).lower()
            if len(text) == 0: continue
            # 保证全部都是简体
            text = convert(text, 'zh-cn')
            # 加入数据列表中
            line = '{"audio_filepath":"%s", "duration":%.2f, "text":"%s"}' % (
                audio_path.replace('\\', '/'), duration, text)
            if annotation_text == 'test.txt':
                test_list.append(line)
            else:
                data_list.append(line)

    # 按照音频长度升序
    data_list.sort(key=lambda x: json.loads(x)["duration"], reverse=False)
    if len(test_list) > 0:
        test_list.sort(key=lambda x: json.loads(x)["duration"], reverse=False)
    # 数据写入到文件中
    f_train = open(train_manifest_path, 'w', encoding='utf-8')
    f_test = open(test_manifest_path, 'w', encoding='utf-8')
    for line in test_list:
        f_test.write(line + '\n')
    interval = 500
    if len(data_list) / 500 > max_test_manifest:
        interval = len(data_list) // max_test_manifest
    for i, line in enumerate(data_list):
        if i % interval == 0:
            if len(test_list) == 0:
                f_test.write(line + '\n')
            else:
                f_train.write(line + '\n')
        else:
            f_train.write(line + '\n')
    f_train.close()
    f_test.close()
    print("完成生成数据列表，数据集总长度为{:.2f}小时！".format(sum(durations) / 3600.))


# 改变音频采样率为16000Hz
def change_rate(audio_path):
    data, sr = soundfile.read(audio_path)
    if sr != 16000:
        data = librosa.resample(data, sr, target_sr=16000)
        soundfile.write(audio_path, data, samplerate=16000)


# 过滤非法的字符
def is_ustr(in_str):
    out_str = ''
    for i in range(len(in_str)):
        if is_uchar(in_str[i]):
            out_str = out_str + in_str[i]
    return out_str


# 判断是否为中文字符或者英文字符
def is_uchar(uchar):
    if uchar == ' ': return True
    if u'\u4e00' <= uchar <= u'\u9fa5':
        return True
    if u'\u0030' <= uchar <= u'\u0039':
        return False
    if (u'\u0041' <= uchar <= u'\u005a') or (u'\u0061' <= uchar <= u'\u007a'):
        return True
    if uchar in ('-', ',', '.', '>', '?'):
        return False
    return False


# 获取全部字符
def count_manifest(counter, manifest_path):
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            line = json.loads(line)
            for char in line["text"].replace('\n', ''):
                counter.update(char)
    if os.path.exists(manifest_path.replace('train', 'test')):
        with open(manifest_path.replace('train', 'test'), 'r', encoding='utf-8') as f:
            for line in tqdm(f.readlines()):
                line = json.loads(line)
                for char in line["text"].replace('\n', ''):
                    counter.update(char)


# 计算数据集的均值和标准差
def compute_mean_std(feature_method, manifest_path, output_path, num_samples=-1, num_workers=8):
    normalizer = FeatureNormalizer(feature_method=feature_method,
                                   mean_std_filepath=None,
                                   manifest_path=manifest_path,
                                   num_samples=num_samples,
                                   num_workers=num_workers)
    # 将计算的结果保存的文件中
    normalizer.write_to_file(output_path)
    print('计算的均值和标准差已保存在 %s！' % output_path)
