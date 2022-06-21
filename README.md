# ASR
这是一个语音识别项目，模型为deepspeech2。pytorch实现

## 特定词汇语音识别项目

数据集采用了`THCHS-30`，通过本设计意在实现针对特定词汇的语音识别训练。

### ------------------------------项目的前期准备------------------------------

##### 确定网络结构

首先看几篇比较新的关于使用深度学习处理语音识别的综述。大致了解当前语音识别使用的主流架构。

Deep Speech2 这是百度在15年发布的一个语音识别端对端架构，主要使用了conv、LSTM和CTC。暂时考虑使用该模型处理该项目。考虑到THCHS-30数据集数据量太小，可能会使用预训练模型，在`TCHCHS-30`上做微调。

##### CTC结构connectionist temporal classification

了解CTC结构的数学原理，它是如何推导得到的。

##### THCHS-30数据集的预处理和加载

了解`THCHS-30`数据集的基本信息，了解如何使用该数据集，如何处理该数据集并将其转化为模型可以处理的格式。

### -----------------------项目代码整体逻辑结构------------------------------

#### 数据预处理

1. ##### 加载数据

   获取`THCHS-30`数据集，下载、解压。然后将标签和对应的音频文件路径保存在`thchs_30.txt`文件中。

2. ##### 划分数据

   读取上一步得到的`thchs_30.txt`文件，调整音频采样频率，读取音频获取音频的长度，过滤非法的字符，保证所有的标签都是简体中文，将数据划分为训练集和测试集，保存为json文件。每条数据包含`"audio_filepath", "duration", "text"`三个keys。

3. ##### 创建字典

   根据所有数据对应的标签，统计所有标签出现的词的词频，删除词频小于阈值的字，保存为字典

4. ##### 计算均值和标准差

   从训练数据的音频数据计算均值和标准差。

   读取音频-->音频归一化(将输入的音频归一化到`RMS均方根`为`-20db`)-->利用快速傅里叶变换计算线性语谱图-->然后根据声谱图计算均值和方差
   
 #### 训练模型

1. ##### 模型搭建

   该项目考虑使用`deepspeech2`模型。
   
1. ##### 数据加载

   ```python
   from torch.utils.data import Dataset
   
   class MASRDataset(Dataset):
       def __init__(self,):
           self.data_list  # 使用该变量保存音频路径和对应的标签
       def __getitem__(self,idx):
           """能够通过传入索引的方式获取数据"""
           audio_file, transcript = self.data_list[idx]
           1,音频处理：读取音频->音频归一化(将输入的音频归一化到`RMS均方根`为-20db)->利用快速傅里叶变换计算线性语谱图->利用预处理阶段得到均值和方差对音频特征的归一化
           2，标签处理：分词器分词->利用词典得到每个字对应的编号
           return feature, transcript
       def __len__(self):
           """能够实现通过全局的len()方法获取其中的元素个数"""
           return len(self.data_list)
   ```

   ```python
   from torch.utils.data import DataLoader
   train_dataset = MASRDataset(...)
   # 将一小段数据合并成数据列表，默认为False。若设置True，系统会在返回前会将张量数据Tensors复制到CUDA内存中。
   train_loader = DataLoader(dataset=train_dataset,collate_fn=collate_fn..)
   ```

   ```python
   # 函数输入为list，list中的元素为欲取出的一系列样本。
   def collate_fn(batch):
       """通过该函数，将所有音频特征和标签都转化为等长的tensor,短数据padding处理"""
   ```

2. ##### 训练模型

   ```python
   # 设置优化器
   optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate, weight_decay=1e-6)
   # 动态调整学习率
   scheduler = StepLR(optimizer, step_size=1, gamma=0.93)
   # 获取损失函数
   ctc_loss = torch.nn.CTCLoss(reduction='none', zero_infinity=True)
   # 开始训练：加载输入和对应的标签，正向传播-计算CTC损失-反向传播-参数更新。打印信息，保存模型
   # 每个epoch结束，在测试集上验证，输入测试数据-正向传播-计算损失-利用解码器得到最终的输出语句-将得到的语句和对应的标签计算字错率/词错率。
   ```

   衡量句子的相似性使用`莱文斯坦距离`，其描述由一个字串转化成另一个字串最少的编辑操作次数，其中的操作包括插入、删除和替换。

3. ##### 模型评估

   加载测试数据-->创建模型-->加载模型参数-->正向传播-->计算损失-->利用解码器得到最终的输出句子-->计算标签和预测句子的字错率。

4. ##### 模型预测

   首先，加载训练阶段保存的最优模型-->在基础模型前端加入数据正则化-->在模型后端加上softmax层，得到每个rnn节点输出每个字符对应的概率值-->将得到的最终预测模型和得到的字典保存。

   对于短语音识别-->加载模型和词汇表-->加载测试语音-->输入模型得到模型的输出-->将得到的输出解码得到最终的预测句子(解码使用贪婪法，每个字符都选择概率最大的那个字符)
