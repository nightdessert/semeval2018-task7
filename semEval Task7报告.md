# semEval Task7报告

#### 姓名：郭一诺 & 吴文浩

#### 学号：1801213682 & 1801213710

#### 代码github地址：<https://github.com/nightdessert/semeval2018-task7>

----

## 引言

关系分类任务目的是预测给定名词对之间的语义关系，可以被形式化的定义为：给定一个包含名词对e1和e2的句子S，我们的目标是识别e1和e2之间的语义关系。该任务吸引了很多关注,人们对其感兴趣，既因为该目的本身，又因为其可作为许多自然语言处理任务的中间步骤的应用。

对于关系分类任务而言，最具代表性的方法是有监督的方法，这类方法对该任务非常有效并且可以看到在该任务上有很好的表现。

> Dmitry Zelenko, Chinatsu Aone, and Anthony Richardella. 2003. Kernel methods for relation extraction. The
> Journal of Machine Learning Research, 3:1083–1106.
>
> Raymond J Mooney and Razvan C Bunescu. 2005. Subsequence kernels for relation extraction. In Advances in
> neural information processing systems, pages 171–178.
>
> GuoDong Zhou, Su Jian, Zhang Jie, and Zhang Min. 2005. Exploring various knowledge in relation extraction.
> In Proceedings of the 43rd Annual Meeting on Association for Computational Linguistics, pages 427–434.

有监督的方法可以进一步划分为基于特征的方法和基于内核的方法。基于特征的方法把文本分析提取到的一系列特征转化为特征向量以此作为文本的表示。相反，基于内核的方法需要以解玺书的形式（例如依存关系树）。这些方法很高效因为他们使用了大量的语言知识。尽管如此，提取的特征或精心设计的内核通常来自预先存在的自然语言系统的输出，这导致错误在现有工具中传播并阻碍系统的性能。因此，寻找独立现有工具的方法进行特征提取更加有吸引力以及调整性。

> Nguyen Bach and Sameer Badaskar. 2007. A review of relation extraction. Literature review for Language and
> Statistics II。

在本次任务中，我们实现了两种方法来进行实体关系的分类，一种是传统的人工设计特征去提取文本的信息，对于实体而言，包括一些TF-IDF特征, 单词长度，词语位置，词向量表示等特征，进一步使用svm对输入的特征进行学习并分类。

另一种方法，我们采取了端到端的方法完成对文本特征的学习以及关系的分类，2018年谷歌提出的BERT预训练模型，横扫多项NLP任务。对于关系分类任务，我们同样尝试使用在BERT上fine-tune, 取得了不错的效果，可以看到NLP预处理模型的强大。

> Devlin J, Chang M W, Lee K, et al. Bert: Pre-training of deep bidirectional transformers for language understanding[J].

-----

## 相关工作

## semEval Task7

随着信息爆炸性的增长，科研工作者面临的负担也随着与日俱增。将自然语言处漏技术应用在科技文献中，能大大的减小科研工作者的工作量。其中最主要用于处理科技文献的自然语言处理任务是信息抽取任务：概念与实体识别并判断它们的关系。当前的任务将语义关系提取和分类分为6类，所有这些类别都是科学文献的特定。

semEval Task7包括关系分类和关系抽取两个子任务，其中本次作业的主要任务是关系分类，即给出一段科技文献和文中的实体对，对他们进行关系分类，任务包含两个子数据集，分别是干净良好的数据集和噪音较大的数据。

>Gábor, Kata, et al. "Semeval-2018 Task 7: Semantic relation extraction and classification in scientific papers." Proceedings of The 12th International Workshop on Semantic Evaluation. 2018.

### SVM
在机器学习中，支持向量机（英语：support vector machine，常简称为SVM，又名支持向量网络[1]）是在分类与回归分析中分析数据的监督式学习模型与相关的学习算法。给定一组训练实例，每个训练实例被标记为属于两个类别中的一个或另一个，SVM训练算法创建一个将新的实例分配给两个类别之一的模型，使其成为非概率二元线性分类器。SVM模型是将实例表示为空间中的点，这样映射就使得单独类别的实例被尽可能宽的明显的间隔分开。然后，将新的实例映射到同一空间，并基于它们落在间隔的哪一侧来预测所属类别.除了进行线性分类之外，SVM还可以使用所谓的核技巧有效地进行非线性分类，将其输入隐式映射到高维特征空间中。

使用SVM 对句子/文本的特征进行分类能取得不错的效果，因而常常被用于 NLP 中的文本分类等任务。

> Cortes, C.; Vapnik, V. Support-vector networks. Machine Learning. 1995, 20 (3): 273–297. doi:10.1007/BF00994018
>
>Ben-Hur, Asa, Horn, David, Siegelmann, Hava, and Vapnik, Vladimir; "Support vector clustering" (2001) Journal of Machine Learning Research

### Bert

Bert是 google 提出的基于上下文的语言模型，它将双向Transformer模型用于两个预训练任务，分别是Masked LM和Next Sentence Prediction,使用海量语料训练后得到了强大语言模型。本次使用的 bert 模型是google 公开的在大型语料上训练过的12层的基础模型。

![preview](https://pic1.zhimg.com/v2-d942b566bde7c44704b7d03a1b596c0c_r.jpg)

> Devlin, Jacob, et al. "Bert: Pre-training of deep bidirectional transformers for language understanding." *arXiv preprint arXiv:1810.04805* (2018).

Bert的Em'bedding由以下三种Em'bedding求和而成

![preview](https://pic2.zhimg.com/v2-11505b394299037e999d12997e9d1789_r.jpg)

google之后将bert应用到多个 NLP 任务上，都取得了突破性的成果,以下是对应于不同的子任务BERT的输入以及fine-tune的模型，本实验模仿SQuAD的形式，进行关系分类的预测

![preview](https://pic2.zhimg.com/v2-b054e303cdafa0ce41ad761d5d0314e1_r.jpg)



---

## 模型与框架

### SVM

使用 Svm 进行关系分类主要包括两点: 1.输入特征选取 2.Svm 内核选取。

特征选取：相比于深度学习自动学习特征的特性，传统的 Svm 等分类器更加依赖人工提取的数据特征，在本关系分类任务中，我们对比了几种特征向量选取对最终效果的影响。选取的特征包括: 实体在语料中的 TF-IDF, 实体的词长， 词语在文中的位置，glove 词向量。

内核选取：Svm 是一种基于核方法的分类器，可以使用所谓的核技巧有效地进行非线性分类，将其输入隐式映射到高维特征空间中。我们对比实验了非线性内核 RBF 和线性内核的效果。

### bert fine-tune

在bert上叠加一层线性层，对输出的特征做softmax。

输入是当前两个实体对所在的摘要，以此上下文作为第一句话输入，两个实体中间用 **[BREAK]** 隔开作为第二句话。

即输入为:

**[CLS] 摘要原文 [SEP] 实体一 [BREAK] 实体二 [CLS]**

be'r't的输出后接一个线性分类层，预测当前实体的关系

----

## 实验与结果

### 实验数据介绍

关系分类任务一共包括六种离散的类别，这些关系特定于科学领域，他们的示例在科学论文的摘要和介绍中出现。任务提供完整的科学论文摘要，每篇摘要大约包含100个词左右。训练数据和测试数据中均标注了实体以及实体之间的关系方向。具体的数据形式是：给定摘要中的一对实体，任务包括对它们之间的语义关系进行分类，一个预定义好的关系目录已给定，如下表所示：

| 关系种类   | 示例                                                         |
| ---------- | ------------------------------------------------------------ |
| USAGE      | approach – model / approach – parsing / MT system – Japanese / parse – sentence |
| RESULT     | order – performance / ambiguity – sentence / parser – performance |
| MODEL      | categories – words / interpretation – utterance / categories – words |
| PART_WHOLE | ontology – concepts / knowledge – domain /  expressions – text |
| TOPIC      | paper – method / research – speech                           |
| COMPARISON | result – standard                                            |

具体任务分为两个子任务

1. 无噪声数据上的关系分类

   训练数据和测试数据中均为手工标注的实体，训练数据中，实体之间手工标注出语义关系，测试数据中，仅给出实体注释和未标记关系的示例，任务是预测出实体之间的语义关系，如下所示是测试集中的一个示例：

   ```
   Korean, a <entity id=”H01-1041.10”>verb final language</entity>with <entity id=”H01-1041.11”>overt case markers</entity>(...)
   ```

   一个关系示例使用唯一的标号被定义为**(H01-1041.10, H01-1041.11)**.需要预测的即为关系类别标签例如：

    **MODEL-FEATURE(H01-1041.10, H01-1041.11)**.

   

2. 有噪声数据上的关系分类

   训练数据和测试数据中均为自动标注的实体，实体中可能会出现定界错误的实例，训练数据中，在测试数据中，仅给出自动实体注释和未标记关系的示例，任务是预测出实体之间的语义关系，如下所示是测试集中的一个示例

   ```
   This <entity id=”L08-1203.8”> paper </entity> introduces a new <entity id=”L08-
   1203.9”>architecture</entity>(...)
   ```

   关系示例表示为**(L08-1203.8, L08-1203.9).**，需要预测的即为关系类别标签例如：**TOPIC(L08-1203.8, L08-1203.9)**

### 模型的训练与超参设置

#### SVM

主要使用scikit-learn机器学习软件包进行特征提取和使用其中的 SVM 分类器，另外使用的NLTK 软件包进行了数据预处理。

选取的特征包括: 实体在语料中的 TF-IDF, 实体的词长， 词语在文中的位置，glove 词向量, 反转(reverse)标记。其中 ,实体的 TF-IDF 是实体中单词的 TF-IDF 值，若实体长度大于1，则取其 TF-IDF平均值, 同时也实验了替换命名实体计算 tf-idf 的方法。glove 词向量对比了50维和300维两种长度的效果。

内核选取: 非线性内核 RBF 和线性内核。

#### bert

| 参数名            | 值                |
| ----------------- | ----------------- |
| model             | bert-base-uncased |
| train_batch_size  | 256               |
| max_seq_length    | 384               |
| learning_rate     | 5e-5              |
| warmup_proportion | 0.1               |

### 实验结果分析与比较

#### SVM

下面对比不同特征在非线性核下的表现,其中两个实体的单个实体特征最终拼接成向量,所有的特征向量后都拼接：

**subtask1.1:**

| 特征                                 | Macro F1 | Micro F1 |
| :----------------------------------- | :------: | :------: |
| tf-idf+reverse                       |   0.20   |   0.51   |
| tf-idf+单词长度+reverse              |   0.20   |   0.51   |
| tf-idf+单词长度+单词位置+reverse     |   0.20   |   0.51   |
| tf-idf(替换实体)+单词长度+reverse    |   0.12   |   0.44   |
| tf-idf+单词长度+glove(50维)+reverse  |   0.33   |   0.62   |
| tf-idf+单词长度+glove(300维)+reverse |   0.24   |   0.58   |
| bert-base-model+linear layer         |   0.45   |   0.65   |

**subtask1.2:**

| 特征                                 | Macro F1 | Micro F1 |
| :----------------------------------- | :------: | :------: |
| tf-idf+reverse                       |   0.29   |   0.52   |
| tf-idf+单词长度+reverse              |   0.29   |   0.52   |
| tf-idf+单词长度+单词位置+reverse     |   0.29   |   0.52   |
| tf-idf(替换实体)+单词长度+reverse    |   0.06   |   0.35   |
| tf-idf+单词长度+glove(50维)+reverse  |   0.66   |   0.78   |
| tf-idf+单词长度+glove(300维)+reverse |   0.55   |   0.67   |
| bert-base-model + linear layer       |   0.70   |   0.82   |

1.1,1.2两个数据展现了一些一致性：

1.无用特征：首先在使用的特征中，单词位置，单词长度这种在自动文摘等领域较为有用的特征在关系分类中并没有其作用。

2.特征替换:   特征替换因为只是科技名词的稀有性和特征性，整体计算TF-IDF 会产生反作用导致最终 F1值下降。

3.word-embedding: 大语料预训练得到的 word embedding 保留了词语的普适语义因而对最终的分类帮助很大, 而使用的 glove 维度过大的话会"喧宾夺主"，导致 F1值下降。

4.数据集问题:在1.1中的相同特征分类效果明显差于1.2, 查阅资料可知其他算法大都表现如此。

下面对比使用不同内核的分类，以1.1任务为例：

| 特征                                | 线性                        | RBF                         |
| ----------------------------------- | --------------------------- | --------------------------- |
| tf-idf+reverse                      | Macro F1:0.18               | Macro F1:0.20               |
| tf-idf+单词长度+glove(50维)+reverse | Macro F1:0.33 Micro F1:0.57 | Macro F1:0.33 Micro F1:0.62 |

从上可以基本看出非线性内核的数据拟合能力要强于线性。

Bert

可以看到bert的效果明显好于传统的基于特征的SVM等方法，Bert是在大规模语料学出来预训练模型，横扫了多项NLP任务的SOTA，Bert训练起来对机器的要求也比较高，因此这里的模型应该是未完全训练好的模型，对于这种训练数据只有几千条的小规模语料，如果直接训练一个多层的结构复杂的网络，可能会欠拟合，因此选择bert上继续fine-tune，会有比较好的效果。

