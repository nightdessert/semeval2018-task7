# semEval Task7报告

#### 姓名：郭一诺 & 吴文浩

#### 学号：1801213682 &

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

### bert





---

## 模型与框架

### bert fine-tune

借鉴Bert Q&A任务的模型，



----

## 实验与结果

### 实验数据介绍

### 模型的训练与超参设置

### 实验结果分析与比较





