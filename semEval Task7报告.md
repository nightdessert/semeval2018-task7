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
###semEval Task7
随着信息爆炸性的增长，科研工作者面临的负担也随着与日俱增。将自然语言处漏技术应用在科技文献中，能大大的减小科研工作者的工作量。其中最主要用于处理科技文献的自然语言处理任务是信息抽取任务：概念与实体识别并判断它们的关系。当前的任务将语义关系提取和分类分为6类，所有这些类别都是科学文献的特定。

semEval Task7包括关系分类和关系抽取两个子任务，其中本次作业的主要任务是关系分类，即给出一段科技文献和文中的实体对，对他们进行关系分类，任务包含两个子数据集，分别是干净良好的数据集和噪音较大的数据。

>Gábor, Kata, et al. "Semeval-2018 Task 7: Semantic relation extraction and classification in scientific papers." Proceedings of The 12th International Workshop on Semantic Evaluation. 2018.

### SVM
在机器学习中，支持向量机（英语：support vector machine，常简称为SVM，又名支持向量网络[1]）是在分类与回归分析中分析数据的监督式学习模型与相关的学习算法。给定一组训练实例，每个训练实例被标记为属于两个类别中的一个或另一个，SVM训练算法创建一个将新的实例分配给两个类别之一的模型，使其成为非概率二元线性分类器。SVM模型是将实例表示为空间中的点，这样映射就使得单独类别的实例被尽可能宽的明显的间隔分开。然后，将新的实例映射到同一空间，并基于它们落在间隔的哪一侧来预测所属类别.除了进行线性分类之外，SVM还可以使用所谓的核技巧有效地进行非线性分类，将其输入隐式映射到高维特征空间中。


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
