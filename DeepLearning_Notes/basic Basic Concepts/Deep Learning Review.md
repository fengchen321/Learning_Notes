# Deep Learning Review

[Deep Learning](https://www.nature.com/articles/nature14539)(2015)

[toc]

## 引言

​      *Deep-learning methods are representation-learning methods with multiple levels of representa-tion, obtained by composing simple but non-linear modules that each transform the representation at one level (starting with the raw input) into a representation at a higher, slightly more abstract level. With the composition of enough such transformations, very complex functions can be learned.*深度学习方法是具有多级表示的学习方法，通过组合简单但非线性的模块来获得，有的模块将表示一个级别（从原始输入开始）转化为一个更高的、略微抽象的层次的表示。通过组成足够多的这种变换，可以学习到非常复杂的功能。

### 应用领域

深度学习的应用领域：

1. 计算机视觉

​       图像识别，人脸识别，自动驾驶

2. 自然语言处理 

​      智能搜索、机器翻译，文本摘要，人机对话...

​      医学图像分析，制药，游戏助理....

## 监督学习

​       *We compute an objective function that measures the error (or dis-tance) between the output scores and the desired pattern of scores. The machine then modifies its internal adjustable parameters to reduce this error. These adjustable parameters, often called weights, are real numbers that can be seen as ‘knobs’ that define the input–output func-tion of the machine. In a typical deep-learning system, there may be hundreds of millions of these adjustable weights, and hundreds of millions of labelled examples with which to train the machine.* 通过计算一个目标函数来测量出输出分数和期望模式分数之间的误差（或距离）。然后机器会修改其内部可调参数，以减少这种误差（就是反向传播的过程）。这些可调节的参数，通常被称为权值，它们是一些实数，可以被看作是定义了机器的输入输出功能的“旋钮”。在典型的深度学习系统中，可能有数以亿计的样本和权值和带有标签的样本，用来训练机器。

### 梯度下降：

不断用梯度对参数进行微调，直到找到相应的位置。

## 反向传播算法

通过计算输出层与期望值之间的误差来调整网络参数，使得误差最小。

<center>
<img 
src="Basic Concepts picture/deeplearning_BP.png">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;">BP</div>
</center>

##  卷积神经网络CNN

<center>
<img 
src="Basic Concepts picture/deeplearning_CNN.png">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;">CNN</div>
</center>

1）预处理的数据用于卷积的计算，加上偏置得到feature map

2）将前面卷积的结果进行非线性激活函数的处理，目前常用的Relu

3）池化（取区域的最大值或者是平均值），保留其显著的特征

4）全连接层，对结果的输出和确认

CNN能够减少参数，提高效率，降低计算复杂度。

## 分布表示和语言模型

### Word Embedding 词向量

 Word Embedding 的过程是将文本空间的某个word通过一定的方法映射或嵌入到另一个数值的向量空间。 传统的词的表示是one-hot Embedding，在词典中词的位置表示一个词的含义，仅仅表示词的一个特殊的位置，并不表示词的含义。Word Embedding可以进行词的向量空间的映射，在向量空间中离的比较近的词意思是比较接近的。

<center>
<img 
src="Basic Concepts picture/deeplearning_Word Embedding.png">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;">Word Embedding</div>
</center>

### Word2Vex算法

## 循环神经网络RNN

<center>
<img 
src="Basic Concepts picture/deeplearning_RNN.png">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;">RNN</div>
</center>

<center>
<img 
src="Basic Concepts picture/deeplearning_RNN1.png">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;">RNN</div>
</center>

<center>
<img 
src="Basic Concepts picture/deeplearning_RNN2.png">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;">RNN</div>
</center>

### LSTM(Long Short-term Memory)

<center>
<img 
src="Basic Concepts picture/deeplearning_LSTM.png">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;">LSTM</div>
</center>

## 未来

* 非监督学习
* 强化学习
* GAN生成对抗网络
* 自监督学习

