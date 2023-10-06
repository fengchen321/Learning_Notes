[toc]

# 基本概念

## 卷积核

卷积核可以找到图像中与自身纹理最相似的部分，相似度越高，得到的响应值越大。

### $1\times1$卷积

[Network In Network](https://arxiv.org/abs/1312.4400)

* 降维或升维

  > 通过控制卷积核的数量达到通道数大小的放缩。特征降维带来的好处是可以减少参数和减少计算量。

* 跨通道信息交融

* 较少参数量

* 增加模型深度

* 增加非线性

  > 1×1卷积核的卷积过程相当于全连接层的计算过程，并且还加入了非线性激活函数，从而可以增加网络的非线性，使得网络可以表达更加复杂的特征。

## 感受野(Receptive Field)

   感受野指的是卷积神经网络每一层输出的特征图(feature map)上每个像素点映射回输入图像上的区域大小，神经元感受野的范围越大表示其能接触到的原始图像范围就越大，也意味着它能学习更为全局，语义层次更高的特征信息，相反，范围越小则表示其所包含的特征越趋向局部和细节。因此感受野的范围可以用来大致判断每一层的抽象层次，并且我们可以很明显地知道网络越深，神经元的感受野越大。

<center>
<img 
src="Basic Concepts.assets/感受野.png" width="300"  height = "300" />
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;">感受野</div>
</center>


感受野计算公式：$F(i)=(F(i+1)-1)\times stride+Ksize$

Feautre map：$F=1$；pool1：$F=(1-1)\times2+2=2$；Conv1：$F=(2-1)\times2+3=5$

> $F(i)$为第$i$层感受野
>
> stride为第$i$层步距
>
> Ksize为卷积核或池化核尺寸

Feautre map：$F=1$；

Conv$3\times3(3)$：$F=(1-1)\times1+3=3$

Conv$3\times3(2)$：$F=(3-1)\times1+3=5$

Conv$3\times3(1)$：$F=(5-1)\times1+3=7$

## 分辨率(Resolution)

  ​       分辨率指的是输入模型的图像尺寸，即长宽大小。通常情况会根据模型下采样次数n和最后一次下采样后feature map的分辨率k×k来决定输入分辨率的大小，即：
$$
r = k\times2^n
$$
从输入$r×r$到最后一个卷积特征feature map的k×k，整个过程是一个信息逐渐抽象化的过程，即网络学习到的信息逐渐由低级的几何信息转变为高级的语义信息，这个feature map的大小可以是3×3，5×5，7×7，9×9等等，k太大会增加后续的计算量且信息抽象层次不够高，影响网络性能，k太小会造成非常严重的信息丢失，如原始分辨率映射到最后一层的feature map有效区域可能不到一个像素点，使得训练无法收敛。
        在ImageNet分类任务中，通常设置的5次下采样，并且考虑到其原始图像大多数在300分辨率左右，所以把最后一个卷积特征大小设定为7×7，将输入尺寸固定为224×224×3。在目标检测任务中，很多采用的是416×416×3的输入尺寸，当然由于很多目标检测模型是全卷积的结构，通常可以使用多尺寸训练的方式，即每次输入只需要保证是32×的图像尺寸大小就行，不固定具体数值。但这种多尺度训练的方式在图像分类当中是不通用的，因为分类模型最后一层是全连接结构，即矩阵乘法，需要固定输入数据的维度。

## 深度(Depth)

  ​      神经网络的深度决定了网络的表达能力，它有两种计算方法，早期的backbone设计都是直接使用卷积层堆叠的方式，它的深度即神经网络的层数，后来的backbone设计采用了更高效的module(或block)堆叠的方式，每个module是由多个卷积层组成，它的深度也可以指module的个数，这种说法在神经架构搜索(NAS)中出现的更为频繁。通常而言网络越深表达能力越强，但深度大于某个值可能会带来相反的效果，所以它的具体设定需要不断调参得到。

## 宽度(Width)

​        宽度决定了网络在某一层学到的信息量，但网络的宽度时指的是卷积神经网络中最大的通道数，由卷积核数量最多的层决定。通常的结构设计中卷积核的数量随着层数越来越多的，直到最后一层feature map达到最大，这是因为越到深层，feature map的分辨率越小，所包含的信息越高级，所以需要更多的卷积核来进行学习。通道越多效果越好，但带来的计算量也会大大增加，所以具体设定也是一个调参的过程，并且各层通道数会按照8×的倍数来确定，这样有利于GPU的并行计算。

<center>
<img 
src="Basic Concepts.assets/概念.png">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;">宽度</div>
</center>


## 下采样(Down-Sample)

​      下采样层有两个作用，一是减少计算量，防止过拟合，二是增大感受野，使得后面的卷积核能够学到更加全局的信息。下采样的设计有两种：

1. 采用stride为2的池化层，如Max-pooling或Average-pooling，目前通常使用Max-pooling，因为它计算简单且最大响应能更好保留纹理特征；
2. 采用stride为2的卷积层，下采样的过程是一个信息损失的过程，而池化层是不可学习的，用stride为2的可学习卷积层来代替pooling可以得到更好的效果，当然同时也增加了一定的计算量。
   (**突然想到为啥不使用双线性插值，向下插值来代替Pooling，这个虽然比MaxPooling计算量更大，但是保留的信息应该更丰富才是)**

## 上采样(Up-Sampling)

​       在卷积神经网络中，由于输入图像通过卷积神经网络(CNN)提取特征后，输出的尺寸往往会变小，而有时我们需要将图像恢复到原来的尺寸以便进行进一步的计算(如图像的语义分割)，这个使图像由小分辨率映射到大分辨率的操作，叫做上采样，它的实现一般有三种方式：

- 插值，一般使用的是双线性插值，因为效果最好，虽然计算上比其他插值方式复杂，但是相对于卷积计算可以说不值一提；

- 转置卷积又或是说反卷积，通过对输入feature map间隔填充0，再进行标准的卷积计算，可以使得输出feature map的尺寸比输入更大；

- Max Unpooling，在对称的max pooling位置记录最大值的索引位置，然后在unpooling阶段时将对应的值放置到原先最大值位置，其余位置补0；

  <center>

<center>
<img 
src="Basic Concepts.assets/上采样.png">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;">上采样</div>
</center>

## 参数量(Params)

​       参数量指的网络中可学习变量的数量，包括卷积核的权重weight，批归一化(BN)的缩放系数γ，偏移系数β，有些没有BN的层可能有偏置bias，这些都是可学习的参数 ，即在模型训练开始前被赋予初值，在训练过程根据链式法则中不断迭代更新，整个模型的参数量主要由卷积核的权重weight的数量决定，参数量越大，则该结构对运行平台的内存要求越高，参数量的大小是轻量化网络设计的一个重要评价指标。

## 计算量(FLOPs)

​       神经网络的前向推理过程基本上都是乘累加计算，所以它的计算量也是指的前向推理过程中乘加运算的次数，通常用FLOPs来表示，即floating point operations(浮点运算数)。计算量越大，在同一平台上模型运行延时越长，尤其是在移动端/嵌入式这种资源受限的平台上想要达到实时性的要求就必须要求模型的计算量尽可能地低，但这个不是严格成正比关系，也跟具体算子的计算密集程度(即计算时间与IO时间占比)和该算子底层优化的程度有关。

## 各种计算

### 卷积层

输入$H\cdot W\cdot M$；卷积尺寸$K\times K$；$N$个卷积核，输出feature map为$H' \cdot W' \cdot N$。

**参数量**

> $K\times K\times M\times N+N$（每个卷积核一个偏置项）
>
> 通常写作：<font color=#f12c60>$K\times K\times M\times N$</font>（不考虑偏置项）

**计算量FLOPS**

> 一次卷积乘法次数  $K\times K\times M$（卷积核感受野大小）
>
> 一次卷积加法次数  $K\times K\times M-1$（n个数相加做n-1次加法）
>
> 共进行$H' \cdot W' \cdot N$次卷积（输出feature map大小）
>
> 乘加运算总次数：$(2\times K \times K \times M-1)\times H' \times W' \times N$
>
> 通常写作：<font color=#f12c60>$K\times K\times M\times H' \times W' \times N$</font>（只考虑乘法）

**内存访问量MAC**

> 输入$H\times W\times M$
>
> 输出$H' \times W' \times N$
>
> 权重$K\times K\times M\times N$
>
> MAC=上述之和：<font color=#f12c60>$H\times W\times M+H' \times W' \times N+K\times K\times M \times N$</font>

### 全连接层

输入$C_i$个神经元；输出$C_o$个神经元

**参数量**

> <font color=#f12c60>$C_i\times C_o+C_o$</font>

**计算量FLOPS**

> 一次神经元乘法次数  $C_i$
>
> 一次神经元加法次数  $C_i-1$
>
> 共有个$C_o$神经元
>
> 乘加运算总次数：<font color=#f12c60>$(2\times C_i-1)\times C_o$</font>

**内存访问量MAC**

> 输入$C_i$
>
> 输出$C_o$
>
> 权重$C_\times C_o$
>
> MAC=上述之和：<font color=#f12c60>$C_i+C_o+C_i\times C_o$</font>

### BN层

参数量

> $s\times C_i$ （$\gamma,\beta$）

# 卷积计算类型

[不同类型卷积的综合介绍](https://zhuanlan.zhihu.com/p/366744794)

## 卷积操作

<center>
<img 
src="Basic Concepts.assets/卷积操作.png" >
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;">卷积操作</div>
</center>

$$
f(t)*g(t)=\int_{-\infty}^{\infty}f(\tau)g(t-\tau)d\tau
$$

## 互相关

称为两个函数的滑动点积或滑动内积。互相关中的滤波器没有反转。它直接滑过函数$f$。$f$和$g$之间的交叉区域是互相关。

<center>
<img 
src="Basic Concepts.assets/互相关.jpeg" >
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;">互相关</div>
</center>

## 标准卷积 (Convolution)



<center>
<img 
src="Basic Concepts.assets/cnn.png">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;">标准卷积</div>
</center>

<center>
<img 
src="Basic Concepts.assets/conv later.png">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;">标准卷积计算</div>
</center>
## 可分离卷积

### 空间可分离卷积（Spatial Separable Convolution)

空间可分离卷积：将一个卷积核分为两部分（降低计算复杂度，但并非所有的卷积核都可以分）

主要处理图像和卷积核（kernel）的空间维度：宽度和高度。 （另一个维度，“深度”维度，是每个图像的通道数）

$n\times n$卷积分解成$n \times1$卷积和$1\times n$卷积

n越大，节省的运算量越大

<center>
<img 
src="Basic Concepts.assets/InceptionV3_卷积分解1.png" width="400"  height = "300" />
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;">Inception Module B</div>
</center>

### 深度卷积 (Depthwise Convolution)

​        深度卷积与标准卷积相比，顾名思义是在深度上做了文章，而这里的深度跟网络的深度无关，它指的通道，标准卷积中每个卷积核都需要与feature map的所有层进行计算，所以每个卷积核的通道数等于输入feature map的通道数，通过设定卷积核的数量可以控制输出feature map的通道数。而深度卷积每个卷积核都是单通道的，维度为(1,1,D~k~,D~k~) ，卷积核的个数为iC，即第i个卷积核与feature map第i个通道进行二维的卷积计算，最后输出维度为(1,M,D~H~,D~W~)，它不能改变输出feature map的通道数，所以通常会在深度卷积后面接上一个(N,M,1,1)的标准卷积来代替3×3或更大尺寸的标准卷积，总的计算量为M×D~k~×D~k~×D~H~×D~W~+M×D~k~×D~H~×N，是普通卷积的1/N+1/(D~k~×D~k~)，大大减少了计算量和参数量，又可以达到相同的效果，这种结构被称为深度可分离卷积(Depthwise Separable Convolution)，**逐层卷积处理每个特征通道上的空间信息，逐点卷积进行通道间的特征融合。**在MobileNet V1被提出，后来渐渐成为轻量化结构设计的标配。

<center>
<img 
src="Basic Concepts.assets/depthwise conv.png">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;">深度卷积</div>
</center>



​     深卷积之前一直被吐槽在GPU上运行速度还不如一般的标准卷积，因为depthwise 的卷积核复用率比普通卷积要小很多，计算和内存访问的比值比普通卷积更小，因此会花更多时间在内存开销上，而且per-channel的矩阵计算很小不容易并行导致的更慢，但理论上计算量和参数量都是大大减少的，只是底层优化的问题。

## 分组卷积 (Group Convolution)

 当分组数量等于输入map数量，输出map数量也等于输入map数量，即g=N=C、N个卷积核每个尺寸为1∗K∗K时，Group Convolution就成了Depthwise Convolution
​       但由于feature map组与组之间相互独立，存在信息的阻隔，所以ShuffleNet提出对输出的feature map做一次channel shuffle的操作，即通道混洗，打乱原先的顺序，使得各个组之间的信息能够交互起来。

## 空洞卷积 (Dilated Convolution)

​       空洞卷积是针对图像语义分割问题中下采样会降低图像分辨率、丢失信息而提出的一种卷积思路。通过间隔取值扩大感受野，让原本3x3的卷积核，在相同参数量和计算量下拥有更大的感受野。这里面有个扩张率(dilation rate)的系数，这个系数定义了这个间隔的大小，标准卷积相当于dilation rate为1的空洞卷积，下图展示的是dilation rate为2的空洞卷积计算过程，可以看出3×3的卷积核可以感知标准的5×5卷积核的范围，还有一种理解思路就是先对3×3的卷积核间隔补0，使它变成5×5的卷积，然后再执行标准卷积的操作。

<center>
<img 
src="Basic Concepts.assets/Dilated Convolution.gif">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;">空洞卷积</div>
</center>

## 转置卷积 (Transposed Convolutions)

​       转置卷积又称反卷积(Deconvolution)，它和空洞卷积的思路正好相反，是为上采样而生，也应用于语义分割当中，而且他的计算也和空洞卷积正好相反，先对输入的feature map间隔补0，卷积核不变，然后使用标准的卷积进行计算，得到更大尺寸的feature map。

<center>
<img 
src="Basic Concepts.assets/Transposed Convolutions.png">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;">转置卷积</div>
</center>

## 可变形卷积 (deformable convolution)

​       以上的卷积计算都是固定的，每次输入不同的图像数据，卷积计算的位置都是完全固定不变，即使是空洞卷积/转置卷积，0填充的位置也都是事先确定的。而可变性卷积是指卷积核上对每一个元素额外增加了一个h和w方向上偏移的参数，然后根据这个偏移在feature map上动态取点来进行卷积计算，这样卷积核就能在训练过程中扩展到很大的范围。而显而易见的是可变性卷积虽然比其他卷积方式更加灵活，可以根据每张输入图片感知不同位置的信息，类似于注意力，从而达到更好的效果，但是它比可行变卷积在增加了很多计算量和实现难度，目前感觉只在GPU上优化的很好，在其他平台上还没有见到部署。

<center>
<img 
src="Basic Concepts.assets/deformable convolution.png">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;">可变形卷积</div>
</center>

# 其他算子

## 池化(pooling)

​        池化这个操作比较简单，一般在上采样和下采样的时候用到，没有参数，不可学习，但操作极为简单，和depthwise卷积类似，只是把乘累加操作替换成取最大/取平均操作。

### 最大池化和平均池化

<center>
<img 
src="Basic Concepts.assets/Max pooling.png">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;">最大池化</div>
</center>

<center>
<img 
src="Basic Concepts.assets/Average pooling.png">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;">平均池化</div>
</center>

### 全局平均池化

​       全局平均池化的操作是对一个维度为(C,H,W)的feature map，在HW方向整个取平均，然后输出一个长度为C的向量，这个操作一般在分类模型的最后一个feature map之后出现，然后接一个全连接层就可以完成分类结果的输出了。早期的分类模型都是把最后一个feature map直接拉平成C×H×W的向量，然后再接全连接层，但是显然可以看出来这个计算量极大，甚至有的模型最后一个全连接层占了整个模型计算量的50%以上，之后由研究人员发现对这个feature map做一个全局平均池化，然后再加全连接层可以达到相似的效果，且计算量降低到了原来的1/HW。

## 全连接计算(Full Connected)

这个本质其实就是矩阵乘法，输入一个(B, iC)的数据，权重为(iC, oC)，那么输出为(B, oC)，在多层感知机和分类模型最后一层常常见到。

## Addition / Concatenate分支

​       Addition和Concatenate分支操作统称为shortcut，如下图所示，操作极为简单。Addition是在ResNet中提出，两个相同维度的feature map相同位置点的值直接相加，得到新的相同维度feature map，这个操作可以融合之前的特征，增加信息的表达，Concatenate操作是在Inception中首次使用，被DenseNet发扬光大，和addition不同的是，它只要求两个feature map的HW相同，通道数可以不同，然后两个feature map在通道上直接拼接，得到一个更大的feature map，它保留了一些原始的特征，增加了特征的数量，使得有效的信息流继续向后传递。

<center>
<img 
src="Basic Concepts.assets/AdditionConcatenate.png">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;">分支</div>
</center>

## Channel shuffle

​       channel shuffle是ShuffleNet中首次提出，主要是针对分组卷积中不同组之间信息不流通，对不同组的feature map进行混洗的一个操作，如下图所示，假设原始的feature map维度为(1,9,H,W)，被分成了3个组，每个组有三个通道，那么首先将这个feature map进行reshape操作，得到(1,3,3,H,W)，然后对中间的两个大小为3的维度进行转置，依然是(1,3,3,H,W)，最后将通道拉平，变回(1,9,H,W)，就完成了通道混洗，使得不同组的feature map间隔保存，增强了信息的交互。

<center>
<img 
src="Basic Concepts.assets/Channel shuffle.png">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;">Channel shuffle</div>
</center>

# 常用激活函数

## ReLU系列

​       这里主要指常用的ReLU，ReLU6和leaky ReLU。ReLU比较好部署，小于0的部分为0，大于0的部分为原始值，只需要判断一下符号位就行；ReLU6与ReLU相比也只是在正向部分多了个阈值，大于6的值等于6，在实现时多了个比较也不算麻烦；而leaky ReLU和ReLU正向部分一样，都是大于0等于原始值，但负向部分却是等于原始值的1/10，浮点运算的话乘个0.1就好了，如果因为量化要实现整数运算，这块可以做个近似，如0.1用13>>7来代替，具体实现方法多种多样 ，还算简单。

**ReLu6**：防止激活变得太大

> <center>
> <img 
> src="Basic Concepts.assets/ReLu6.png">
> <br>
> <div style="color:orange; border-bottom: 1px solid #d9d9d9;
> display: inline-block;
> color: #999;
> padding: 2px;">ReLu6</div>
> </center>

**leaky ReLU**

> $$\begin{numcases}{\phi(x)=}
> x &if i>0\\
> 0.1x &otherwise
> \end{numcases}$$

<center>
<img 
src="Basic Concepts.assets/ReLu.png">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;">ReLu</div>
</center>

## Sigmoid系列

​       这里主要指sigmoid，还有和他相关的swish:
$$
sigmoid(x)=\frac{1}{1+e^{-x}}\\
swish = x\times{sigmoid(x)}
$$
​       可以看出，如果按照公式来实现sigmoid对低性能的硬件来说非常不友好，因为涉及到大量的exp指数运算和除法运算，于是有研究人员针对此专门设计了近似的硬件友好的函数h-sigmoid和h-swish函数，这里的h指的就是hardware的意思：
$$
Hsigmoid(x)=max(0,min(1,\frac{x+1}{2}))\\
Hswish=x\frac{ReLU6(x+3)}{6}
$$
​        可视化的对比如下图所示，可以看出在保证精度的同时又能大大方便硬件的实现，当然要直接实现sigmoid也是可以的，毕竟sigmoid是有限输出，当输入小于-8或大于8的时候，输出基本上接近于-1和1，可以根据这个特点设计一个查找表，速度也超快，且我们实测对精度没啥影响。

<center>
<img 
src="Basic Concepts.assets/sigmoid.png">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;">sigmoid</div>
</center>
# 训练算法

归一化

[](https://scortex.io/batch-norm-folding-an-easy-way-to-improve-your-network-speed/)

## Dropout

动机：一个好的模型需要对输入数据的扰动鲁棒

* 使用有噪音的数据等价于[Tikhonov正则](https://www.cnblogs.com/picassooo/p/13082208.html)
* 丢弃法：在层之间加入噪音
  * 无偏差的加入噪音
    * 对$x$加入噪音得到下$x^,$，我们希望$E[x^,]=x$
    * 丢弃法对每一个元素进行如下扰动$\begin{numcases}{x_i^,=} 0,&with probability p\\\frac{x_i}{1-p} ,&otherise\end{numcases}$

通常将Dropout作用在隐藏全连接层的输出上。

# 拓展阅读

[通过可视化对卷积的直观理解](https://towardsdatascience.com/a-comprehensive-introduction-to-different-types-of-convolutions-in-deep-learning-669281e58215)

[Intuitively Understanding Convolutions for Deep Learning](https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1)

[卷积算法指南](https://github.com/vdumoulin/conv_arithmetic)

