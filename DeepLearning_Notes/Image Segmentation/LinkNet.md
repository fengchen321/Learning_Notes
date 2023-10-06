# LinkNet

[LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation](https://arxiv.org/abs/1707.03718)(IEEE 2017)

* ==替换主网络：==ResNet101或ResNet50-->ResNet34或ResNet18
* ==减少通道数：==避免出现2048，1024等过多的通道数
* ==减少卷积层==
* ==将卷积层替换为组卷积层==
* ==增加前期数据处理==
* ==减少复杂融合方式==
* ==避免使用全连接==

**网络结构**

* ==创新点：==每个编码器与解码器相连接，编码器的输入连接对应解码器的输出上。恢复降采样操作中丢失的空间信息；可以减少解码器的参数，因为解码器是共享从编码器的每一层学习到的参数。



conv代表卷积，full-conv代表全卷积，/2代表下采样的步长是2，*2代表上采样的因子是2.在卷积层之后加BN,后加ReLU.左半部分表示编码，右半部分表示解码。编码块基于ResNet18。

# SAM

> 文章标题：[Segment Anything](https://arxiv.org/abs/2304.02643)
> 作者：[Alexander Kirillov](https://arxiv.org/search/cs?searchtype=author&query=Kirillov%2C+A), [Eric Mintun](https://arxiv.org/search/cs?searchtype=author&query=Mintun%2C+E), [Nikhila Ravi](https://arxiv.org/search/cs?searchtype=author&query=Ravi%2C+N), [Hanzi Mao](https://arxiv.org/search/cs?searchtype=author&query=Mao%2C+H), [Chloe Rolland](https://arxiv.org/search/cs?searchtype=author&query=Rolland%2C+C), [Laura Gustafson](https://arxiv.org/search/cs?searchtype=author&query=Gustafson%2C+L), [Tete Xiao](https://arxiv.org/search/cs?searchtype=author&query=Xiao%2C+T), [Spencer Whitehead](https://arxiv.org/search/cs?searchtype=author&query=Whitehead%2C+S), [Alexander C. Berg](https://arxiv.org/search/cs?searchtype=author&query=Berg%2C+A+C), [Wan-Yen Lo](https://arxiv.org/search/cs?searchtype=author&query=Lo%2C+W), [Piotr Dollár](https://arxiv.org/search/cs?searchtype=author&query=Dollár%2C+P), [Ross Girshick](https://arxiv.org/search/cs?searchtype=author&query=Girshick%2C+R)
> 发表时间：2023
> [论文主页](https://segment-anything.com/)
