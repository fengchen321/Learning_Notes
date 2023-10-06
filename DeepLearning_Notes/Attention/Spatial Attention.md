# 基于 RNN 的方法

## RAM

> 文章标题：
> 作者：
> 发表时间：()

# 使用子网络来显式预测相关区域

## STN

# 使用子网络来隐式预测软掩码以选择重要区域

## GENet

# 自注意力相关

## Non-local

> 文章标题：[Non-local Neural Networks](https://arxiv.org/abs/1711.07971v3)
> 作者：Xiaolong Wang,  Ross Girshick,  Abhinav Gupta,  Kaiming He
> 发表时间：(CVPR 2018)
>
> [official code](https://github.com/facebookresearch/video-nonlocal-net)
>
> [Non-local_pytorch](https://github.com/AlexHex7/Non-local_pytorch)   [Non-Local-NN-Pytorch](https://github.com/tea1528/Non-Local-NN-Pytorch)

<center>
<img 
src="Spatial Attention.assets/Non-Local.png" width="400">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;">Non-Local</div>
</center>


$$
y = softmax(x^TW_\theta^T W_\phi x)g(x)
\\ y_i=\frac{1}{C(x)}\sum_{\forall j}f(x_i,x_j)g(x_j)
\\ z_i= W_zy_i+x_i
\\ Q,K \ dot\ product\rightarrow softmax
$$

- f 函数式计算i和j的相似度；g 函数计算feature map在j位置的表示；最终的y是通过响应因子C(x) 进行标准化处理以后得到的

[Non-Local neural networks的理解与实现](https://www.cnblogs.com/pprp/p/12153255.html)
