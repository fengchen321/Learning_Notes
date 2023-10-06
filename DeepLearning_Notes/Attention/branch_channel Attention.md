# different branches

## SKNet

> 文章标题：[Selective Kernel Networks](https://arxiv.org/abs/1903.06586)
> 作者：Xiang Li, Wenhai Wang, Xiaolin Hu, Jian Yang
> 发表时间：(CVPR 2019)
>
> [Official Code](https://github.com/implus/SKNet)

<center>
<img 
src="branch_channel Attention.assets/SK-pipeline.jpg" >
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;">SK_module</div>
</center>

**用multiple scale feature汇总的information来channel-wise地指导如何分配侧重使用哪个kernel的表征**

一种非线性方法来聚合来自多个内核的信息，以实现神经元的自适应感受野大小


> **Split**：生成具有不同内核大小的多条路径，这些路径对应于不同感受野(RF，receptive field) 大小的神经元
>
> > $X\in R^{H'\times W'\times C'} $
> >
> > $\tilde F:X\to \tilde U \in R^{H\times W\times C} $    kernel size $3\times3$
> >
> > $\hat F:X\to \hat U \in R^{H\times W\times C}$   kernel size $5\times5$：使用空洞卷积$3\times3$,空洞系数为2。
>
> **Fuse**：聚合来自多个路径的信息，以获得选择权重的全局和综合表示。 
>
> > $$
> > U=\tilde U+\hat U\\
> > s_c=F_{gp}(U_c)=\frac{1}{H\times W}\sum_{i=1}^H\sum_{j=1}^WU_c(i,j)\\
> > z=F_{fc}(s)=\delta(B(Ws)) 降维处理\\
> > $$
> >
> > $s\in R^c$；$\delta$：ReLU；$z\in R^{d\times1}$；$W\in R^{d\times C}$：批量归一化；
> >
> > $d=max(C/r,L)$       L：d的最小值，本文设置32
>
> **Select**：根据选择权重聚合不同大小内核的特征图
>
> > 在channel-wise应用softmax操作
> > $$
> > a_c=\frac{e^{A_cz}}{e^{A_cz}+e^{B_cz}}\\
> > b_c=\frac{e^{B_cz}}{e^{A_cz}+e^{B_cz}}\\
> > $$
> > $ A,B ∈R^{C\times d}$ ,$ a,b$ 分别表示 $\tilde U,\hat U$的软注意力向量。$A_c ∈ R^{1\times d }$是 A 的第$ c $行，$a_c$ 是 a 的第 $c $个元素，同理$B_c,b_c$。
> > $$
> > V_c=a_c\cdot\tilde U_c + b_c\cdot \hat U_c\\\
> > a_c+b_c=1\\
> > V_c\in R^{H\times W}
> > $$
>
> <center>
> <img 
> src="branch_channel Attention.assets/SK-pipeline-3.jpg" >
> <br>
> <div style="color:orange; border-bottom: 1px solid #d9d9d9;
> display: inline-block;
> color: #999;
> padding: 2px;">Selective Kernel Convolution三分支</div>
> </center>
>
>
>  $SK[M,G,r]\to SK[2,32,16]$
>
> > M：确定要聚合的不同内核的选择数量
> >
> > G：控制每条路径的基数的组号
> >
> > r：reduction ratio

$$
U_k=F_k(X)
\\ U = \sum_{k=1}^K U_k
\\ z = \delta(BN(WGAP(U)))
\\ s_k^{(c)} = \frac{e^{W_k^{(c)}z}}{\sum_{k=1}^K e^{W_k^{(c)}z}}
\\ Y=\sum_{k=1}^K s_kU_k
\\ global\ average\ pooling\rightarrow MLP\rightarrow softmax
$$

```python
class SKAttention(nn.Module):
    
    def __init__(self, channel=512,kernels=[1,3,5,7],reduction=16,group=1,L=32):
        super().__init__()
        self.d=max(L,channel//reduction)
        self.convs=nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv',nn.Conv2d(channel,channel,kernel_size=k,padding=k//2,groups=group)),
                    ('bn',nn.BatchNorm2d(channel)),
                    ('relu',nn.ReLU())
                ]))
            )
        self.fc=nn.Linear(channel,self.d)
        self.fcs=nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d,channel))
        self.softmax=nn.Softmax(dim=0)
        
    def forward(self, x):
        bs, c, _, _ = x.size()
        conv_outs=[]
        ### split
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats=torch.stack(conv_outs,0)#k,bs,channel,h,w

        ### fuse
        U=sum(conv_outs) #bs,c,h,w

        ### reduction channel
        S=U.mean(-1).mean(-1) #bs,c
        Z=self.fc(S) #bs,d

        ### calculate attention weight
        weights=[]
        for fc in self.fcs:
            weight=fc(Z)
            weights.append(weight.view(bs,c,1,1)) #bs,channel
        attention_weughts=torch.stack(weights,0)#k,bs,channel,1,1
        attention_weughts=self.softmax(attention_weughts)#k,bs,channel,1,1

        ### fuse
        V=(attention_weughts*feats).sum(0)
        return V
```

# different conv kernels

## CondConv

> 文章标题：[CondConv: Conditionally Parameterized Convolutions for Efficient Inference](https://arxiv.org/abs/1904.04971?context=cs.LG)
> 作者：Brandon Yang, Gabriel Bender, Quoc V. Le, Jiquan Ngiam
> 发表时间：(NIPS 2019)
>
> [official code](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/condconv)
>
> [pytorch版](https://github.com/xmu-xiaoma666/External-Attention-pytorch/blob/master/model/conv/CondConv.py)



<center>
<img 
src="branch_channel Attention.assets/CondConv.png" >
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;">Condconv</div>
</center>

$$
\alpha = \sigma(W_r(GAP(X)))
\\ Y = (\alpha _1W_1+\dots +\alpha_nW_n) *X
\\ global\ average\ pooling\rightarrow linear \ layer\rightarrow sigmoid
$$

## DynamicConv

> 文章标题：[Dynamic Convolution: Attention over Convolution Kernels](https://arxiv.org/abs/1912.03458)
> 作者：Yinpeng Chen, Xiyang Dai, Mengchen Liu, Dongdong Chen, Lu Yuan, Zicheng Liu
> 发表时间：(CVPR 2020)
>
> [pytorch版](https://github.com/xmu-xiaoma666/External-Attention-pytorch/blob/master/model/conv/DynamicConv.py)

<center>
<img 
src="branch_channel Attention.assets/DynamicConv.png" >
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;">DynamicConv</div>
</center>

动态卷积使用K个相同大小和输入/输出维度的并行卷积核，而不是每层一个核。与 SE 块一样，它采用挤压和激发机制来为不同的卷积核生成注意力权重。然后这些内核通过加权求和动态聚合并应用于输入特征图 X
$$
s = softmax(W_2\delta(W_1GAP(X)))
\\ DyConv = \sum_{i=1}^K s_kConv_k
\\Y = DyConv(X)
$$
