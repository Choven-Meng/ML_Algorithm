# AdaBoost

AdaBoost算法是模型为**加法模型**、损失函数为**指数函数**、学习算法为**前向分步算法**时的二类分类学习方法。

AdaBoost是adaptive boosting( **自适应增强** )的缩写，其运行过程如下：   
> &emsp;&emsp;训练数据中的每个样本，并赋予其一个权重，这些权重构成了向量D，一开始，这些权重都初始化成**相等值**。首先在训练数据上训练出一个若分类器并计算该分类器的错误率，然后在同一数据集上再次训练弱分类器。在分类器的第二次训练中，将会重新调整每个样本的权重，其中第一次分对的样本的权重将会降低，而第一次分错的样本的权重将会提高。为了从所有弱分类器中得到最终的分类结果，AdaBoost为每个分类器都分配了一个权重值alpha，这些alpha值是基于每个弱分类器的错误率进行计算的。  计算出alpha值之后，可以对权重向量D进行更新，以使得那些正确分类的样本的权重降低而错分样本的权重升高。AdaBoost算法会不断地重复训练和调整权重的过程，直到训练错误率为0或弱分类器的数目达到指定值为止。

## 算法思想
关于Adaboost，它是boosting算法，从bias-variance（偏差-方差）的角度来看，boosting算法主要关注的是降低偏差。仔细想想便可理解，因为boosting算法每个分类器都是弱分类器，而弱分类器的特性就是high-bias & low variance（**高偏差-低方差**），其与生俱来的优点就是**泛化性能好**。因此，将多个算法组合起来之后，可以达到降偏差的效果，进而得到一个偏差小、方差小的泛化能力好的模型。另外，Adaboost的损失函数是**指数损失** L(y,f(x))=e <sup>−yf(x)</sup> 。

具体说来，整个Adaboost 迭代算法就3步：

* 第一步：初始化训练数据的权值分布。如果有N个样本，则每一个训练样本最开始时都被赋予相同的权值：1/N。
* 第二步：训练弱分类器。迭代M次，每次都根据错误率e <sub>m</sub> 不断修改训练数据的权值分布（此处需要确保弱学习器的错误率e小于0.5，因为二分类问题随机猜测的概率是0.5），样本权值更新规则为增加分类错误样本的权重，减少分类正确样本的权重；
* 第三步：根据每个弱分类器器的系数α <sub>m</sub> ，将M个弱分类器组合成强分类器。各个弱分类器的训练过程结束后，加大分类误差率小的弱分类器的权重，使其在最终的分类函数中起着较大的决定作用，而降低分类误差率大的弱分类器的权重，使其在最终的分类函数中起着较小的决定作用。换言之，误差率低的弱分类器在最终分类器中占的权重较大，否则较小。

## AdaBoost算法流程：

（1）初始化训练数据的权值分布，每一个训练样本最开始时都被赋予相同的权值：1/N。  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<img title="Adaboost 算法的原理" class="imgshow" alt="Adaboost 算法的原理"  src="http://static.yihaodou.com/tec_data/2016/03/56369856efea72dad24vJIiJ.jpg">

(2)对m=1，2，...，M   

&emsp;&emsp;(a)使用具有权值分布D <sub>m</sub>的训练数据集学习，得到基本分类器（选取让误差率最低的阈值来设计基本分类器）：  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;
<img src="http://static.yihaodou.com/tec_data/2016/03/56380056efead8926210gdC3.jpg">

&emsp;&emsp;(b)计算第m次分类器G <sub>m</sub> (x)在训练数据集上的分类误差率  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;
<img src="http://static.yihaodou.com/tec_data/2016/03/56380856efeae09c2689nnOl.jpg">    
&emsp;&emsp;&emsp; **e <sub>m</sub> = 未正确分类的样本数目/所有样本数目**  

&emsp;&emsp;(c)计算第m次分类器G <sub>m</sub> (x)的系数  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;
<img  src="http://static.yihaodou.com/tec_data/2016/03/56381756efeae94964bBf8Xg.jpg">   
由上式可知，a <sub>m</sub> 随着e <sub>m</sub> 的减小而增大，意味着分类误差率越小的基本分类器在最终分类器中的作用越大。     

&emsp;&emsp;(d)更新训练数据集的权值分布（目的：得到样本的新的权值分布），用于下一轮迭代,使得被基本分类器Gm(x)误分类样本的权值增大，而被正确分类样本的权值减小。就这样，通过这样的方式，AdaBoost方法能“重点关注”或“聚焦于”那些较难分的样本上。   
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;
<img src="http://static.yihaodou.com/tec_data/2016/03/56382656efeaf2372733bVFy.jpg">       
&emsp;&emsp;&emsp;&emsp;其中，Z <sub>m</sub> 是规范化因子，使得D <sub>m</sub> +1成为一个概率分布：   
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;
<img src="http://static.yihaodou.com/tec_data/2016/03/56383356efeaf9bd0d1EXrW9.jpg">   
&emsp;&emsp;;通过简化，可把上述的更新权值公式简化为：  
&emsp;&emsp;&emsp;&emsp;* **正确分类时：w<sub>2i</sub> = w<sub>1i</sub>/2(1-e <sub>m</sub> )**       
&emsp;&emsp;&emsp;&emsp;* **错误分类时：w<sub>2i</sub> = w<sub>1i</sub>/2e<sub>m</sub>**   

(3)构建基本分类器的线性组合    
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;
<img src="http://static.yihaodou.com/tec_data/2016/03/56384156efeb01683c55dFzZ.jpg">    
&emsp;&emsp;&emsp;&emsp;&emsp;得到最终分类器：   
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;
<img src="http://static.yihaodou.com/tec_data/2016/03/56384856efeb08a44546tlhu.jpg">

公式推导过程详见[链接](https://blog.csdn.net/dream_angel_z/article/details/52348135)

AdaBoost最基本的性质是能在学习过程中**不断减少训练误差（训练误差以指数速率下降）**，即在训练数据集上的分类误差率。  

### AdaBoost的优点和缺点

#### 优点

     （1）Adaboost提供一种框架，在框架内可以使用各种方法构建子分类器。可以使用简单的弱分类器，不用对特征进行筛选，也不存在过拟合的现象。

     （2）Adaboost算法不需要弱分类器的先验知识，最后得到的强分类器的分类精度依赖于所有弱分类器。无论是应用于人造数据还是真实数据，Adaboost都能显著的提高学习精度。

     （3）Adaboost算法不需要预先知道弱分类器的错误率上限，且最后得到的强分类器的分类精度依赖于所有弱分类器的分类精度，可以深挖分类器的能力。Adaboost可以根据弱分类器的反馈，自适应地调整假定的错误率，执行的效率高。

     （4）Adaboost对同一个训练样本集训练不同的弱分类器，按照一定的方法把这些弱分类器集合起来，构造一个分类能力很强的强分类器，即“三个臭皮匠赛过一个诸葛亮”。

#### 缺点：

     在Adaboost训练过程中，Adaboost会使得难于分类样本的权值呈指数增长，训练将会过于偏向这类困难的样本，导致Adaboost算法易受噪声干扰。此外，Adaboost依赖于弱分类器，而弱分类器的训练时间往往很长。
