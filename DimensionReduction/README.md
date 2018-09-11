降维和度量学习

在高维情形下出现的样本稀疏、计算距离困难等问题，是所有机器学习方法共同面临的严重障碍，被称为**维数灾难。**

缓解维数灾难的一个重要途径是降维，亦称维数简约。即通过某种数学变换将原始高维属性空间转换为一个低维“子空间”，在这个子空间中样本密度大幅提高，距离计算也变的更加容易。为什么能进行降维？因为在很多时候，人们观测或收集到的数据样本虽是高维的，但与学习任务密切相关的也许是某个低维分布，即高维空间中的一个低维“嵌入”。原始高维空间中的样本点在这个低维嵌入子空间中更容易学习。降维是对数据高维度特征的一种预处理方法。降维是将高维度的数据保留下最重要的一些特征，去除噪声和不重要的特征，从而实现提升数据处理速度的目的。在实际的生产和应用中，降维在一定的信息损失范围内，可以为我们节省大量的时间和成本。降维也成为了应用非常广泛的数据预处理方法。

1、降维方法分为线性和非线性降维，非线性降维又分为基于核函数和基于特征值的方法。   
> (1)线性降维：PCA、ICA、LDA、LFA、LPP    
> (2)非线性降维方法：①基于核函数的方法：KPCA、KICA、KDA ;②基于特征值的方法：ISOMAP、LLE、LE、LPP、LTSA、MVU 

或者将降维方法如下图分类：  
<img src="https://github.com/Choven-Meng/ML_Algorithm/blob/master/DimensionReduction/photo/%E9%99%8D%E7%BB%B4%E6%96%B9%E6%B3%95.png" alt="图片描述" title="">

2、降维的作用：（为什么会有这些作用？）   
> （1）降低时间的复杂度和空间复杂度   
> （2）节省了提取不必要特征的开销  
> （3）去掉数据集中夹杂的噪音   
> （4）较简单的模型在小数据集上有更强的鲁棒性    
> （5）当数据能有较少的特征进行解释，我们可以更好地解释数据，是的我们可以提取知识   
> （6）实现数据的可视化 

3、降维的目的   
用来进行特征选择和特征提取。     
> (1)特征选择：选择重要的特征子集，删除其余特征；    
> (2)特征提取：由原始特征形成的较少的新特征。    
在特征提取中，我们要找到k个新的维度的集合，这些维度是原来k个维度的组合，这个方法可以是监督的，也可以是非监督的，如PCA是非监督的，LDA是监督的。 

4、降维的本质  
学习一个映射函数f：x->y。（x是原始数据点的表达，目前最多的是用向量来表示，Y是数据点映射后的低维向量表达。）f可能是：显示的、隐式的、线性的、非线性的。 

## 一. PCA主成分分析

[code]()

  PCA(principal Component Analysis)，即主成分分析方法，是一种使用最广泛的数据压缩算法。在PCA中，数据从原来的坐标系转换到新的坐标系，由数据本身决定。转换坐标系时，以**方差最大**的方向作为坐标轴方向，因为数据的最大方差给出了数据的最重要的信息。第一个新坐标轴选择的是原始数据中方差最大的方法，第二个新坐标轴选择的是与第一个新坐标轴**正交且方差次大**的方向。重复该过程，重复次数为原始数据的特征维数。  
  通过这种方式获得的新的坐标系，我们发现，大部分方差都包含在前面几个坐标轴中，后面的坐标轴所含的方差几乎为0,。于是，我们可以忽略余下的坐标轴，只保留前面的几个含有绝大部分方差的坐标轴。事实上，这样也就相当于只保留包含绝大部分方差的维度特征，而忽略包含方差几乎为0的特征维度，也就实现了对数据特征的降维处理。  
  那么，我们如何得到这些包含最大差异性的主成分方向呢？事实上，通过计算数据矩阵的协方差矩阵，然后得到协方差矩阵的特征值及特征向量，选择特征值最大（也即包含方差最大）的N个特征所对应的特征向量组成的矩阵，我们就可以将数据矩阵转换到新的空间当中，实现数据特征的降维（N维）。  
  既然，说到了协方差矩阵，那么这里就简单说一下方差和协方差之间的关系，首先看一下均值，方差和协方差的计算公式：  
  <img style="display: block; margin-left: auto; margin-right: auto" src="https://github.com/Choven-Meng/ML_Algorithm/blob/master/DimensionReduction/photo/%E6%96%B9%E5%B7%AE%E5%8D%8F%E6%96%B9%E5%B7%AE%E8%AE%A1%E7%AE%97%E5%85%AC%E5%BC%8F.png" alt="">  
  
 由上面的公式，我们可以得到一下两点区别：   
 > （1）方差的计算公式，我们知道方差的计算是针对一维特征，即针对同一特征不同样本的取值来进行计算得到；而协方差则必须要求至少满足二维特征。可以说方差就是协方差的特殊情况。　  
 > （2）方差和协方差的除数是n-1，这样是为了得到方差和协方差的无偏估计。具体推导过程可以参见[博文](https://blog.csdn.net/maoersong/article/details/21819957)     
 >  (4)协方差矩阵的对角上是方差，非对角线上是协方差。协方差是衡量两个变量同时变化的变化程度。协方差大于0表示x和y中若一个增，另一个也增；小于0表示一个增一个减。
 
**PCA算法实现**

将数据转换为只保留前N个主成分的特征空间的伪代码如下所示：  
```
去除平均值,xi = xi - mean
计算协方差矩阵
计算协方差矩阵的特征值和特征向量
将特征值排序
保留前N个最大的特征值对应的特征向量
将数据转换到上面得到的N个特征向量构建的新空间中（实现了特征压缩）
```

## 二. LDA线性判别分析

线性判别式分析（Linear Discriminant Analysis），简称为LDA。也称为Fisher线性判别（Fisher Linear Discriminant，FLD），是模式识别的经典算法。  
基本思想是将高维的模式样本投影到最佳鉴别矢量空间，以达到抽取分类信息和压缩特征空间维数的效果，投影后保证模式样本在新的子空间有**最大的类间距离和最小的类内距离，**即模式在该空间中有最佳的可分离性。   
LDA与前面介绍过的PCA都是常用的降维技术。PCA主要是从特征的协方差角度，去找到比较好的投影方式。LDA更多的是考虑了标注，即希望投影后不同类别之间数据点的距离更大，同一类别的数据点更紧凑。

给定N个特征为d维的样例x<sup>(i)</sup>{x1<sup>(i)</sup>,x2<sup>(i)</sup>,...,xd<sup>(i)</sup>}，其中有N1个样例属于类别w1，另外N2个样例属于类别w2。现在我们要将原始数据降低到只有一维，降维函数（或者叫投影函数）是：y=w<sup>T</sup>x，最后我们就依靠每个样例对应的y值来判别它属于哪一类。

形象图为：   
![](https://github.com/Choven-Meng/ML_Algorithm/blob/master/DimensionReduction/photo/LDA.png)

我们就是要找到这个最佳的w，使得样例映射到y后最易于区分。   
定义每类样例的均值点：<a href="https://www.codecogs.com/eqnedit.php?latex=u_{i}&space;=&space;\frac{1}{N_{i}}\sum&space;x" target="_blank"><img src="https://latex.codecogs.com/gif.latex?u_{i}&space;=&space;\frac{1}{N_{i}}\sum&space;x" title="u_{i} = \frac{1}{N_{i}}\sum x" /></a>  
样例投影到y后有均值点为：<a href="https://www.codecogs.com/eqnedit.php?latex=\tilde{u}_{i}&space;=&space;\frac{1}{N_{i}}\sum&space;w^{T}x" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\tilde{u}_{i}&space;=&space;\frac{1}{N_{i}}\sum&space;w^{T}x" title="\tilde{u}_{i} = \frac{1}{N_{i}}\sum w^{T}x" /></a>   
我们希望投影后两类样例中心尽量地分离，即:<a href="https://www.codecogs.com/eqnedit.php?latex=|\tilde{u}_{i}&space;-&space;\tilde{u}_{i}|&space;=&space;|w^{T}(u_{1}&space;-&space;u_{2})|" target="_blank"><img src="https://latex.codecogs.com/gif.latex?|\tilde{u}_{i}&space;-&space;\tilde{u}_{i}|&space;=&space;|w^{T}(u_{1}&space;-&space;u_{2})|" title="|\tilde{u}_{i} - \tilde{u}_{i}| = |w^{T}(u_{1} - u_{2})|" /></a> **越大越好**   
同时我们希望投影之后类内部的方差<a href="https://www.codecogs.com/eqnedit.php?latex=\tilde{s}_{i}^{2}&space;=&space;\sum&space;(y-u_{i})^{2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\tilde{s}_{i}^{2}&space;=&space;\sum&space;(y-u_{i})^{2}" title="\tilde{s}_{i}^{2} = \sum (y-u_{i})^{2}" /></a> **越小越好。**   
由于得到目标函数：   
<a href="https://www.codecogs.com/eqnedit.php?latex={\color{Red}&space;max&space;J(w)&space;=&space;\frac{|\widetilde{u}_{1}&space;-&space;\widetilde{u}_{2}|^2}{s^1&space;&plus;&space;s^2}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?{\color{Red}&space;max&space;J(w)&space;=&space;\frac{|\widetilde{u}_{1}&space;-&space;\widetilde{u}_{2}|^2}{s^1&space;&plus;&space;s^2}}" title="{\color{Red} max J(w) = \frac{|\widetilde{u}_{1} - \widetilde{u}_{2}|^2}{s^1 + s^2}}" /></a>

又是个最优化问题。最终解得    
<a href="https://www.codecogs.com/eqnedit.php?latex=w&space;=&space;(s_1&plus;s_2)^{2}(u_1&space;-&space;u_2)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?w&space;=&space;(s_1&plus;s_2)^{2}(u_1&space;-&space;u_2)" title="w = (s_1+s_2)^{2}(u_1 - u_2)" /></a>  s1和s2分别中原始样例的方差。   
如果<a href="https://www.codecogs.com/eqnedit.php?latex={\color{red}&space;y&space;=&space;w^Tx&space;-&space;w^Tu>0}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?{\color{red}&space;y&space;=&space;w^Tx&space;-&space;w^Tu>0}" title="{\color{red} y = w^Tx - w^Tu>0}" /></a> (u是所有样本的均值)，就属于类别C<sub>1</sub>,否则就属于类别C<sub>2</sub>.

假设有C个类别，降以一维已经不能满足分类要求了，我们需要k个基向量来做投影，W=[w1|w2|...|wk] 。样本点在这k维投影后的结果为[y1,y2,...,yk]，且有
y<sub>i</sub> = w<sub>i</sub><sup>T</sup>x; y = W<sup>T</sup>x。

**使用LDA的限制**  
> 1. LDA至多可生成C-1维子空间   
> 2. LDA不适合对非高斯分布的样本进行降维   
> 3. LDA在样本分类信息依赖方差而不是均值时，效果不好。   
> 4. LDA可能过度拟合数据。

