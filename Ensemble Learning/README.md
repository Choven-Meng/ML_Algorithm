[Forest （Isolation Forest）孤立森林](https://www.jianshu.com/p/5af3c66e0410?utm_campaign=maleskine)





# 集成学习

集成学习(ensemble learning)通过构建并结合多个学习器来完成学习任务，有时也被称为多分类器系统(multi-classifier system).  
如下图，集成学习的一般结构是：先产生一组“个体学习器”（individual learner），再用某种策略将它们结合起来。  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<img src="http://tech-blog-pictures.oss-cn-beijing.aliyuncs.com/2017/集成学习/1.png" >  

&emsp;&emsp;个体学习器通常是用一个现有的学习算法从训练数据产生，例如C4.5决策树算法、BP神经网络算法等。此时集成中只包含同种类型的个体学习器，例如“决策树集成”中的个体学习器全是决策树，“神经网络集成”中就全是神经网络，这样的集成是“同质”（homogeneous）的，同质集成中的个体学习器也称为**“基学习器”**（base learner），相应的学习算法称为“基学习算法”（base learning algorithm）。有同质就有异质（heterogeneous），若集成包含不同类型的个体学习器，例如同时包含决策树和神经网络，那么这时个体学习器一般不称为基学习器，而称作**“组件学习器”**（component leaner）或直接称为个体学习器。   

&emsp;&emsp;而根据个体学习器生成方式的不同，目前集成学习方法大致可分为两大类，即个体学习器间存在强依赖关系、必须**串行**生成的序列化方法，以及个体学习器间不存在强依赖关系、可同时生成的**并行**化方法；前者的代表是Boosting，后者的代表是和Bagging和“随机森林”（Random Forest）.  




## Bagging：基于数据随机重抽样的分类器构建方法

自举汇聚法(bootstrap aggregating)，也被称为bagging方法，是在从原始数据集选择S次之后得到S个新数据集的一种技术。新数据集和原始数据集的大小相等。每个数据集都是通过在原始数据集中随机选择一个样本进行替换而得到的。  
在S个数据集建好之后，将某个学习算法分别作用于每个数据集就得到S个分类器。当我们要对新数据进行分类时，就可以应用这S个分类器进行分类。与此同时，选择分类器投票结果中最多的类别作为最后的分类结果。    
bagging可以简单的理解为：放回抽样，多数表决(分类)或简单平均(回归).
更先进的bagging方法如下的随机森林。

### * 随机森林



## Boosting

boosting是一种与bagging很类似的技术。不论是在boosting还是bagging当中，所使用的多个分类器的类型都是一致的。但是前者中，不同的分类器是通过**串行**训练获得的，每个分类器都根据已训练出的分类器的性能来进行训练。boosting是通过集中关注被已有分类器错分的那些数据来获得新的分类器。  
由于boosting分类的结果是基于所有分类器的**加权求和**结果的，因此boosting和bagging不太一样。bagging中的分类器**权重相等**，而boosting中分类器权重不等，每个权重代表的是其对应分类器在**上一轮迭代中的成功度。**

### * Adaboost

### * GBDT



**bagging算法只能改善模型高方差（high variance）情况，Boosting算法对同时控制偏差（bias）和方差都有非常好的效果，而且更加高效**
