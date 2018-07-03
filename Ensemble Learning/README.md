集成学习(ensemble learning)通过构建并结合多个学习器来完成学习任务，有时也被称为多分类器系统(multi-classifier system).  
如下图，集成学习的一般结构是：先产生一组“个体学习器”（individual learner），再用某种策略将它们结合起来。  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<img src="http://tech-blog-pictures.oss-cn-beijing.aliyuncs.com/2017/集成学习/1.png" >  

个体学习器通常是用一个现有的学习算法从训练数据产生，例如C4.5决策树算法、BP神经网络算法等。此时集成中只包含同种类型的个体学习器，例如“决策树集成”中的
个体学习器全是决策树，“神经网络集成”中就全是神经网络，这样的集成是“同质”（homogeneous）的，同质集成中的个体学习器也称为“基学习器”（base learner），
相应的学习算法称为“基学习算法”（base learning algorithm）。有同质就有异质（heterogeneous），若集成包含不同类型的个体学习器，
例如同时包含决策树和神经网络，那么这时个体学习器一般不称为基学习器，而称作“组件学习器”（component leaner）或直接称为个体学习器。
