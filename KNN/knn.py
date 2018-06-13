'''
kNN: k Nearest Neighbors

输入:  newInput:   (1xN)的待分类向量  
       dataSet:    (NxM)的训练数据集  
       labels:     训练数据集的类别标签向量               
        k:         近邻数  
输出:      可能性最大的分类标签  

'''

import numpy as np

#创建一个数据集，包含2个类别共4个样本
def createDataSet():
    # 生成一个矩阵，每行表示一个样本
    group = np.array([[1.0, 0.9], [1.0, 1.0], [0.1, 0.2], [0.0, 0.1]])
    # 4个样本分别所属的类别
    labels = ['A', 'A', 'B', 'B']
    return group, labels
       
#KNN分类算法函数定义

def kNNClassify(newInput, dataSet, labels, k):
    numSamples = dataSet.shape[0]   # shape[0]表示行数

#     step 1: 计算距离[
#     假如：
#     Newinput：[1,0,2]
#     Dataset:
#     [1,0,1]
#     [2,1,3]
#     [1,0,2]
#     计算过程即为：
#     1、求差
#     [1,0,1]       [1,0,2]
#     [2,1,3]   --   [1,0,2]
#     [1,0,2]       [1,0,2]
#     =
#     [0,0,-1]
#     [1,1,1]
#     [0,0,-1]
#     2、对差值平方
#     [0,0,1]
#     [1,1,1]
#     [0,0,1]
#     3、将平方后的差值累加
#     [1]
#     [3]
#     [1]
#     4、将上一步骤的值求开方，即得距离
#     [1]
#     [1.73]
#     [1]
#      ]
    diff = np.tile(newInput, (numSamples, 1)) - dataSet  # 按元素求差值
    squaredDiff = diff ** 2  # 将差值平方
    squaredDist = np.sum(squaredDiff, axis = 1)   # 按行累加
    distance = squaredDist ** 0.5  # 将差值平方和求开方，即得距离   
    #output: array([1.08166538, 1.14017543, 0.1       , 0.2236068 ])
    
    # step 2: 对距离排序
    # argsort() 返回排序后的索引值   // array([2, 3, 0, 1], dtype=int64)
    sortedDistIndices = np.argsort(distance)
    classCount = {} # define a dictionary (can be append element)
    
    for i in range(k):
        # step 3: 选择k个最近邻
        voteLabel = labels[sortedDistIndices[i]]

        # step 4: 计算k个最近邻中各类别出现的次数
        ##get()返回指定键的值，如果值不在字典中返回default=0值
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

    # step 5: 返回出现次数最多的类别标签
    maxCount = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key

    return maxIndex
    
#测试

# 生成数据集和类别标签
dataSet, labels = createDataSet()
# 定义一个未知类别的数据
testX = np.array([1.2, 1.0])
k = 3
# 调用分类函数对未知数据分类
outputLabel = kNNClassify(testX, dataSet, labels, 3)
print ("Your input is:", testX, "and classified to class: ", outputLabel)

testX = np.array([0.1, 0.3])
outputLabel = kNNClassify(testX, dataSet, labels, 3)
print ("Your input is:", testX, "and classified to class: ", outputLabel)

#output: Your input is: [1.2 1. ] and classified to class:  A
#        Your input is: [0.1 0.3] and classified to class:  B




'''Sklearn代码实现'''
# weights参数有’uniform’和‘distance’两种
#algorithm可选参数有‘ball_tree’、 ‘kd_tree’ 、‘brute’和‘auto’ 

from sklearn import neighbors   
knn = neighbors.KNeighborsClassifier(n_neighbors=4,weights='distance') #取得knn分类器    
#data = np.array([[3,104],[2,100],[1,81],[101,10],[99,5],[98,2]])
data = dataSet
#labels = np.array([1,1,1,2,2,2])
labels = labels
knn.fit(data,labels) #导入数据进行训练   
print(knn.predict([[1.2,1.0],[0.1, 0.3]])) 

