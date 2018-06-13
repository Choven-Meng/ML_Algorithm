# coding=UTF-8
from numpy import *
import re
import warnings
warnings.filterwarnings('ignore')

def loadTrainDataSet():
    # 读取训练集
    fileIn = open('testSet.txt')
    postingList = []   # 邮件表，二维数组
    classVec = []   # 标签数组

    for line in fileIn.readlines():
        lineArr = line.strip().split()
        temp = []
        for i in range(len(lineArr)):
            if i == 0:
                classVec.append(int(lineArr[i]))    # 转化为整形
            else:
                temp.append(lineArr[i])
        postingList.append(temp)
    return postingList, classVec

def createVocabList(dataSet):
    # 创建词典
    vocabSet = set([])  # 定义list型的集合
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    # 对于每一个训练样本，得到其特征向量
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            pass
            # print("\'%s\' 不存在于词典中"%word)
    return returnVec

def createTrainMatrix(vocabList,postingList):
    # 生成训练矩阵，即每个样本的特征向量
    trainMatrix = []   # 训练矩阵
    for i in range(len(postingList)):
        curVec = setOfWords2Vec(vocabList, postingList[i])
        trainMatrix.append(curVec)
        return trainMatrix


def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)  # 样本数量
    numWords = len(trainMatrix[0])  # 样本特征数
    pAbusive = sum(trainCategory)/float(numTrainDocs) #p(y=1)
    # 分子赋值为1，分母赋值为2（拉普拉斯平滑）
    p0Num = ones(numWords)   # 初始化向量，代表所有0类样本中词j出现次数
    p1Num = ones(numWords)   # 初始化向量，代表所有1类样本中词j出现次数
    p0Denom = p1Denom = 2.0  # 代表0类1类样本的总词数
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # p1Vect = p1Num/p1Denom  # 概率向量(p(x0=1|y=1),p(x1=1|y=1),...p(xn=1|y=1))
    # p0Vect = p0Num/p0Denom  # 概率向量(p(x0=1|y=0),p(x1=1|y=0),...p(xn=1|y=0))
    # 取对数，之后的乘法就可以改为加法，防止数值下溢损失精度
    p1Vect = log( p1Num/p1Denom)
    p0Vect = log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive


def classifyNB(vocabList,testEntry,p0Vec,p1Vec,pClass1):  #朴素贝叶斯分类
    # 先将输入文本处理成特征向量
    regEx = re.compile('\\W*') # 正则匹配分割，以字母数字的任何字符为分隔符
    testArr = regEx.split(testEntry)
    testVec = array(setOfWords2Vec(vocabList, testArr))
    # 此处的乘法并非矩阵乘法，而是矩阵相同位置的2个数分别相乘
    # 矩阵乘法应当 dot(A,B) 或者 A.dot(B)
    # 下式子是原式子取对数，因此原本的连乘变为连加
    p1 = sum(testVec * p1Vec) + log(pClass1)
    p0 = sum(testVec * p0Vec) + log(1.0 - pClass1)
    # 比较大小即可
    if p1 > p0:
        return 1
    else:
        return 0


#测试方法
def testingNB():
    postingList, classVec = loadTrainDataSet()
    vocabList = createVocabList(postingList)

    trainMat = []
    for postinDoc in postingList:
        trainMat.append(setOfWords2Vec(vocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(classVec))

    # 输入测试文本，单词必须用空格分开
    testEntry = 'welcome to my blog!'
    #testEntry = 'fuck you bitch'
    print('测试文本为： '+testEntry)
    if classifyNB(vocabList,testEntry,p0V,p1V,pAb):
        print("--------侮辱性邮件--------")
    else:
        print("--------正常邮件--------")


testingNB()

# output
# 测试文本为： welcome to my blog!
# --------正常邮件--------
