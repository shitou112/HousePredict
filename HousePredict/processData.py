import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler

FILENAME1 = "E:\\kaggle\\dataset\\housepredicting\\train.csv"
FILENAME2 = "E:\\kaggle\\dataset\\housepredicting\\test.csv"


def loadDataSet(filename1, filename2):
    traindata = pd.read_csv(filename1).set_index("Id")
    testdata = pd.read_csv(filename2).set_index("Id")
    return traindata, testdata

def fillNa(dataSet):
    columns = dataSet.columns.tolist()
    for column in columns:
        dataSet[column] = dataSet[column].fillna(dataSet[column].mode()[0])
    return dataSet

def standardData(dataSet):
    scaler = StandardScaler().fit(dataSet)
    dataSet = scaler.transform(dataSet)
    return dataSet

def processData(filename1, filename2):
    traindata, testdata = loadDataSet(filename1, filename2)
    traindataNum = np.shape(traindata)[0]
    all_data = pd.concat((traindata, testdata))

    #去掉'Alley'列，其缺失值太多
    all_data = all_data.drop('Alley',axis=1)

    #将偏度值大于0.8的属性进行对数处理，从而转为正太分布
    numeric_feats = all_data.dtypes[all_data.dtypes != 'object'].index
    skewed_feats = traindata[numeric_feats].apply(lambda x: skew(x.dropna()))  # compute skewness
    skewed_feats = skewed_feats[skewed_feats > 0.8]
    skewed_feats = skewed_feats.index

    all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

    #对离散数据进行dummy处理转化为数值型数据
    all_data = pd.get_dummies(all_data)

    #填充缺失值，用均值来弥补缺失值
    all_data = all_data.fillna(all_data.mean())

    traindata = all_data[:traindataNum]
    trainLabels = traindata['SalePrice']
    testdata = all_data[traindataNum:]



    traindata = traindata.drop('SalePrice', axis=1)
    testdata = testdata.drop('SalePrice', axis=1)



    # print(traindata.isnull())
    # traindata = standardData(traindata)
    # testdata = standardData(testdata)


    return traindata, trainLabels, testdata

if __name__ == "__main__":
    processData(FILENAME1, FILENAME2)
# print(traindata.groupby(traindata.LotArea//10*10).min())
# precise = pd.DataFrame({"price":traindata["SalePrice"]})
# precise.hist()
# plt.show()
