#_*_conding=utf-8_*_
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import matplotlib.mlab as mlab
import matplotlib.colors
import matplotlib.pyplot as plt
from functools import reduce

from scipy import stats
from scipy.stats import multivariate_normal
from sklearn import preprocessing
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.pairwise import pairwise_distances_argmin
import math
from scipy.integrate import tplquad,dblquad,quad
import scipy.stats

#from imblearn.over_sampling import RandomOverSampler

from sklearn.decomposition import PCA
from sklearn import manifold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import TruncatedSVD 

from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier


from scipy.interpolate import make_interp_spline

from sklearn import neighbors

from sklearn.feature_extraction.text import CountVectorizer  # 从sklearn.feature_extraction.text里导入文本特征向量化模块
from sklearn.naive_bayes import GaussianNB     # 从sklean.naive_bayes里导入朴素贝叶斯模型
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score,fbeta_score
from sklearn.metrics import roc_auc_score
import numpy as np;
import matplotlib.pyplot as plt;


from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import RandomUnderSampler

from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
from imblearn.under_sampling import AllKNN

from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.under_sampling import OneSidedSelection
from imblearn.under_sampling import InstanceHardnessThreshold

from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
import random

from sklearn import preprocessing
from sklearn.metrics import accuracy_score


from collections import  Counter

# 绘制背景的边界
from matplotlib.colors import ListedColormap

mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False
#数据预处理 usecols=np.arange(0,50)
def dataGetPromise12():
    data = [];
    data1 = np.loadtxt('..\\所用数据集\\promise\\ant-1.7.csv', dtype=np.float,delimiter=',', skiprows=1,usecols=np.arange(3,24));
    data2 = np.loadtxt('..\\所用数据集\\promise\\camel-1.6.csv', dtype=np.float,delimiter=',', skiprows=1,usecols=np.arange(3,24));
    data3 = np.loadtxt('..\\所用数据集\\promise\\ivy-2.0.csv', dtype=np.float,delimiter=',', skiprows=1,usecols=np.arange(3,24));
    data4 = np.loadtxt('..\\所用数据集\\promise\\jedit-4.0.csv', dtype=np.float,delimiter=',', skiprows=1,usecols=np.arange(3,24));
    data5 = np.loadtxt('..\\所用数据集\\promise\\log4j-1.0.csv', dtype=np.float,delimiter=',', skiprows=1,usecols=np.arange(3,24));
    data6 = np.loadtxt('..\\所用数据集\\promise\\lucene-2.4.csv', dtype=np.float,delimiter=',', skiprows=1,usecols=np.arange(3,24));
    data7 = np.loadtxt('..\\所用数据集\\promise\\poi-3.0.csv', dtype=np.float,delimiter=',', skiprows=1,usecols=np.arange(3,24));
    data8 = np.loadtxt('..\\所用数据集\\promise\\synapse-1.2.csv', dtype=np.float,delimiter=',', skiprows=1,usecols=np.arange(3,24));
    data9 = np.loadtxt('..\\所用数据集\\promise\\tomcat.csv', dtype=np.float,delimiter=',', skiprows=1,usecols=np.arange(3,24));
    data10 = np.loadtxt('..\\所用数据集\\promise\\velocity-1.6.csv', dtype=np.float,delimiter=',', skiprows=1,usecols=np.arange(3,24));
    data11 = np.loadtxt('..\\所用数据集\\promise\\xalan-2.4.csv', dtype=np.float,delimiter=',', skiprows=1,usecols=np.arange(3,24));
    data12 = np.loadtxt('..\\所用数据集\\promise\\xerces-1.3.csv', dtype=np.float,delimiter=',', skiprows=1,usecols=np.arange(3,24));
    data.append(data1)
    data.append(data2)
    data.append(data3)
    data.append(data4)
    data.append(data5)
    data.append(data6)
    data.append(data7)
    data.append(data8)
    data.append(data9)
    data.append(data10)
    data.append(data11)
    data.append(data12)
    return data;

def dataGetNasa7():
    data = [];
    data1 = np.loadtxt('..\\所用数据集\\nasa\\CM1.csv', dtype=np.float,delimiter=',', skiprows=1,usecols=np.arange(0,21));
    data2 = np.loadtxt('..\\所用数据集\\nasa\\KC1.csv', dtype=np.float,delimiter=',', skiprows=1,usecols=np.arange(0,21));
    data3 = np.loadtxt('..\\所用数据集\\nasa\\KC3.csv', dtype=np.float,delimiter=',', skiprows=1,usecols=np.arange(0,21));
    data4 = np.loadtxt('..\\所用数据集\\nasa\\MC2.csv', dtype=np.float,delimiter=',', skiprows=1,usecols=np.arange(0,21));
    data5 = np.loadtxt('..\\所用数据集\\nasa\\MW1.csv', dtype=np.float,delimiter=',', skiprows=1,usecols=np.arange(0,21));
    data6 = np.loadtxt('..\\所用数据集\\nasa\\PC2.csv', dtype=np.float,delimiter=',', skiprows=1,usecols=np.arange(0,21));
    data7 = np.loadtxt('..\\所用数据集\\nasa\\PC4.csv', dtype=np.float,delimiter=',', skiprows=1,usecols=np.arange(0,21));
    data.append(data1)
    data.append(data2)
    data.append(data3)
    data.append(data4)
    data.append(data5)
    data.append(data6)
    data.append(data7)
    return data;

def dataPreTreated(data):
    List = [];
    target = [];
    for i in range(len(data)):
        data_new, data_t = np.split(data[i], [-1, ], axis=1);
        List.append(preprocessing.scale(data_new));
        for j in range(len(data_t)):
            if(data_t[j] > 1):
                data_t[j] = 1;
        target.append(data_t);
    return List,target;

import GMM_kp_sample;



data = dataGetPromise12()
#data = dataGetNasa7()
data,target = dataPreTreated(data);
print(len(data));
print(data[0].shape);
print(data[0][0])


#获得JS信息
JS = np.loadtxt('..\\js-结果\\promise12-NEW.txt', dtype=np.float,delimiter=',');
#JS = np.loadtxt('..\\js-结果\\nasa7-NEW.txt', dtype=np.float,delimiter=',');

print(JS)
maxjs = [];
for maxi in  range(len(JS)):
    maxjs.append(np.argsort(JS[maxi])[1]);
print("maxjs",maxjs);

from sklearn.metrics import classification_report
from liblinearutil import *
from sklearn.metrics import confusion_matrix
k = [0.25,0.5,1,2,4];
percent = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
size = len(data)

list_f_binary = np.zeros((size,2))
list_g_mean = np.zeros((size,2))
list_auc = np.zeros((size,2))


#当前测试的项目为第i个项目
for numi in range(0,size):
    #根据js找到当前最接近的项目
    jsChoose = maxjs[numi];
    print("number",numi)
    print(jsChoose) 
    #运行1次
    for time in range(0,1):
        max_f_binary = 0
        max_g_mean = 0
        max_auc = 0

        for ki in range(5):
            for percenti in range(9):
                ros = GMM_kp_sample.GMM_Separate(k=k[ki],percent=percent[percenti]);
                X_resampled, y_resampled = ros.fit_sample(data[jsChoose],target[jsChoose])

                data_test = data[numi];
                target_test = target[numi];

                y_resampled_count = sum(y_resampled == 1);
                y_resampled_count_0 = sum(y_resampled == 0);


                #某一项太少了不执行此项
                if(y_resampled_count <= 1) | (y_resampled_count_0 <= 1):
                    f_binary = 0
                    auc = 0
                    g_mean = 0
                else:
                    clf = train(y_resampled.ravel(), X_resampled,'-s 0')
                    p_label, p_acc, p_val = predict(target_test.ravel(), data_test, clf)
                    #mnb = KNeighborsClassifier(n_neighbors=5);
                    #mnb = GaussianNB();  # 使用默认配置初始化朴素贝叶斯
                    #mnb.fit(X_resampled,y_resampled.ravel()) 
                    #p_label = mnb.predict(np.array(data_test));

                    print(classification_report(target_test, p_label))
                    f_binary = f1_score(target_test, p_label, average="binary");
                    auc = roc_auc_score(target_test, p_label);
                    C_matrix = confusion_matrix(target_test, p_label,labels=[1,0])
                    tp = C_matrix[0][0]
                    tn = C_matrix[1][1]
                    fp = C_matrix[1][0]
                    fn = C_matrix[0][1]
                    re = tp/(tp+fn)
                    pf_1 = tn/(tn + fp)
                    g_mean = math.sqrt( pf_1 * re )

                if(max_auc < auc):
                    max_f_binary = f_binary;
                    max_g_mean = g_mean;
                    max_auc = auc;

        list_f_binary[numi][time] = max_f_binary;
        list_g_mean[numi][time] = max_g_mean;
        list_auc[numi][time] = max_auc;

print("f_binary")
for var_f_binary in list_f_binary.tolist():
    print(var_f_binary)
print("g_mean")
for var_g_mean in list_g_mean.tolist():
    print(var_g_mean)
print("list_auc")
for var_auc in list_auc.tolist():
    print(var_auc)
