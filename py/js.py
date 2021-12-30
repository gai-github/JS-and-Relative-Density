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

from imblearn.over_sampling import RandomOverSampler


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
        target.append(data_t);
    return List,target;

def mygmm(x):
    gmmMean = [];
    gmmCov = [];
    gmmWeight = [];
    gmms = []
    for i in range(len(x)):
        gmm = GaussianMixture(n_components=5, covariance_type='diag', random_state=0);
        gmm.fit(x[i]); 
        means = gmm.means_;
        covs = gmm.covariances_;
        weights = gmm.weights_;

        gmmMean.append(means);
        gmmCov.append(covs);
        gmmWeight.append(weights);
        gmms.append(gmm);

    return gmmMean,gmmCov,gmmWeight,gmms;

def gmm_js(gmm_p,gmm_q,n_samples=10**6):
    X = gmm_p.sample(n_samples)[0]
    #print(X.shape)
    #print(gmm_p.score_samples(X))
    log_p_x = gmm_p.score_samples(X);
    log_q_x = gmm_q.score_samples(X);
    log_mix_x = np.logaddexp(log_p_x,log_q_x);

    Y = gmm_q.sample(n_samples)[0]
    log_p_Y = gmm_p.score_samples(Y);
    log_q_Y = gmm_q.score_samples(Y);
    log_mix_Y = np.logaddexp(log_p_Y,log_q_Y);

    return (log_p_x.mean() - (log_mix_x.mean() - np.log(2)) + log_q_Y.mean() - (log_mix_Y.mean() - np.log(2))) / 2



def js(data,gmmMean,gmmCov,gmmWeight,gmms):
    #JS = [];
    JS = np.zeros((len(data),len(data)));

    for i in range(len(data)):
        for j in range(len(data)):
            result_js = gmm_js(gmms[i],gmms[j]);
            print("i",i,"j",j)
            print(result_js);
            #JS.append(result_js);
            JS[i][j] = result_js;
    np.set_printoptions(precision=16, suppress=True)
    print(JS)
    np.savetxt('..\\JS运行\\promise12-NEWCHECK3.txt',JS,fmt="%.16f",delimiter=',',newline='\n')
    #np.savetxt('..\\JS运行\\nasa7-NEWCHECK3.txt',JS,fmt="%.16f",delimiter=',',newline='\n')
    


#获取数据 
data = dataGetPromise12();
#data = dataGetNasa7()


data,target = dataPreTreated(data);
print(len(data));
print(data[0].shape);
print(data[0][0])
gmmMean,gmmCov,gmmWeight,gmms = mygmm(data);

js(data,gmmMean,gmmCov,gmmWeight,gmms);

