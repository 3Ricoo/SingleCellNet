import numpy as np
from sklearn import metrics
import string
from sklearn.metrics import roc_auc_score


################# GUO#########################
#### transform to boolean
def to_bool(wei,leng):
    wei1 = wei
    for i in range(5):
        wei1[i][i] = 0
    indsort = np.argsort(abs(wei1))[:,leng-2:leng]
    for i in range(leng):
        ind = indsort[i]
        for j in range(len(ind)):
            if(wei[0][i][ind[j]]>0):
                wei1[i][ind[j]]=1
            else:
                wei1[i][ind[j]]=-1

    for i in range(leng):
        for j in range(leng):
            if(wei1[i][j]!=1 and wei1[i][j]!=-1):
                wei1[i][j] = 0
    return wei1

def valid(y_true,y_pred,avg):
    precision = metrics.precision_score(y_true, y_pred, average=avg)
    recall = metrics.recall_score(y_true, y_pred, average=avg)
    f1 = metrics.f1_score(y_true, y_pred, average=avg)
    return precision,recall,f1


def Guo_result(wei,leng):
    #guo2names = ["'Nanog'","'Gata6'","'Fgf4'","'Fgfr2'","'Gata4'"]
    Guo3264_network = np.array([[0,-1,1,0,0],[-1,1,0,0,1],[0,0,0,1,0],[-1,1,0,0,1],[0,0,0,0,0]])
    Guo3264_network2 = abs(Guo3264_network)
    wei1 = to_bool(wei,leng)
    print(valid(np.reshape(Guo3264_network,25),np.reshape(wei1,25),'macro'))
    print(valid(np.reshape(Guo3264_network,25),np.reshape(wei1,25),'micro'))
    print('auc',roc_auc_score(np.reshape(Guo3264_network2,25),np.reshape(wei1,25)))

############################## SCODE2 ##############################
def get_scode2_benchmark():
    f = open('input_data/sco2/reference_TFTF_network.txt', "r")
    refer = []
    for line in f.readlines():
        #print(line)
        li = line.split('\t',4)
        edge = (int(li[2]),int(li[-1][0:-1]))
        refer.append(edge)
    refer

    benchmat = np.zeros((100,100),dtype =int)
    benchmat[refer[0]]

    for edge in refer:
        benchmat[edge] = 1
        benchmat[edge[1],edge[0]] = 1
    return benchmat

def scode2_result(network,threshold,gnum):
    bench = get_scode2_benchmark()
    network = np.array(network)
    network1 = network.reshape(gnum*gnum)
    netsort = np.sort(abs(network1))
    thres = netsort[threshold]
    boolnet = np.where(abs(network)>thres,1,0)
    print(valid(np.reshape(bench,gnum*gnum),np.reshape(boolnet,gnum*gnum),'macro'))
    print(valid(np.reshape(bench,gnum*gnum),np.reshape(boolnet,gnum*gnum),'micro'))
    print('auc',roc_auc_score(np.reshape(bench,gnum*gnum),network1))
    return 
