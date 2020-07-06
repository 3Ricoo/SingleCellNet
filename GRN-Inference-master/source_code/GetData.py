import numpy as np
import pandas as pd
import string
'''
 process GUO data 
'''
def getGuo():
    bool_data = pd.read_csv("input_data/bool438.csv")
    dist_matrix = pd.read_csv("input_data/guo_distanceMatrix.csv")
    energy_matrix = pd.read_csv("input_data/guo_energyMatrix.csv")
    data = pd.read_csv("input_data/guo438.csv")
    guo_weight = pd.read_csv("input_data/guo_weightMatrix.csv")
    guo2names = ["'Nanog'","'Gata6'","'Fgf4'","'Fgfr2'","'Gata4'"]
    #33 19 13 14 18

    df = data[guo2names]
    guo_data2 = df.values[170:439]
    ## row:cell; column:gene

    guo_dist = dist_matrix.values[170:438,171:439]
    guo_en = energy_matrix.values[0][170:439]
    guo_weight.index = guo_weight.columns[1:]
    weightGuo = guo_weight.loc[guo2names][guo2names].values
    return (guo_data2,weightGuo,guo_en)

def sort_by_time(data,time):
    return data[np.argsort(time)]

def getSco2():
    tf_name = []
    for line in open('input_data/sco2/tf.txt').readlines():
        line = line[0:-1]
        tf_name.append(line)

    time = []
    for line in open('input_data/sco2/time.txt').readlines():
        pos = []
        for i in range(len(line)):
            s = line[i]
            if(s=='\t'):
                pos.append(i)
        time.append(line[pos[0]+1:pos[1]])

    #sort by time

    expdata0 = pd.read_csv('input_data/sco2/sco2_expr.csv')
    expdata = expdata0.values[:,1:]
    expdata1 = sort_by_time(expdata,time)
    expdata1 =  expdata1.astype(np.float32)
    init_weight = np.cov(expdata1.T)
    energydata0 = pd.read_csv('input_data/sco2/sco2_energyMatrix.csv')
    energydata1 = energydata0.values[0]
    energydata1 = sort_by_time(energydata1,time)
    return (expdata1,init_weight,energydata1)
