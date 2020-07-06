from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import string

def pca_select(X):
    selnum = 3
    num_feat = np.shape(X)[1]
    pca = PCA(n_components=5)
    pca.fit(X)
    evla = abs(np.dot(pca.explained_variance_,pca.components_))
    signif_ind = np.argsort(evla)[selnum-1:]
    print(signif_ind)
    return X[:,signif_ind]
mat = pca.components_
envdata = np.dot(mat,X.T).T

def constructEnv(envdata):
    gradnum = 10
    boundmax = max([max(envdata[:,0]),max(envdata[:,1])])+1
    boundmin = min([min(envdata[:,0]),min(envdata[:,1])])-1
    dicten = {}
    for i in range(gradnum):
        for j in range(gradnum):
            dicten[(i,j)] = 0
    delta = (boundmax-boundmin)/gradnum
    for da in envdata:
        (xcoor,ycoor) = getCoor(da[0],da[1],delta,boundmin)
        print(da[0],da[1],(xcoor,ycoor))
        dicten[(xcoor,ycoor)]+=1
    return dicten

def getCoor(x0,y0,delta,boundmin):
    xcoor = int((x0-boundmin)/delta)
    ycoor = int((y0-boundmin)/delta)
    return (xcoor,ycoor)