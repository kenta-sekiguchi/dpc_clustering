import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors


from mypackages import preprocessing
from mypackages import DPC
from mypackages import density

dic_colors = {0:(.8,0,0),1:(0,.8,0),2:(0,0,.8),3:(.8,.8,0),4:(.8,0,.8),5:(0,.8,.8),6:(0,0,0), 7:(.4, 0, 0), 8:(0, .4, 0), 9:(0, 0, .4), 10:(.4, .4, 0), 11:(.4, 0, .4), 12:(0, .4, .4), 13:(1,0,0),14:(0,1,0), 15:(0,0,1),16:(1,1,0), 17:(1,0,1), 18:(0,1,1), 19:(1,1,1), 20:(.2,0,0), 21:(0,.2,0), 22:(0,0,.2), 23:(.2,.2,0), 24:(.2,0,.2), 25:(0,.2,.2),26:(.6,0,0), 27:(0,.6,0),28:(0,0,.6), 29:(.6,.6,0),30:(.6,0,.6), 31:(0,.6,.6),}

def density_peak_cluster_kNN(datas, k, center_num, ans, name, pic=True):
    
    dists = DPC.getDistanceMatrix(datas)
    
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(datas)
    dist_asc_index = nbrs.kneighbors(datas)[1][:, 1:]
    dist_asc = nbrs.kneighbors(datas)[0][:, 1:]    
    

    rho = density.get_density_knn(dist_asc, k)
    deltas, nearest_neighbor= DPC.get_deltas(dists,rho)
    
    if pic==True:
        DPC.draw_decision(rho,deltas)
        
    centers = DPC.find_centers_K(rho, deltas, center_num)

    print("cluster-centers",centers)
    labs = DPC.cluster_PD(rho, centers, nearest_neighbor)
    
    if pic==True:
        DPC.draw_cluster(datas,labs,centers,  name)
        
    print('正解率：', preprocessing.accuracy(ans, labs))
    print('ARI：',  adjusted_rand_score(ans, labs))
    print('NMI：', normalized_mutual_info_score(ans, labs))
    
    return labs, centers


def density_peak_cluster_kNN_for(datas, k, center_num, ans, name):
    
    dists = DPC.getDistanceMatrix(datas)

    rho, dist_asc, dist_asc_index = density.get_density_knn(datas, dists, k)
    deltas, nearest_neighbor= DPC.get_deltas(dists,rho)
    # draw_decision(rho,deltas)
    centers = DPC.find_centers_K(rho, deltas, center_num)
    # print("cluster-centers",centers)
    labs = DPC.cluster_PD(rho, centers, nearest_neighbor)
    # draw_cluster(datas,labs,centers, dic_colors, name)

    
    return labs, centers, preprocessing.accuracy(ans, labs), adjusted_rand_score(ans, labs), normalized_mutual_info_score(ans, labs)