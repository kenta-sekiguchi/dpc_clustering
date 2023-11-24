import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.cluster import DBSCAN
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import confusion_matrix

from mypackages import preprocessing


def dbscan_cluster(datas, ans, eps, min_samples, pic=True):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labs = dbscan.fit_predict(datas)
    
    if -1 in labs:
        cluster_num = len(np.unique(labs))-1
        
    else:
        cluster_num = len(np.unique(labs))
        
    
    cluster_accuracy =preprocessing.accuracy(ans, labs)
    ARI = adjusted_rand_score(ans, labs)
    NMI = normalized_mutual_info_score(ans, labs)
    # print('MO', MO)
    # print('Md', Md)
    # print('MO_d', MO_d)
    print('クラスタ数：', cluster_num)
    print('正解率：', cluster_accuracy)
    print('ARI：',  ARI)
    print('NMI：', NMI)
    
    if pic==True:
        draw_cluster2(datas, labs)
    
    return cluster_num, labs

def dbscan_time(datas, ans, eps, min_samples):
    
    start = time.time()
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    
    labs = dbscan.fit_predict(datas)
    fin = time.time()
    print('実行時間：{:.5f}'.format(fin-start))
    

# 勝手にグリッドサーチをしてくれるように定義
# パラメータの決定方法はshared nearest neighborを参照
def dbscan_cluster2(datas, ans, eps_list=np.linspace(0, 1, 101)[1:], min_samples_list=np.linspace(0, 50, 51)[1:], pic=True):
    
    cluster_accuracy_default = -np.inf
    ARI_default = -np.inf
    NMI_default = -np.inf
    cluster_num = 0
    eps_default = 0
    min_samples_default = 0
    
    for eps in eps_list:
        for min_samples in min_samples_list:
            
            dbscan = DBSCAN(eps=eps, min_samples=int(min_samples))
            labs = dbscan.fit_predict(datas)
            labs_copy = labs.copy()
            
            # 書き足した部分
            out = len(labs[labs==-1])
            out_index = np.where(labs==-1)[0]
            
            for i, index in enumerate(out_index):
                labs[index] = np.unique(labs)[-1] + 1
            
            cluster_accuracy = preprocessing.accuracy(ans, labs)
            ARI = adjusted_rand_score(ans, labs)
            NMI = normalized_mutual_info_score(ans, labs)
            
            
            if cluster_accuracy_default < cluster_accuracy:
                cluster_accuracy_default = cluster_accuracy
                ARI_default = ARI
                NMI_default = NMI 
                labs_def = labs

                if -1 in labs_copy:
                    cluster_num = len(np.unique(labs_copy))-1

                else:
                    cluster_num = len(np.unique(labs_copy))

                eps_default = eps
                min_samples_default = min_samples
    
    if pic==True:
        draw_cluster2(datas, labs_def)
        
    print('eps：', eps_default, 'minpts：', min_samples_default)
    print('クラスタ数：', cluster_num)
    print('正解率：', cluster_accuracy_default)
    print('ARI：',  ARI_default)
    print('NMI：', NMI_default)
    
    return labs_def


def dbscan_cluster3(datas, ans, eps_list=np.linspace(0, 1, 101)[1:], min_samples_list=np.linspace(0, 50, 51)[1:], pic=True):
    
    cluster_accuracy_default = -np.inf
    ARI_default = -np.inf
    NMI_default = -np.inf
    cluster_num = 0
    eps_default = 0
    min_samples_default = 0
    
    for eps in eps_list:
        for min_samples in min_samples_list:
            
            dbscan = DBSCAN(eps=eps, min_samples=int(min_samples))
            labs = dbscan.fit_predict(datas)
            labs_copy = labs.copy()
            
            cluster_accuracy = preprocessing.accuracy(ans, labs)
            ARI = adjusted_rand_score(ans, labs)
            NMI = normalized_mutual_info_score(ans, labs)
            
            
            if cluster_accuracy_default < cluster_accuracy:
                cluster_accuracy_default = cluster_accuracy
                ARI_default = ARI
                NMI_default = NMI 
                labs_def = labs

                if -1 in labs_copy:
                    cluster_num = len(np.unique(labs_copy))-1

                else:
                    cluster_num = len(np.unique(labs_copy))

                eps_default = eps
                min_samples_default = min_samples
    
    if pic==True:
        draw_cluster2(datas, labs_def)
        
    print('eps：', eps_default, 'minpts：', min_samples_default)
    print('クラスタ数：', cluster_num)
    print('正解率：', cluster_accuracy_default)
    print('ARI：',  ARI_default)
    print('NMI：', NMI_default)
    
    return labs_def




def draw_cluster2(datas, labs):
    
    dic_colors = {0:(.8,0,0),1:(0,.8,0),2:(0,0,.8),3:(.8,.8,0),4:(.8,0,.8),5:(0,.8,.8),6:(0,0,0), 
                  7:(.4, 0, 0), 8:(0, .4, 0), 9:(0, 0, .4), 10:(.4, .4, 0), 11:(.4, 0, .4), 12:(0, .4, .4), 
                  13:(1,0,0),14:(0,1,0), 15:(0,0,1),16:(1,1,0), 17:(1,0,1), 18:(0,1,1), 19:(1,1,1), 
                  20:(.2,0,0), 21:(0,.2,0), 22:(0,0,.2), 23:(.2,.2,0), 24:(.2,0,.2), 25:(0,.2,.2),
                  26:(.6,0,0), 27:(0,.6,0),28:(0,0,.6), 29:(.6,.6,0),30:(.6,0,.6), 31:(0,.6,.6),}
    
    plt.cla()
    K = np.unique(labs)
    df_new = pd.DataFrame(datas)
    df_new[2] = labs
    # sns.scatterplot(data=df_new, x=0, y=1, hue=2, palette='Paired_r', legend=False)
    
    
    for k in K:
        sub_index = np.where(labs == k)
        sub_datas = datas[sub_index]
        
        if k != -1:
            plt.scatter(sub_datas[:, 0],sub_datas[:, 1],s=16.,color=dic_colors[k%32])  
            
        else:
            plt.scatter(sub_datas[:, 0], sub_datas[:, 1], s=16., color='black', marker='x')
            
            

'''
-------------------------------
パッケージなしDBSCAN
-------------------------------
'''

def MyDBSCAN(dists, eps, MinPts):
    """
    Cluster the dataset `D` using the DBSCAN algorithm.
    
    MyDBSCAN takes a dataset `D` (a list of vectors), a threshold distance
    `eps`, and a required number of points `MinPts`.
    
    It will return a list of cluster labels. The label -1 means noise, and then
    the clusters are numbered starting from 1.
    """
    
    labels = np.full(len(dists), -2)

    # Cはクラスタラベル  
    C = 0
    
    for P in range(0, len(dists)):
        
    
        # 選択した点がほかのクラスタに割り当てられている場合はパス
        if not (labels[P] == -2):
            continue
        
        #　点Pのeps内の点を探索
        NeighborPts = find_region(dists, P, eps)

        if len(NeighborPts) < MinPts:
            labels[P] = -1

        else: 
            growCluster(dists, labels, P, NeighborPts, C, eps, MinPts)
            C += 1
    
    return labels


def growCluster(dists, labels, P, NeighborPts, C, eps, MinPts):
    """
    Grow a new cluster with label `C` from the seed point `P`.
    
    This function searches through the dataset to find all points that belong
    to this new cluster. When this function returns, cluster `C` is complete.
    
    Parameters:
      `dists`      - 全ての点間の距離
      `labels` - List storing the cluster labels for all dataset points
      `P`      - Index of the seed point for this new cluster
      `NeighborPts` - All of the neighbors of `P`
      `C`      - The label for this new cluster.  
      `eps`    - Threshold distance
      `MinPts` - Minimum required number of neighbors
    """

    # はじめに選択した点にクラスタラベルCを付与
    labels[P] = C

    i = 0
    while i < len(NeighborPts):    
        
        # Get the next point from the queue.        
        Pn = NeighborPts[i]
       
        # 外れ値が割り当てられている場合の処理
        if labels[Pn] == -1:
            labels[Pn] = C
        
        # 未割当の点の処理
        elif labels[Pn] == -2:
            
            labels[Pn] = C
            
            # Pnの周囲にminpts以上の点があるかどうかの確認
            PnNeighborPts = find_region(dists, Pn, eps)

            if len(PnNeighborPts) >= MinPts:
                NeighborPts = np.append(NeighborPts, PnNeighborPts)
            #else:
                # Do nothing                
                #NeighborPts = NeighborPts   
        
        i += 1

def find_region(dists, P, eps):
    
    neighbors = np.where(dists[P]<=eps)[0]
    
    return neighbors



def dbscan_faces(dists, ans, eps, MinPts):
    
    labs = MyDBSCAN(dists, eps, MinPts)
    
    if -1 in labs:
        cluster_nums = len(np.unique(labs))
    else:
        cluster_nums = len(np.unique(labs))
    
    cluster_accuracy = preprocessing.accuracy(ans, labs)
    ARI = adjusted_rand_score(ans, labs)
    NMI = normalized_mutual_info_score(ans, labs)

    print('正解率：', cluster_accuracy)
    print('ARI：',  ARI)
    print('NMI：', NMI)
    
    return cluster_nums, labs



def dbscan_faces_for(dists, ans, eps_list=np.linspace(0, 1, 101)[1:], min_samples_list=np.linspace(0, 50, 51)[1:]):
    

    cluster_accuracy_default = -np.inf
    ARI_default = -np.inf
    NMI_default = -np.inf
    cluster_num = 0
    eps_default = 0
    min_samples_default = 0
    
    for eps in eps_list:
        for min_samples in min_samples_list:
            
            labs = MyDBSCAN(dists, eps=eps, MinPts=int(min_samples))
            
            cluster_accuracy = preprocessing.accuracy(ans, labs)
            ARI = adjusted_rand_score(ans, labs)
            NMI = normalized_mutual_info_score(ans, labs)
            
            
            if cluster_accuracy_default < cluster_accuracy:
                cluster_accuracy_default = cluster_accuracy
                ARI_default = ARI
                NMI_default = NMI 
                labs_def = labs

                eps_default = eps
                min_samples_default = min_samples
        
    print('eps：', eps_default, 'minpts：', min_samples_default)
    print('クラスタ数：', cluster_num)