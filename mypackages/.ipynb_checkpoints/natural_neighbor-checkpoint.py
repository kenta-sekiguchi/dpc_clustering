import numpy as np
from sklearn.neighbors import NearestNeighbors

def natural_neighbor(datas):
    
    '''
    natural neighborを実行する関数
    
    Args:
        datas（numpy配列）:データの配列
        
    Returns:
        r（整数）:natural neighborで求められる最適なk近傍
        Nan_Num（1次元numpy配列）:各点が持つNatural Neihborの数
        Nan_Edge（2次元numpy配列）:各点がどの点とNatural Neighborになっているか
        KNN_r（2次元numpy配列）:各点のk近傍点が格納
    '''
    
    r = 1
    N = datas.shape[0]
    Nan_Num = np.zeros(datas.shape[0])
    Nan_Edge = np.zeros((datas.shape[0], datas.shape[0]))
    rep = 0
    cnt = datas.shape[0]
    distance_max = 0
    distance_min = np.inf
    
    while True:
        nbrs = NearestNeighbors(n_neighbors=r+1, algorithm='ball_tree').fit(datas)
        KNN_r = nbrs.kneighbors(datas)[1][:, 1:]
        knn_r = nbrs.kneighbors(datas)[1][:, -1]
        KNN_r_dist = nbrs.kneighbors(datas)[0][:, 1:]
        knn_r_dist = nbrs.kneighbors(datas)[0][:, -1]
        
        for x_i in range(N):
            if x_i in KNN_r[knn_r[x_i]] and Nan_Edge[knn_r[x_i], x_i]==0:
                Nan_Edge[x_i, knn_r[x_i]] = 1
                Nan_Edge[knn_r[x_i], x_i] = 1
                Nan_Num[x_i] += 1
                Nan_Num[knn_r[x_i]] += 1
                

                
        cnt_new = len(Nan_Num[Nan_Num==0])
        
        if all(Nan_Num != 0) or cnt == cnt_new:
            break

        # if cnt == cnt_new:
        #     rep += 1
        # else:
        #     rep = 0
        
        cnt = cnt_new
            
        # if all(Nan_Num!=0) or (rep >= np.sqrt(r-rep)):
        #     break
        
        r += 1
        
        # if r==3:
        #     break
        
        
    return r, Nan_Num, Nan_Edge, KNN_r, KNN_r_dist