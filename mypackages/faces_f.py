import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
from sklearn.neighbors import NearestNeighbors

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.semi_supervised import LabelPropagation
from sklearn import metrics, datasets
from sklearn.metrics import mean_squared_error,accuracy_score,mean_absolute_error,f1_score
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score

from sklearn.datasets import fetch_olivetti_faces
from skimage import data
from skimage import color
from skimage import img_as_float
from skimage.color import rgb2gray
import cv2
import ssim.ssimlib as pyssim
from skimage.metrics import structural_similarity as ssim
from sklearn.cluster import SpectralClustering, AffinityPropagation

import warnings

from mypackages import preprocessing
from mypackages import assign
from mypackages import merge
from mypackages import DPC

dic_colors = {0:(.8,0,0),1:(0,.8,0),2:(0,0,.8),3:(.8,.8,0),4:(.8,0,.8),5:(0,.8,.8),6:(0,0,0), 7:(.4, 0, 0), 8:(0, .4, 0), 9:(0, 0, .4), 10:(.4, .4, 0), 11:(.4, 0, .4), 12:(0, .4, .4), 13:(1,0,0),14:(0,1,0), 15:(0,0,1),16:(1,1,0), 17:(1,0,1), 18:(0,1,1), 19:(1,1,0.1), 20:(.2,0,0), 21:(0,.2,0), 22:(0,0,.2), 23:(.2,.2,0), 24:(.2,0,.2), 25:(0,.2,.2),26:(.6,0,0), 27:(0,.6,0),28:(0,0,.6), 29:(.6,.6,0),30:(.6,0,.6), 31:(0,.6,.6),}



'''
k近傍法
'''

def NearestNeighbors_faces(dists, n_neighbors):
    
    dists_sort = np.sort(dists, axis=1)
    dists_sort_index = np.argsort(dists, axis=1)
    
    KNN_r = dists_sort_index[:, 1:(n_neighbors+1)]
    knn_r = dists_sort_index[:, n_neighbors]
    KNN_r_dist = dists_sort[:, 1:(n_neighbors+1)]
    knn_r_dist = dists_sort[:, n_neighbors]
    
    return KNN_r, knn_r, KNN_r_dist, knn_r_dist


'''
密度計算
'''

def get_density_knn_faces(dists, k):
    N = dists.shape[0]
    
    dist_asc_index, knn_r, dist_asc, knn_r_dist = NearestNeighbors_faces(dists, k)
    
    rho = np.exp(-(1/k)*(np.sum(dist_asc**2, axis=1)))
        
    return rho, dist_asc, dist_asc_index

'''
密度計算
'''

def get_density_knn_faces_2(dists, k):
    N = dists.shape[0]
    
    dist_asc_index, knn_r, dist_asc, knn_r_dist = NearestNeighbors_faces(dists, k)
    
    rho = (1/k) * (np.sum(np.exp(-(dist_asc**2)), axis=1))
        
    return rho, dist_asc, dist_asc_index


'''
Natural Neighbor
'''
def natural_neighbor_faces(dists):
    
    r = 1
    N = dists.shape[0]
    Nan_Num = np.zeros(dists.shape[0])
    Nan_Edge = np.zeros((dists.shape[0], dists.shape[0]))
    rep = 0
    cnt = dists.shape[0]
    distance_max = 0
    distance_min = np.inf
    
    while True:
        
        KNN_r, knn_r, KNN_r_dist, knn_r_dist = NearestNeighbors_faces(dists, r)
        
        for x_i in range(N):
            if x_i in KNN_r[knn_r[x_i]] and Nan_Edge[knn_r[x_i], x_i]==0:
                Nan_Edge[x_i, knn_r[x_i]] = 1
                Nan_Edge[knn_r[x_i], x_i] = 1
                Nan_Num[x_i] += 1
                Nan_Num[knn_r[x_i]] += 1
                
                if distance_max <= knn_r_dist[x_i]:
                    distance_max = knn_r_dist[x_i]
                
                if distance_min >= knn_r_dist[x_i]:
                    distance_min = knn_r_dist[x_i]
                
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
        
        
    return r, Nan_Num, Nan_Edge, KNN_r



def get_representative_nartural_faces(rho, r, dist_asc_index, real_center):
    
    N = rho.shape[0]
    
    while True:
        rep = np.full(N, np.nan)

        dist_asc_index = dist_asc_index[:, :r]


        # 代表点を選択する
        # 密度中心点が複数ある場合は最も大きい密度に割り当てる
        for i in range(N):

            # 自分自身の密度とｋ近傍の点のどちらの密度が大きいかを判断
            # 大きいほうをMrho_iに、代表点候補のindexをrep_iにする
            if max(rho[dist_asc_index[i]]) >= rho[i]:
                Mrho_i = max(rho[dist_asc_index[i]])
                rep_i = dist_asc_index[i][np.where(rho[dist_asc_index[i]]==Mrho_i)[0][0]]

            else:
                Mrho_i = rho[i]
                rep_i = i

            # 代表点の更新
            # 現在の代表点の点の密度よりも、新たな点の密度が大きければ代表点を更新する
            if np.isnan(rep[i]):
                rep[i] = rep_i

            elif Mrho_i >= rho[int(rep[i])]:
                rep[i] = rep_i

            for x_j in dist_asc_index[i]:

                if np.isnan(rep[x_j]):
                    rep[x_j] = rep_i

                elif Mrho_i >= rho[int(rep[x_j])]:
                    rep[x_j] = rep_i

        # 代表点遷移
        rep_unique = np.unique(rep)
        rep_of_rep = rep[rep_unique.astype(int)]

        non_rep = np.setdiff1d(rep_unique, np.unique(rep_of_rep))

        while True:

            for non_rep_i in non_rep:
                rep = np.where(rep==non_rep_i, rep_of_rep[np.where(rep_unique==non_rep_i)[0][0]], rep)

            # 更新後の中心点のユニークな値
            rep_unique = np.unique(rep)

            # 更新後の中心点の中心点
            rep_of_rep = rep[rep_unique.astype(int)]

            # 更新後の中心点のユニークな値と中心点の中心点のユニークな値が等しければbreak
            if len(rep_unique) == len(np.unique(rep_of_rep)):
                break

            non_rep = np.setdiff1d(rep_unique, np.unique(rep_of_rep))


        rep_data = rep.copy()
        centers = np.unique(rep_of_rep).astype(int)
        
        '''
        ------------------------------変更箇所--------------------------------------
        '''
        
        if len(centers)>=real_center:
            new_r = r
            print('近傍数', new_r)
            break
            
        # if len(centers)>real_center:
        #     new_r = r
        #     print('近傍数', new_r)
        #     break
        
        '''----------------------------------------    '''
        
        print('rの値を変更しました', r)
        r = r-1
    
    return rep_data, centers, new_r 


def get_cluster_distance_faces(dists, labs):

    cluster_distance = np.zeros([len(np.unique(labs)), len(np.unique(labs))])
    
    for m in range(len(np.unique(labs)-1)):
        for n in range(m+1, len(np.unique(labs))):
            cluster_M = np.where(labs==m)[0]
            cluster_N = np.where(labs==n)[0]

            cluster_distance[m, n] = np.min(dists[cluster_M][:, cluster_N])
            
    return cluster_distance


def merge_cluster_4_faces(dists, labs, dist_asc_index, real_center, rho):
    
    K = len(np.unique(labs))

                
    if K == real_center:
        return labs

            
    MOL = np.array([])
    Md = np.array([])
    labs_1 = labs.copy()
    labs_2 = labs.copy()
    
    labs_memory = labs.copy()
    
    OL_K = merge.get_overlapping_2(labs_1, dist_asc_index)
    OL_K_copy = OL_K.copy()
    # cluster_distance = get_cluster_distance(dists, labs_1)
    cluster_distance = get_cluster_distance_faces(dists, labs_1)
    cluster_distance_copy = cluster_distance.copy()
    
    MOL_K = np.max(OL_K)
    if MOL_K == 0:
        
        return labs_1

    
    for j in range(K-real_center):
        
        # オーバーラッピング計算
        # 式4
        MOL_K = np.max(OL_K)
        cluster_distance_max = np.max(cluster_distance)
        OL_K_copy = OL_K.copy()
        cluster_distance_copy = cluster_distance.copy()
        
        if MOL_K != 0:

            OL_K_copy = OL_K_copy/MOL_K
            cluster_distance_copy = cluster_distance_copy/cluster_distance_max
            # OL_K_copy = np.where(OL_K_copy==0, -np.inf, OL_K_copy)
            cluster_distance_copy = np.where(cluster_distance_copy==0, np.inf, cluster_distance_copy)


            OL_K_cluster = OL_K_copy + (1-cluster_distance_copy)
            p = np.where(OL_K_cluster==np.max(OL_K_cluster))[0][0]
            q = np.where(OL_K_cluster==np.max(OL_K_cluster))[1][0]            


            MOL = np.append(MOL, OL_K[p, q])
            Md = np.append(Md, cluster_distance[p, q])

            cluster_P = np.where(labs_1==p)[0]
            cluster_Q = np.where(labs_1==q)[0]


            # クラスタのマージ
            labs_1[cluster_Q] = p
            labs_1 = np.where(labs_1>q, labs_1-1, labs_1)
            labs_memory = np.vstack((labs_memory, labs_1))

            # OL行列の更新
            # q行目、q列目の削除（マージしたクラスタ）
            OL_K = np.delete(OL_K, q, axis=0)
            OL_K = np.delete(OL_K, q, axis=1)

            # p行、列目の更新
            N = OL_K.shape[0]
            p_column = np.zeros(N)
            p_index = np.zeros(N)
            cluster_M = np.where(labs_1==p)[0]
            M_size = len(cluster_M)

            if p != 0:
                for i in range(p):
                    cluster_N = np.where(labs_1==i)[0]
                    N_size = len(cluster_N)
                    SN_i = 0
                    SN_j = 0

                    for x_i in cluster_M:
                        knn_i = dist_asc_index[x_i]
                        SN_i += len(set(knn_i) & set(cluster_N))

                    for x_j in cluster_N:
                        knn_j = dist_asc_index[x_j]
                        SN_j += len(set(knn_j) & set(cluster_M))  

                    p_column[i] = (SN_i + SN_j)/(M_size + N_size)

                for i in range(p+1, N):
                    cluster_N = np.where(labs_1==i)[0]
                    N_size = len(cluster_N)
                    SN_i = 0
                    SN_j = 0

                    for x_i in cluster_M:
                        knn_i = dist_asc_index[x_i]
                        SN_i += len(set(knn_i) & set(cluster_N))

                    for x_j in cluster_N:
                        knn_j = dist_asc_index[x_j]
                        SN_j += len(set(knn_j) & set(cluster_M))  

                    p_index[i] = (SN_i + SN_j)/(M_size + N_size)  

                OL_K[p] = p_index
                OL_K[:, p] = p_column

            else:
                for i in range(p+1, N):
                    cluster_N = np.where(labs_1==i)[0]
                    N_size = len(cluster_N)
                    SN_i = 0
                    SN_j = 0

                    for x_i in cluster_M:
                        knn_i = dist_asc_index[x_i]
                        SN_i += len(set(knn_i) & set(cluster_N))

                    for x_j in cluster_N:
                        knn_j = dist_asc_index[x_j]
                        SN_j += len(set(knn_j) & set(cluster_M))  

                    p_index[i] = (SN_i + SN_j)/(M_size + N_size)  

                OL_K[p] = p_index

            # cluster_distanceの更新（最小距離の場合）
            # q行目、q列目の削除（マージしたクラスタ）
            cluster_distance = np.delete(cluster_distance, q, axis=0)
            cluster_distance = np.delete(cluster_distance, q, axis=1)

            # p行、列目の更新
            p_dist_column = np.zeros(N)
            p_dist_index = np.zeros(N)
            M_size = len(cluster_M)

            if p != 0:
                for i in range(p):
                    cluster_N = np.where(labs_1==i)[0]
                    p_dist_column[i] = np.min(dists[cluster_N][:, cluster_M])

                for i in range(p+1, N):
                    cluster_N = np.where(labs_1==i)[0]
                    p_dist_index[i] = np.min(dists[cluster_M][:, cluster_N])

                cluster_distance[p] = p_dist_index
                cluster_distance[:, p] = p_dist_column

            else:
                for i in range(p+1, N):
                    cluster_N = np.where(labs_1==i)[0]
                    p_dist_index[i] = np.min(dists[cluster_M][:, cluster_N])

                cluster_distance[p] = p_dist_index
            
        else:
            cluster_distance_copy = cluster_distance_copy/cluster_distance_max

            # OL_K_cluster = (1-OL_K_copy) + cluster_distance_copy
            # p = np.where(OL_K_cluster==np.min(OL_K_cluster))[0][0]
            # q = np.where(OL_K_cluster==np.min(OL_K_cluster))[1][0]

            OL_K_cluster = cluster_distance_copy
            OL_K_cluster = np.where(OL_K_cluster==0, np.inf, OL_K_cluster)
            p = np.where(OL_K_cluster==np.min(OL_K_cluster))[0][0]
            q = np.where(OL_K_cluster==np.min(OL_K_cluster))[1][0]
            
            print(p, q)


            MOL = np.append(MOL, 0)
            Md = np.append(Md, cluster_distance[p, q])

            cluster_P = np.where(labs_1==p)[0]
            cluster_Q = np.where(labs_1==q)[0]


            # クラスタのマージ
            labs_1[cluster_Q] = p
            labs_1 = np.where(labs_1>q, labs_1-1, labs_1)
            labs_memory = np.vstack((labs_memory, labs_1))


            # cluster_distanceの更新（最小距離の場合）
            # q行目、q列目の削除（マージしたクラスタ）
            cluster_distance = np.delete(cluster_distance, q, axis=0)
            cluster_distance = np.delete(cluster_distance, q, axis=1)
            
            # p行、列目の更新
            N = cluster_distance.shape[0]
            cluster_M = np.where(labs_1==p)[0]
            M_size = len(cluster_M)

            # p行、列目の更新
            p_dist_column = np.zeros(N)
            p_dist_index = np.zeros(N)

            if p != 0:
                for i in range(p):
                    cluster_N = np.where(labs_1==i)[0]
                    p_dist_column[i] = np.min(dists[cluster_N][:, cluster_M])

                for i in range(p+1, N):
                    cluster_N = np.where(labs_1==i)[0]
                    p_dist_index[i] = np.min(dists[cluster_M][:, cluster_N])

                cluster_distance[p] = p_dist_index
                cluster_distance[:, p] = p_dist_column

            else:
                for i in range(p+1, N):
                    cluster_N = np.where(labs_1==i)[0]
                    p_dist_index[i] = np.min(dists[cluster_M][:, cluster_N])

                cluster_distance[p] = p_dist_index
        
    K_star = real_center
    
    return labs_memory[K-K_star]







'''
DPC_顔面用
'''

def density_peaks_cluster_faces(dists, percent, center_num, ans, name, method='Gaussian'):
    
    dc = DPC.select_dc(dists, percent)
    # dc = 0.07
    print("カットオフ距離", dc)
    rho = DPC.get_density(dists, dc, method)# we can use other distance such as 'manhattan_distance'
    deltas, nearest_neighbor= DPC.get_deltas(dists,rho)
    # draw_decision(rho,deltas)
    centers = DPC.find_centers_K(rho,deltas, center_num)
    print("cluster-centers",centers)
    labs = DPC.cluster_PD(rho,centers,nearest_neighbor)

    print('正解率：', preprocessing.accuracy(ans, labs))
    print('ARI：',  adjusted_rand_score(ans, labs))
    print('NMI：', normalized_mutual_info_score(ans, labs))
    
    return labs




