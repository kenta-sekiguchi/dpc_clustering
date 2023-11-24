import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors



# オーバーラッピング度合いを見る（その2）
# 分子は1を足すんじゃなくて、k近傍点が他のクラスタに含まれる数を格納
# 分母は選択した2つのクラスタに含まれる点の数
def get_overlapping_2(labs, dist_asc_index):
    
    '''
    サブクラスタ同士の重なり具合（オーバーラッピング）を見る
    分子：あるクラスタに属する点のk近傍点の中で、対象のクラスタに入っている点の数
    分母：2つのクラスタ点の合計数
    
    Args:
        labs:各点が属するサブクラスタ情報
        dist_asc_index:各点のr近傍までが記載されたデータ
    
    
    Returns:
        OL（2次元numpy配列）:クラスタ同士の重なり具合が記載された2次元numpy配列
    
    '''
    
    N = np.shape(labs)[0]
    # 距離の昇順のインデックス番号を取得

    OL = np.zeros([len(np.unique(labs)), len(np.unique(labs))])
    
    for m in range(len(np.unique(labs)-1)):
        for n in range(m+1, len(np.unique(labs))):
            cluster_M = np.where(labs==m)[0]
            cluster_N = np.where(labs==n)[0]
            M_size = len(cluster_M)
            N_size = len(cluster_N)
            SN_i = 0
            SN_j = 0
        
            for x_i in cluster_M:
                knn_i = dist_asc_index[x_i]
                SN_i += len(set(knn_i) & set(cluster_N))

            for x_j in cluster_N:
                knn_j = dist_asc_index[x_j]
                SN_j += len(set(knn_j) & set(cluster_M))

            OL[m, n] = (SN_i + SN_j)/(M_size + N_size)

            
    return OL



def get_cluster_distance(dists, labs):
    
    '''
    クラスタ間の距離を計測（最小距離の場合）
    
    Args:
        dists（numpy配列）:点間の距離が記載された2次元numpy配列
        labs（numpy配列）：各点のサブクラスタが割り当てられたnumpy配列

    Return:
        cluster_distance（numpy配列）:クラスタ間の距離が格納された上三角行列
    '''

    cluster_distance = np.zeros([len(np.unique(labs)), len(np.unique(labs))])
    
    for m in range(len(np.unique(labs)-1)):
        for n in range(m+1, len(np.unique(labs))):
            cluster_M = np.where(labs==m)[0]
            cluster_N = np.where(labs==n)[0]

            cluster_distance[m, n] = np.min(dists[cluster_M][:, cluster_N])
            
    return cluster_distance


    
def merge_cluster_4(dists, labs, dist_asc_index, real_center):
    '''
    真のクラスタ数までサブクラスタをマージする関数
    
    Args:
        dists（numpy配列）:各点の距離
        labs（numpy配列）:サブクラスタの割り当てが記載
        dist_asc_index（numpy配列）:各点のr近傍点までが記載
        real_center（整数）:真の中心数
    
    Returns:
        最終的なクラスタ結果
    '''
    
    
    K = len(np.unique(labs))

                
    if K == real_center:
        return labs

            
    MOL = np.array([])
    Md = np.array([])
    labs_1 = labs.copy()
    labs_2 = labs.copy()
    
    labs_memory = labs.copy()
    
    OL_K = get_overlapping_2(labs_1, dist_asc_index)
    OL_K_copy = OL_K.copy()
    # cluster_distance = get_cluster_distance(dists, labs_1)
    cluster_distance = get_cluster_distance(dists, labs_1)
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
            OL_K_copy = np.where(OL_K_copy==0, -np.inf, OL_K_copy)

            OL_K_cluster = (1-OL_K_copy) + cluster_distance_copy
            p = np.where(OL_K_cluster==np.min(OL_K_cluster))[0][0]
            q = np.where(OL_K_cluster==np.min(OL_K_cluster))[1][0]



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
            
            print('重複度が0')
            print(p, q)


            MOL = np.append(MOL, 0)
            Md = np.append(Md, cluster_distance[p, q])

            cluster_P = np.where(labs_1==p)[0]
            cluster_Q = np.where(labs_1==q)[0]


            # クラスタのマージ
            labs_1[cluster_Q] = p
            labs_1 = np.where(labs_1>q, labs_1-1, labs_1)
            labs_memory = np.vstack((labs_memory, labs_1))
            
            '''
            -------------追加--------------------------
            '''
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
                
            '''
            -----------------ここまで-----------------------
            '''


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
==================================================================
クラスタをマージする（その5）
dists：距離行列
labs：各点のクラスタ番号が割り当てられてられたnumpy配列
k：何個の近傍点まで考慮するか（k-NN）
K：クラスタ数

重複度のみを考慮

return：上三角行列

experiment14用に改良
代表候補点として選択した点が真のクラスタ数よりも小さい場合の処理を追加
==================================================================
'''    
def merge_cluster_5(datas, dists, labs, k, dist_asc_index, real_center):
    
    K = len(np.unique(labs))

                
    if K == real_center:
        return labs

            
    MOL = np.array([])
    Md = np.array([])
    labs_1 = labs.copy()
    labs_2 = labs.copy()
    
    labs_memory = labs.copy()
    
    OL_K = get_overlapping_2(labs_1, dist_asc_index)
    OL_K_copy = OL_K.copy()
    
    # cluster_distance = get_cluster_distance(dists, labs_1)

    
#     MOL_K = np.max(OL_K)
#     if MOL_K == 0:
        
#         return labs_1

    
    for j in range(K-real_center):
        
        # オーバーラッピング計算
        # 式4
        MOL_K = np.max(OL_K)
        # cluster_distance_max = np.max(cluster_distance)
        
        OL_K_copy = OL_K.copy()
        # cluster_distance_copy = cluster_distance.copy()
        
        if MOL_K != 0:
            OL_K_copy = OL_K_copy/MOL_K
            # cluster_distance_copy = cluster_distance_copy/cluster_distance_max
            # OL_K_copy = np.where(OL_K_copy==0, -np.inf, OL_K_copy)
            # cluster_distance_copy = np.where(cluster_distance_copy==0, np.inf, cluster_distance_copy)

            # OL_K_cluster = (1-OL_K_copy) + cluster_distance_copy
            # p = np.where(OL_K_cluster==np.min(OL_K_cluster))[0][0]
            # q = np.where(OL_K_cluster==np.min(OL_K_cluster))[1][0]
            

            OL_K_cluster = OL_K_copy.copy()
            
            p = np.where(OL_K_cluster==np.max(OL_K_cluster))[0][0]
            q = np.where(OL_K_cluster==np.max(OL_K_cluster))[1][0]



            MOL = np.append(MOL, OL_K[p, q])
            # Md = np.append(Md, cluster_distance[p, q])

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

#             cluster_distanceの更新（最小距離の場合）
#             q行目、q列目の削除（マージしたクラスタ）
            # cluster_distance = np.delete(cluster_distance, q, axis=0)
            # cluster_distance = np.delete(cluster_distance, q, axis=1)

#             # p行、列目の更新
#             p_dist_column = np.zeros(N)
#             p_dist_index = np.zeros(N)
#             M_size = len(cluster_M)

#             if p != 0:
#                 for i in range(p):
#                     cluster_N = np.where(labs_1==i)[0]
#                     p_dist_column[i] = np.min(dists[cluster_N][:, cluster_M])

#                 for i in range(p+1, N):
#                     cluster_N = np.where(labs_1==i)[0]
#                     p_dist_index[i] = np.min(dists[cluster_M][:, cluster_N])

#                 cluster_distance[p] = p_dist_index
#                 cluster_distance[:, p] = p_dist_column

#             else:
#                 for i in range(p+1, N):
#                     cluster_N = np.where(labs_1==i)[0]
#                     p_dist_index[i] = np.min(dists[cluster_M][:, cluster_N])

#                 cluster_distance[p] = p_dist_index
            
        else:
            print('重複度が0')
            cluster_distance = get_cluster_distance(dists, labs_1)
            cluster_distance_copy = cluster_distance.copy()
            cluster_distance_max = np.max(cluster_distance_copy)
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







def merge_cluster_6(dists, labs, dist_asc_index):
    '''
    中心数を自動決定するver
    （1-重複度）とクラスタ間距離の和が最大になったところをクラスタ数とする
    （(1-重複度)+距離が小さいほどマージしやすい→大きい部分はマージしにくいので、無理にマージしている）
    
    
    Args:
        dists（numpy配列）:各点の距離
        labs（numpy配列）:サブクラスタの割り当てが記載
        dist_asc_index（numpy配列）:各点のr近傍点までが記載
    
    Returns:
        最終的なクラスタ結果
    '''
    
    
    K = len(np.unique(labs))

            
    MOL = np.array([])
    Md = np.array([])
    labs_1 = labs.copy()
    labs_2 = labs.copy()
    
    labs_memory = labs.copy()
    
    OL_K = get_overlapping_2(labs_1, dist_asc_index)
    OL_K_copy = OL_K.copy()
    # cluster_distance = get_cluster_distance(dists, labs_1)
    cluster_distance = get_cluster_distance(dists, labs_1)
    cluster_distance_copy = cluster_distance.copy()
    
    MOL_K = np.max(OL_K)

    
    for j in range(K-1):
        
        # オーバーラッピング計算
        # 式4
        MOL_K = np.max(OL_K)
        cluster_distance_max = np.max(cluster_distance)
        OL_K_copy = OL_K.copy()
        cluster_distance_copy = cluster_distance.copy()
        
        if MOL_K != 0:
            OL_K_copy = OL_K_copy/MOL_K
            cluster_distance_copy = cluster_distance_copy/cluster_distance_max
            OL_K_copy = np.where(OL_K_copy==0, -np.inf, OL_K_copy)

            OL_K_cluster = (1-OL_K_copy) + cluster_distance_copy
            p = np.where(OL_K_cluster==np.min(OL_K_cluster))[0][0]
            q = np.where(OL_K_cluster==np.min(OL_K_cluster))[1][0]



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
            
            print('重複度が0')
            print(p, q)


            MOL = np.append(MOL, 0)
            Md = np.append(Md, cluster_distance[p, q])

            cluster_P = np.where(labs_1==p)[0]
            cluster_Q = np.where(labs_1==q)[0]


            # クラスタのマージ
            labs_1[cluster_Q] = p
            labs_1 = np.where(labs_1>q, labs_1-1, labs_1)
            labs_memory = np.vstack((labs_memory, labs_1))
            
            '''
            -------------追加--------------------------
            '''
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
                
            '''
            -----------------ここまで-----------------------
            '''


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
        
        
    # クラスタ数の自動決定ゾーン
    # MOL = MOL/np.max(MOL)
    # Md = Md/np.max(Md)
    MOLd = (1-MOL) + Md
    
    x = np.arange(0, len(MOLd))
    y = MOLd
    x_label = np.arange(len(MOLd)+1, 1, -1)
    
    plt.plot(x, y, marker='o')
    plt.xticks(x, x_label)
    plt.show()
    
    MOLd_diff = np.diff(MOLd)
    K_star = np.argmax(MOLd_diff)+1
    
    # K_star = np.argmax(MOLd)
        
    print('中心数', K-K_star)
    
    
    
    return labs_memory[K_star]
