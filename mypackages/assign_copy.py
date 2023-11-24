import numpy as np
from sklearn.neighbors import NearestNeighbors

"""
----------------共有近傍点の数え方-----------------------
"""

'''
k近傍点のうち、他の点と共有する点の数を求める。
'''

def snn_count(dist_asc_index):
    
    N = len(dist_asc_index)
    count_matrix = np.zeros([N, N])
    for i in range(N):
        for j in range(i, N):
            if i == j:
                count_matrix[i, j] = 0
            else:
                count_matrix[i, j] = len(np.intersect1d(dist_asc_index[i], dist_asc_index[j]))
                count_matrix[j, i] = count_matrix[i, j] 
    
    return count_matrix

'''
お互いの点が中心点に入っちゃた場合を加味してる
'''
def snn_count_2(dist_asc_index):
    
    N = len(dist_asc_index)
    count_matrix = np.zeros([N, N])
    for i in range(N):
        for j in range(i, N):
            if i == j:
                count_matrix[i, j] = 0
            else:
                if (i in dist_asc_index[j]) & (j in dist_asc_index[i]):
                    count_matrix[i, j] = len(np.intersect1d(dist_asc_index[i], dist_asc_index[j]))+2
                    count_matrix[j, i] = count_matrix[i, j] 
                    
                elif (i in dist_asc_index[j]) or (j in dist_asc_index[i]):
                    count_matrix[i, j] = len(np.intersect1d(dist_asc_index[i], dist_asc_index[j]))+2
                    count_matrix[j, i] = count_matrix[i, j] 
                    
                else:
                    count_matrix[i, j] = len(np.intersect1d(dist_asc_index[i], dist_asc_index[j]))
                    count_matrix[j, i] = count_matrix[i, j] 
    
    return count_matrix


'''

'''

def snn_count_3(dist_asc_index, k):
    
    N = len(dist_asc_index)
    count_matrix = np.zeros([N, N])
    for i in range(N):
        for j in range(i, N):
            if i == j:
                count_matrix[i, j] = 0
            else:
                count_matrix[i, j] = len(np.intersect1d(dist_asc_index[i, :k], dist_asc_index[j, :k]))
                count_matrix[j, i] = count_matrix[i, j] 
    
    return count_matrix


'''
これが理想
'''

def snn_count_4(dist_asc_index, k):
    
    N = len(dist_asc_index)
    count_matrix = np.zeros([N, N])
    for i in range(N):
        for j in range(i, N):
            if i == j:
                count_matrix[i, j] = 0
            else:
                if (i in dist_asc_index[j, :k]) & (j in dist_asc_index[i, :k]):
                    count_matrix[i, j] = len(np.intersect1d(dist_asc_index[i, :k], dist_asc_index[j, :k]))+2
                    count_matrix[j, i] = count_matrix[i, j] 
                    
                elif (i in dist_asc_index[j]) or (j in dist_asc_index[i]):
                    count_matrix[i, j] = len(np.intersect1d(dist_asc_index[i, :k], dist_asc_index[j, :k]))+2
                    count_matrix[j, i] = count_matrix[i, j] 
                    
                else:
                    count_matrix[i, j] = len(np.intersect1d(dist_asc_index[i, :k], dist_asc_index[j, :k]))
                    count_matrix[j, i] = count_matrix[i, j] 
    
    return count_matrix


"""
----------------End-----------------------
"""


"""
----------------実際の割り当て-----------------------
"""

# サブクラスタの生成
# 自分の代表点と同じクラスタに割り当てる
def make_subcluster(rep_data, centers):
    K = np.shape(centers)[0]
    if K == 0:
        print("can not find centers")
        return
    
    N = np.shape(rep_data)[0]
    labs = -1*np.ones(N).astype(int)
    
    # クラスタ番号の割り当て
    # centerに選ばれている点のインデックス番号の点にクラスタ番号を割り当てる
    for i, center in enumerate(centers):
        labs[int(center)] = i
   
    # center以外の点に関しては、その点の代表点に割り当てる
    for i, center in enumerate(rep_data):

        if labs[i] == -1:
            labs[i] = labs[int(center)]
    return labs

'''
6
単純に中心候補点のk近傍点のみを割り当てる。
1. 中心候補点を選択
2. 中心候補点のk近傍点のみを同じクラスタに割り当てる
3. 以上
'''

def snn_assign_in_6(dist_asc_index, k, centers, count_matrix, rho):
    
    N = len(dist_asc_index)
    labs = -1*np.ones(N).astype(int)
    centers_copy = centers.copy()
    centers_copy = centers_copy[np.argsort(rho[centers_copy])][::-1]
    
    # クラスタ番号の割り当て
    # centerに選ばれている点のインデックス番号の点にクラスタ番号を割り当てる
    for i, center in enumerate(centers_copy):
        labs[center] = i
    
    
    for center in centers_copy:
        
        points = dist_asc_index[center]
        points = points[labs[points]==-1]
        labs[points] = labs[center]
    
        
    return labs


'''
9
1. 中心候補点を選択（中心候補点を密度の降順で並び替える）
2. 中心候補点のk近傍点を、中心候補点と同じクラスタに割り当てる
3. 中心候補点と共有する近傍点がk/2以上の点を探す（「2.」で割り当てた点も含むことに注意）
4. その点とそのk近傍点を追加
'''

def snn_assign_in_9(dist_asc_index, k, centers, count_matrix, rho):
    
    N = len(dist_asc_index)
    labs = -1*np.ones(N).astype(int)
    centers_copy = centers.copy()
    centers_copy = centers_copy[np.argsort(rho[centers_copy])][::-1]
    
    # クラスタ番号の割り当て
    # centerに選ばれている点のインデックス番号の点にクラスタ番号を割り当てる
    for i, center in enumerate(centers_copy):
        labs[center] = i
    
    for center in centers_copy:
        
        # 中心候補点のk近傍点を割り当てる
        center_k = dist_asc_index[center, :k]
        center_k = center_k[labs[center_k]==-1]
        labs[center_k] = labs[center]
        
        # 中心候補点と共有するk近傍点がk/2以上である点を求める（index）
        index = np.where(count_matrix[center] > k//2)[0]
        index_yet = index[np.where(labs[index]==-1)[0]]
        
        # indexのk近傍点を求める
        index_around = np.unique(np.ravel(dist_asc_index[index, :k]))
        index_around = index_around[np.where(labs[index_around]==-1)[0]]
        all_points = np.unique(np.concatenate([index_yet, index_around], 0))
        
        if len(all_points) != 0:
            labs[all_points] = labs[center]
    
        
    return labs


'''
10
MUtual nearest neighborを考える

中心候補点のmutual nearest neighborのみを割り当てる
'''

def snn_assign_in_10(dist_asc_index, k, centers, rho, Nan_Edge):
    
    N = len(dist_asc_index)
    labs = -1*np.ones(N).astype(int)
    centers_copy = centers.copy()
    centers_copy = centers_copy[np.argsort(rho[centers_copy])][::-1]
    
    # クラスタ番号の割り当て
    # centerに選ばれている点のインデックス番号の点にクラスタ番号を割り当てる
    for i, center in enumerate(centers_copy):
        labs[center] = i
    
    for center in centers_copy:
        
        # 中心候補点のmutual nearest neighborを割り当てる
        center_mutual = np.where(Nan_Edge[center]==1)[0]
        center_mutual = center_mutual[labs[center_mutual]==-1]
        labs[center_mutual] = labs[center]
    
        
    return labs


'''
11
MUtual nearest neighborを考える

先に中心候補点のmutualを割り当ててから、mutualのmutualを割り当て

中心候補点のmutual nearest neighborと、そのmutulal~を割り当てる
'''

def snn_assign_in_11(dist_asc_index, centers, rho, Nan_Edge):
    
    N = len(dist_asc_index)
    labs = -1*np.ones(N).astype(int)
    mutual_list = np.array([])
    centers_copy = centers.copy()
    centers_copy = centers_copy[np.argsort(rho[centers_copy])][::-1]
    
    
    # クラスタ番号の割り当て
    # centerに選ばれている点のインデックス番号の点にクラスタ番号を割り当てる
    for i, center in enumerate(centers_copy):
        labs[center] = i
    
    for center in centers_copy:
        
        # 中心候補点のmutual nearest neighborを割り当てる
        center_mutual = np.where(Nan_Edge[center]==1)[0]
        center_mutual = center_mutual[labs[center_mutual]==-1]
        labs[center_mutual] = labs[center]
               
        
        mutual_list = np.append(mutual_list, center_mutual)
        
    
    mutual_list = mutual_list.astype('int')
    
    mutual_list = mutual_list[np.argsort(rho[mutual_list])[::-1]]
    
    
    for mutual in mutual_list:
        mutual_mutual = np.where(Nan_Edge[mutual]==1)[0]
        mutual_mutual = mutual_mutual[labs[mutual_mutual]==-1]
        labs[mutual_mutual] = labs[mutual]
    
        
    return labs


'''
12
MUtual nearest neighborを考える

中心候補点のmutualとmutualのmutualを同時に割り当て


中心候補点のmutual nearest neighborと、そのmutulal~を割り当てる
'''

def snn_assign_in_12(dist_asc_index, centers, rho, Nan_Edge):
    
    N = len(dist_asc_index)
    labs = -1*np.ones(N).astype(int)
    mutual_list = np.array([])
    centers_copy = centers.copy()
    centers_copy = centers_copy[np.argsort(rho[centers_copy])][::-1]
    
    
    # クラスタ番号の割り当て
    # centerに選ばれている点のインデックス番号の点にクラスタ番号を割り当てる
    for i, center in enumerate(centers_copy):
        labs[center] = i
    
    for center in centers_copy:
        
        # 中心候補点のmutual nearest neighborを割り当てる
        center_mutual = np.where(Nan_Edge[center]==1)[0]
        center_mutual = center_mutual[labs[center_mutual]==-1]
        labs[center_mutual] = labs[center]
        
        for mutual in center_mutual:
            mutual_mutual = np.where(Nan_Edge[mutual]==1)[0]
            mutual_mutual = mutual_mutual[labs[mutual_mutual]==-1]
            labs[mutual_mutual] = labs[mutual]            
    
        
    return labs

'''
13
snn_assign_poを使用しないで、mutual nearest neighborだけを使用
1. 中心候補点を密度（降順）を基準に並び替え
2. 中心候補点のmutual nearest neigbhorを割り当て
3. 新たに割り当てた点を密度を基準に並び替え
4. 「3.」の点のmutual nearest neighborを割り当て
5. 「3.」と「4.」を繰り返す
'''

def snn_assign_in_13(dist_asc_index, k, centers, rho, Nan_Edge, dists):
    
    N = len(dist_asc_index)
    labs = -1*np.ones(N).astype(int)
    mutual_list = np.array([])
    centers_copy = centers.copy()
    centers_copy = centers_copy[np.argsort(rho[centers_copy])][::-1]
    
    
    # クラスタ番号の割り当て
    # centerに選ばれている点のインデックス番号の点にクラスタ番号を割り当てる
    for i, center in enumerate(centers_copy):
        labs[center] = i
    
    for center in centers_copy:
        
        # 中心候補点のmutual nearest neighborを割り当てる
        center_mutual = np.where(Nan_Edge[center]==1)[0]
        center_mutual = center_mutual[labs[center_mutual]==-1]
        labs[center_mutual] = labs[center]
               
        
        mutual_list = np.append(mutual_list, center_mutual)
        
    
    mutual_list = mutual_list.astype('int')
    
    mutual_list = mutual_list[np.argsort(rho[mutual_list])[::-1]]
    
    while True:
        mutual_new = np.array([])
    
        for mutual in mutual_list:
            mutual_mutual = np.where(Nan_Edge[mutual]==1)[0]
            mutual_mutual = mutual_mutual[labs[mutual_mutual]==-1]
            labs[mutual_mutual] = labs[mutual]
            
            mutual_new = np.append(mutual_new, mutual_mutual)
            
        mutual_list = np.copy(mutual_new)
        mutual_list = mutual_list.astype('int')
        mutual_list = mutual_list[np.argsort(rho[mutual_list])[::-1]]  
        
        if len(mutual_list) == 0:
            break
            
            
    # 割り当てがされていない点（Natural Neighborのグラフでつながっていない点）の割り当て
    # クラスタに割り当てられていて、最も近い点と同じクラスタに割り当てる
    if -1 in labs:
        not_assigned_points = np.where(labs==-1)[0]

        assigned_points = np.where(labs!=-1)[0]

        for point in not_assigned_points:
            dist_point = dists[point]
            dist_point = np.argsort(dist_point)[1:]
            
            for j in dist_point:
                if j in assigned_points:

                    labs[point]=labs[j]

                    break

                else:
                    continue              
    
    return labs


'''
possible subordinateの割り当て
'''

def snn_assign_po(dist_asc_index, k, centers, labs):
    
    dist_asc_index_copy = dist_asc_index.copy()

    N = len(dist_asc_index_copy)
    M_matrix = np.zeros([N, len(centers)])

    # 行列M（未割当の点のk近傍点がどのクラスタに属しているかを示した行列）の作成
    for possible in np.where(labs==-1)[0]:

        cluster_num = labs[dist_asc_index_copy[possible]]

        for j in range(len(centers)):
            M_matrix[possible, j] = np.sum(cluster_num==j)
            
    m_max = np.max(M_matrix)
    labs_copy = np.zeros(N)
    
    while True:
        
        if m_max != 0:
            # 更新する点のインデックス番号
            max_points = np.where(M_matrix==m_max)[0]

            # 更新する点のクラスタ番号
            max_cluster = np.where(M_matrix==m_max)[1]

            #点の割り当て
            labs[max_points] = max_cluster
            
            # M_matrixの更新
            for point in max_points:
                
                # 新たに割り当てられた点（point）をk近傍点に含む点（rows）を選択
                rows = np.where(dist_asc_index_copy==point)[0]
                # rows = np.setdiff1d(rows, centers)
                
                # rowsの行列Mの値を更新
                M_matrix[rows, labs[point]] = M_matrix[rows, labs[point]] + 1
            
            points_assigned = np.where(labs != -1)
            M_matrix[points_assigned] = 0
            
            m_max = np.max(M_matrix)

            if -1 not in labs:
                break
            
        
        else:
            if -1 not in labs:
                break
            
            else:
                k += 1
                print('kに1を追加します')

                nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(datas)
                dist_asc_index_copy = nbrs.kneighbors(datas)[1][:, 1:]

                # M_matrixの更新
                for possible in np.where(labs==-1)[0]:

                    cluster_num = labs[dist_asc_index_copy[possible]]

                    for j in range(len(centers)):
                        M_matrix[possible, j] = np.sum(cluster_num==j)

                m_max = np.max(M_matrix)
            
    
    return labs
            
