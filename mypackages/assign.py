import numpy as np
from sklearn.neighbors import NearestNeighbors



"""
----------------実際の割り当て-----------------------
"""

# サブクラスタの生成
# 自分の代表点と同じクラスタに割り当てる
def make_subcluster(rep_data, centers):
    '''
    非中心点を中心候補点に割り当てる。
    1. get_represent_naturalによって求められた各点の代表点に割り当てる
    
    Args:
        rep_data（numpy配列）:各点の代表点（get_represent_naturalによって求められる）
        centers（numpy配列）:中心候補点のインデックス番号が記載されたnumpy配列
    
    Returns:
        labs（numpy配列）:各点の割り当て先が記載されたnumpy配列
    
    '''
    
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


def snn_assign_in_6(dist_asc_index, k, centers, rho):
    
    '''
    非中心点を中心候補点に割り当てる。
    単純に中心候補点のk近傍点のみを割り当てる。（中心候補点は密度の大きいものから選択）
    1. 中心候補点を選択
    2. 中心候補点のk近傍点のみを同じクラスタに割り当てる
    
    Args:
        dist_asc_index（numpy配列）:各点のr近傍点までの情報が記載されたnumpy配列
        k（整数）：近傍数
        centers（numpy配列）:中心候補点のインデックス番号が記載されたnumpy配列
        rho（numpy配列）:各点の密度が記載されたnumpy配列
    
    Returns:
        labs（numpy配列）:各点の割り当て先が記載されたnumpy配列
    
    '''
    
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

def snn_assign_in_10(dist_asc_index, centers, rho, Nan_Edge):
    
    '''
    非中心点を中心候補点に割り当てる。
    単純に中心候補点のk近傍点のみを割り当てる。（中心候補点は密度の大きいものから選択）
    1. 中心候補点を選択
    2. 中心候補点のnatural neighborを割り当てる
    
    Args:
        dist_asc_index（numpy配列）:各点のr近傍点までの情報が記載されたnumpy配列
        centers（numpy配列）:中心候補点のインデックス番号が記載されたnumpy配列
        rho（numpy配列）:各点の密度が記載されたnumpy配列
        Nan_Edge（numpy配列）:その点とnatural neighborの関係にある点の部分が1になっている行列
    
    Returns:
        labs（numpy配列）:各点の割り当て先が記載されたnumpy配列
    
    '''
    
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


def snn_assign_in_11(dist_asc_index, centers, rho, Nan_Edge):
    
    '''
    1. 中心候補点を密度（降順）を基準に並び替え
    2. 中心候補点のmutual nearest neigbhorを割り当て
       （中心候補点に対してすべて行う）
    3. 新たに割り当てた点を密度を基準に並び替え
    4. その点のmutual neigborを割り当て


    12との違い
    全てのcenterのmutualを割り当て→mutualのmutualを割り当て
    
    Args:
        dist_asc_index（numpy配列）:各点のr近傍点までの情報が記載されたnumpy配列
        centers（numpy配列）:中心候補点のインデックス番号が記載されたnumpy配列
        rho（numpy配列）:各点の密度が記載されたnumpy配列
        Nan_Edge（numpy配列）:その点とnatural neighborの関係にある点の部分が1になっている行列
    
    Returns:
        labs（numpy配列）:各点の割り当て先が記載されたnumpy配列
    
    '''
    
    N = len(dist_asc_index)
    labs = -1*np.ones(N).astype(int)
    mutual_list = np.array([])
    centers_copy = centers.copy()
    centers_copy = centers_copy[np.argsort(rho[centers_copy])][::-1]
    
    
    # クラスタ番号の割り当て
    # centerに選ばれている点のインデックス番号の点にクラスタ番号を割り当てる
    labs[centers_copy] = np.arange(len(centers_copy))
    
    
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


def snn_assign_in_12(dist_asc_index, centers, rho, Nan_Edge):
    
    '''
    1. 中心候補点を密度（降順）を基準に並び替え
    2. 中心候補点のmutual nearest neigbhorを割り当て
       （中心候補点1つに対して行う）
    3. 新たに割り当てた点を密度を基準に並び替え
    4. その点のmutual neigborを割り当て
    5. 「2~4」を中心点の数だけ繰り返す


    11との違い
    「1つのcenterのmutualを割り当て→mutualのmutualを割り当て」を繰り返す

    
    Args:
        dist_asc_index（numpy配列）:各点のr近傍点までの情報が記載されたnumpy配列
        centers（numpy配列）:中心候補点のインデックス番号が記載されたnumpy配列
        rho（numpy配列）:各点の密度が記載されたnumpy配列
        Nan_Edge（numpy配列）:その点とnatural neighborの関係にある点の部分が1になっている行列
    
    Returns:
        labs（numpy配列）:各点の割り当て先が記載されたnumpy配列
    '''
    
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

def snn_assign_in_13(dist_asc_index, centers, rho, Nan_Edge, dists):
    
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
    
    '''
    2段階目の割り当て
    まだ割り当てられていない点（中心候補点と中心候補点周辺の点以外）を割り当てる
    
    Args：
        dist_asc_index（numpy配列）:各点のr近傍点までの情報が記載されたnumpy配列
        k（整数）：近傍数
        centers（numpy配列）:中心候補点のインデックス番号が記載されたnumpy配列
        labs（numpy配列）:各点の割り当て先が記載されたnumpy配列

    
    Returns:
        labs（numpy配列）:各点の割り当て先が記載されたnumpy配列
    '''
    
    dist_asc_index_copy = dist_asc_index.copy()

    N = len(dist_asc_index_copy)
    M_matrix = np.zeros([N, len(centers)])

    # 行列M（未割当の点のk近傍点がどのクラスタに属しているかを示した行列）の作成
    # for possible in np.where(labs==-1)[0]:

    #     cluster_num = labs[dist_asc_index_copy[possible]]

    #     for j in range(len(centers)):
    #         M_matrix[possible, j] = np.sum(cluster_num==j)
            
    # 上の処理をfor文を使わずに行う方法
    for possible in np.where(labs==-1)[0]:
        temp = labs[dist_asc_index_copy[possible]][np.where(labs[dist_asc_index_copy[possible]]!=-1)[0]]
        M_matrix[possible, :] = np.bincount(temp, minlength=len(centers))
    
                
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

                nbrs = NearestNeighbors(n_neighbors=k+2, algorithm='ball_tree').fit(datas)
                dist_asc_index_copy = nbrs.kneighbors(datas)[1][:, 1:]

                # M_matrixの更新
                for possible in np.where(labs==-1)[0]:

                    cluster_num = labs[dist_asc_index_copy[possible]]

                    for j in range(len(centers)):
                        M_matrix[possible, j] = np.sum(cluster_num==j)

                m_max = np.max(M_matrix)
            
    
    return labs
            
