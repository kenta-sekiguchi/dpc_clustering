import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import confusion_matrix

from mypackages import preprocessing

dic_colors = {0:(.8,0,0),1:(0,.8,0),2:(0,0,.8),3:(.8,.8,0),4:(.8,0,.8),5:(0,.8,.8),6:(0,0,0), 7:(.4, 0, 0), 8:(0, .4, 0), 9:(0, 0, .4), 10:(.4, .4, 0), 11:(.4, 0, .4), 12:(0, .4, .4), 13:(1,0,0),14:(0,1,0), 15:(0,0,1),16:(1,1,0), 17:(1,0,1), 18:(0,1,1), 19:(1,1,1), 20:(.2,0,0), 21:(0,.2,0), 22:(0,0,.2), 23:(.2,.2,0), 24:(.2,0,.2), 25:(0,.2,.2),26:(.6,0,0), 27:(0,.6,0),28:(0,0,.6), 29:(.6,.6,0),30:(.6,0,.6), 31:(0,.6,.6)}


def getDistanceMatrix(datas):
    N,D = np.shape(datas)
    dists = np.zeros([N,N])
    
    for i in range(N-1):
        for j in range(i+1, N):
            diff = datas[i] - datas[j]
            dists[i, j] = np.dot(diff, diff)
            dists[j, i] = dists[i, j]
    
    return np.sqrt(dists)


# カットオフ距離を求める
# N(N-1)はnC2のこと
#上位position + N番目の点までの距離をカットオフ距離としている
def select_dc(dists, percent):    
    N = np.shape(dists)[0]
    tt = np.triu(dists)
    tt = np.reshape(tt, N*N)
    tt = tt[tt!=0]
    position = int(N*(N-1)/2 * percent/100)
    dc = np.sort(tt)[position]
    
    # tt = np.reshape(dists,N*N)
    # position = int(N * (N - 1) * percent / 100)
    # dc = np.sort(tt)[position  + N]
    
    return dc

'''
----------密度求める部分----------------------------------
'''
# 点iまでの距離がカットオフ距離内に存在する点を求めて、密度を求める
def get_density(dists,dc,method=None):
    N = np.shape(dists)[0]
    rho = np.zeros(N)
    
    if method==None:
        pass
    else:
        rho = np.sum(np.exp(-((dists/dc)**2)), axis=1)
    
    # for i in range(N):
    #     if method == None:
    #         rho[i]  = np.where(dists[i,:]<dc)[0].shape[0]-1
        
        # else:
        #     rho[i] = np.sum(np.exp(-(dists[i,:]/dc)**2))
    return rho

def get_deltas(dists,rho):
    N = np.shape(dists)[0]
    deltas = np.zeros(N)
    nearest_neighbor = np.zeros(N)

    index_rho = np.argsort(-rho)
    for i,index in enumerate(index_rho):

        if i==0:
            continue
        
        # 自身より密度が高い点を探索範囲とする
        index_higher_rho = index_rho[:i]
        
        # index番号の（最初に割り振られている番号）の点のδは、距離行列の中の最小の点
        # deltasには、全ての点の、自分より高密度でかつ最短距離の点までの距離が格納されている
        deltas[index] = np.min(dists[index,index_higher_rho])
        
        
        # index_higher_rhoの中での最小距離となるindex番号を取得
        # nearest_neighborには、全ての点の、自分より高密度点でかつ最短距離の点のインデックス番号が格納されている
        index_nn = np.argmin(dists[index,index_higher_rho])
        nearest_neighbor[index] = index_higher_rho[index_nn].astype(int)
    
    # 最も高密度な点に対するδの処理
    deltas[index_rho[0]] = np.max(dists[index_rho[0]])   
    return deltas,nearest_neighbor


# ρ×δを降順で並べた時のK番目までを代表点として取得  
def find_centers_K(rho,deltas,K):
    rho_delta = rho*deltas
    centers = np.argsort(-rho_delta)
    return centers[:K]


# クラスタ番号の割り当て、密度が大きいやつから順番に
def cluster_PD(rho,centers,nearest_neighbor):
    K = np.shape(centers)[0]
    if K == 0:
        print("can not find centers")
        return
    
    N = np.shape(rho)[0]
    labs = -1*np.ones(N).astype(int)
    
    # クラスタ番号の割り当て
    # centerに選ばれている点のインデックス番号の点にクラスタ番号を割り当てる
    for i, center in enumerate(centers):
        labs[center] = i
   
    # center以外の点に関しては、自分より高密度でかつ最短距離の点のクラスタ番号を割り当てる
    index_rho = np.argsort(-rho)
    for i, index in enumerate(index_rho):

        if labs[index] == -1:
            labs[index] = labs[int(nearest_neighbor[index])]
    return labs

# 決定グラフの作成
def draw_decision(rho,deltas):       
    plt.cla()
    for i in range(np.shape(rho)[0]):
        plt.scatter(rho[i],deltas[i],s=16.,color=(0,0,0))
        # plt.annotate(str(i), xy = (rho[i], deltas[i]),xytext = (rho[i], deltas[i]))
        plt.xlabel("rho")
        plt.ylabel("deltas")
    # plt.savefig(name)
    plt.show()

    
# クラスタ結果の作成    
def draw_cluster(datas, labs, centers, name):  
    
    dic_colors = {0:(.8,0,0),1:(0,.8,0),2:(0, 0, .8),3:(.8,.8,0),4:(.8,0,.8),5:(0,.8,.8),6:(0.5, 0.6, 0.2), 
              7:(.4, 0, 0), 8:(0, .4, 0), 9:(0, 0, .4), 10:(.4, .4, 0), 11:(.4, 0, .4), 12:(0, .4, .4), 
              13:(1,0,0),14:(0,1,0), 15:(0,0,1),16:(0,0,0), 17:(1,0,1), 18:(0,1,1), 
              19:(1,1,0.1), 20:(.2,0,0), 21:(0,.2,0), 22:(0,0,.2), 23:(.2,.2,0), 24:(.2,0,.2), 
              25:(0,.2,.2),26:(.6,0,0), 27:(0,.6,0),28:(0,0,.6), 29:(.6,.6,0),30:(.6,0,.6), 31:(0,.6,.6)}

    plt.cla()
    K = len(np.unique(labs))
    df_new = pd.DataFrame(datas)
    df_new[2] = labs
    # sns.scatterplot(data=df_new, x=0, y=1, hue=2, palette='Paired_r', legend=False)
    
    
    for k in range(K):
        sub_index = np.where(labs == k)
        sub_datas = datas[sub_index]
        plt.scatter(sub_datas[:,0], sub_datas[:,1], s=16., color=dic_colors[k%32])
        # plt.scatter(sub_datas[:,0],sub_datas[:,1],s=16.)
    
    for k in range(len(centers)):   
        plt.scatter(datas[centers[k],0], datas[centers[k],1], color="k", marker="+", s = 200.)
        
    plt.show()
    
    
    
'''
ただのDPC
'''
def density_peak_cluster(datas, percent, center_num, ans, name, pic=True, method='Gausiian'):
    
    dists = getDistanceMatrix(datas)
    dc = select_dc(dists, percent)
    print("カットオフ距離", dc)
    rho = get_density(dists,dc, method)# we can use other distance such as 'manhattan_distance'
    deltas, nearest_neighbor= get_deltas(dists,rho)
    
    if pic==True:
        draw_decision(rho,deltas)
    centers = find_centers_K(rho,deltas, center_num)
    print("cluster-centers",centers)
    labs = cluster_PD(rho,centers,nearest_neighbor)
    
    if pic==True:
        draw_cluster(datas,labs,centers, name)
    print('正解率：', preprocessing.accuracy(ans, labs))
    print('ARI：',  adjusted_rand_score(ans, labs))
    print('NMI：', normalized_mutual_info_score(ans, labs))
    
    return labs, centers
    
def get_decision(datas, percent, method='Gausiian'):
    dists = getDistanceMatrix(datas)
    dc = select_dc(dists, percent)
    print("カットオフ距離", dc)
    rho = get_density(dists, dc, method)# we can use other distance such as 'manhattan_distance'
    deltas, nearest_neighbor= get_deltas(dists, rho)
    draw_decision(rho, deltas)


def DPC_time(datas, ans, percent, center_num, method='Gausiian'):
    
    start = time.time()
    dists = getDistanceMatrix(datas)
    dc = select_dc(dists, percent)
    rho = get_density(dists,dc, method)# we can use other distance such as 'manhattan_distance'
    deltas, nearest_neighbor= get_deltas(dists,rho)
    
    centers = find_centers_K(rho,deltas, center_num)
    labs = cluster_PD(rho,centers,nearest_neighbor)
    
    fin = time.time()
    print('実行時間：{:.5f}'.format(fin-start))

    
    
'''
顔データセット用
'''

def density_peaks_cluster_faces(dists, percent, center_num, ans, name, method='Gaussian'):
    
    dc = select_dc(dists, percent)
    print("カットオフ距離", dc)
    rho = get_density(dists, dc, method)# we can use other distance such as 'manhattan_distance'
    deltas, nearest_neighbor= get_deltas(dists,rho)
    draw_decision(rho,deltas)
    centers = find_centers_K(rho,deltas, center_num)
    print("cluster-centers",centers)
    labs = cluster_PD(rho,centers,nearest_neighbor)
    # draw_cluster(datas,labs,centers, dic_colors, name)
    print('正解率：', preprocessing.accuracy(ans, labs))
    print('ARI：',  adjusted_rand_score(ans, labs))
    print('NMI：', normalized_mutual_info_score(ans, labs))    
    return labs