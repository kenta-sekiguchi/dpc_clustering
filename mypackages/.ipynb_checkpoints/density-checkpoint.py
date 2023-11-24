import numpy as np
from sklearn.neighbors import NearestNeighbors

'''
既存
'''

def get_density_knn(dist_asc, k):
    
    rho = np.exp(-(1/k)*(np.sum(dist_asc**2, axis=1)))
        
    return rho



'''
オリジナル
'''
def get_density_knn_2(dist_asc, k):
    
    '''
    密度の計算式
    
    Args:
        dist_asc（numpy配列）:r近傍点までの距離が格納された配列
        k（整数）:近傍の数（natural neighborで求められたもの）
        
    Returns:
        rho:密度

    '''
    
    rho = (1/k) * (np.sum(np.exp(-(dist_asc**2)), axis=1))
        
    return rho



'''
既存
'''
def get_density_knn_3(dist_asc, k):

    '''
    密度の計算式
    
    Args:
        dist_asc（numpy配列）:r近傍点までの距離が格納された配列
        k（整数）:近傍の数（natural neighborで求められたもの）
        
    Returns:
        rho:密度

    '''
    
    rho = k / (np.sum(dist_asc**2, axis=1))
            
    return rho

'''
Automatic Clustering via Outward Statistical Testing on Density Metricsの密度
'''
def get_density_knn_4(dist_asc, k):

    '''
    密度の計算式
    
    Args:
        dist_asc（numpy配列）:r近傍点までの距離が格納された配列
        k（整数）:近傍の数（natural neighborで求められたもの）
        
    Returns:
        rho:密度

    '''
    
    rho = k / (np.sum(dist_asc, axis=1))
            
    return rho

'''
相対密度
'''
def get_relative_density(rho, dist_asc_index):
    
    knn_ave_rho = np.mean(rho[dist_asc_index], axis=1)
    relative_rho = rho / knn_ave_rho
    
    return relative_rho