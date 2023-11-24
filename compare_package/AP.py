import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.cluster import AffinityPropagation
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import confusion_matrix
import time

from mypackages import preprocessing


def affinity_propagation(datas, ans, preference, randomstate=123, pic=True):
    
    ap = AffinityPropagation(random_state=randomstate, preference=preference).fit_predict(datas)


    cluster_accuracy = preprocessing.accuracy(ans, ap)
    ARI = adjusted_rand_score(ans, ap)
    NMI = normalized_mutual_info_score(ans, ap)
    # print('MO', MO)
    # print('Md', Md)
    # print('MO_d', MO_d)
    print('正解率：', cluster_accuracy)
    print('ARI：',  ARI)
    print('NMI：', NMI)

    if pic==True:
        draw_cluster2(datas, ap)
        
    return ap

def affinity_propagation_act(datas, ans):
    
    ap = AffinityPropagation().fit_predict(datas)


    cluster_accuracy = preprocessing.accuracy(ans, ap)
    ARI = adjusted_rand_score(ans, ap)
    NMI = normalized_mutual_info_score(ans, ap)

    print('正解率：', cluster_accuracy)
    print('ARI：',  ARI)
    print('NMI：', NMI)
        
    return ap

def affinity_time(datas, preference):
    
    start = time.time()
    ap = AffinityPropagation().fit_predict(datas)
    end = time.time()
    # print('実行時間：{:.5f}'.format(end-start))
    
    return (end-start)
    

def affinity_propagation_for(datas, ans, preference_list, randomstate=123):
    
    cluster_accuracy_default = -np.inf
    ARI_default = -np.inf
    NMI_default = -np.inf
    cluster_num = 0
    preference_def = 0
    
    for preference in preference_list:
            
        ap = AffinityPropagation(preference=preference, random_state=randomstate).fit_predict(datas)
        labs = ap.copy()

        cluster_accuracy = preprocessing.accuracy(ans, labs)
        ARI = adjusted_rand_score(ans, labs)
        NMI = normalized_mutual_info_score(ans, labs)


        if cluster_accuracy_default < cluster_accuracy:
            cluster_accuracy_default = cluster_accuracy
            ARI_default = ARI
            NMI_default = NMI 
            labs_def = labs


            cluster_num = len(np.unique(labs))
            preference_default = preference
    
        
        
    print('preference', preference_default)
    print('accuracy', cluster_accuracy_default)


def draw_cluster2(datas, labs):
    dic_colors = {0:(.8,0,0),1:(0,.8,0),2:(0, 0, .8),3:(.8,.8,0),4:(.8,0,.8),5:(0,.8,.8),6:(.5, 1, 0.2), 
                  7:(.4, 0, 0), 8:(0, .4, 0), 9:(0, 0, .4), 10:(.4, .4, 0), 11:(.4, 0, .4), 12:(0, .4, .4), 
                  13:(1,0,0),14:(0,1,0), 15:(0,0,1),16:(0,0,0), 17:(1,0,1), 18:(0,1,1), 
                  19:(1,1,0.1), 20:(.2,0,0), 21:(0,.2,0), 22:(0,0,.2), 23:(.2,.2,0), 24:(.2,0,.2), 
                  25:(0,.2,.2),26:(.6,0,0), 27:(0,.6,0),28:(0,0,.6), 29:(.6,.6,0),30:(.6,0,.6), 31:(0,.6,.6)}
    
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