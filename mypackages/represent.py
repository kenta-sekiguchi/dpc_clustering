import numpy as np
from sklearn.neighbors import NearestNeighbors



def get_representative_nartural(rho, r, dist_asc_index, real_center):
    
    '''
    中心候補点を求める関数。
    ルール1：複数の代表点を持つ場合は最も大きい点を中心点とする
    ルール2：代表点を遷移させる
    の2つを採用。
    
    Args:
        rho（numpy配列）:各点の密度が記載されたnumpy配列
        r（整数）:近傍数
        dist_asc_index（numpy配列）:r近傍点までのインデックスが記載されたnumpy配列
        real_center（整数）:真のクラスタ数
        
        
    Returns:
        rep_data（numpy配列）:各点の代表点が記載されたnumpy配列
        centers（numpy配列）:中心候補点のインデックス番号が記載されたnumpy配列
        new_r（整数）:近傍数（中心候補数が真のクラスタ数よりも小さかった場合に更新される）

    '''
    
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

            # for x_j in dist_asc_index[i]:

            #     if np.isnan(rep[x_j]):
            #         rep[x_j] = rep_i

            #     elif Mrho_i >= rho[int(rep[x_j])]:
            #         rep[x_j] = rep_i
                    
            # 上の処理をfor文を使わずに記述
            rep[dist_asc_index[i]] = np.where(np.isnan(rep[dist_asc_index[i]]), rep_i, rep[dist_asc_index[i]])
            rep[dist_asc_index[i]] = np.where(Mrho_i >= rho[rep[dist_asc_index[i]].astype(int)], rep_i, rep[dist_asc_index[i]])

        
        # 代表点遷移
        # 代表点
        rep_unique = np.unique(rep)
        
        # 代表点の代表点
        rep_of_rep = rep[rep_unique.astype(int)]

        # 遷移により代表点から削除される代表点
        non_rep = np.setdiff1d(rep_unique, np.unique(rep_of_rep))

        # 複数回の遷移が起こることを考慮している
        while True:

            for non_rep_i in non_rep:
                rep = np.where(rep==non_rep_i, rep_of_rep[np.where(rep_unique==non_rep_i)[0][0]], rep)
            
            # 上の処理をfor文を使わずに記述
            # rep = np.where(np.isin(rep, non_rep), rep_of_rep[np.where(np.isin(rep_unique, non_rep))[0]], rep)

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
        
        # if len(centers)>=real_center:
        #     new_r = r
        #     print('近傍数', new_r)
        #     break
            
        if len(centers)>real_center:
            print(len(centers))
            new_r = r
            print('近傍数', new_r)
            break
        
        '''----------------------------------------    '''
        
        print('rの値を変更しました', r)
        r = r-1
    
    return rep_data, centers, new_r
    


def get_representative_nartural_2(rho, r, dist_asc_index, real_center):
    
    '''
    中心候補点を求める関数。
    ルール1：複数の代表点を持つ場合は最も大きい点を中心点とする
    のみ採用。
    代表点を遷移させないことで、より多くの中心点を選択できるように。
    
    Args:
        rho（numpy配列）:各点の密度が記載されたnumpy配列
        r（整数）:近傍数
        dist_asc_index（numpy配列）:r近傍点までのインデックスが記載されたnumpy配列
        real_center（整数）:真のクラスタ数
        
        
    Returns:
        rep_data（numpy配列）:各点の代表点が記載されたnumpy配列
        centers（numpy配列）:中心候補点のインデックス番号が記載されたnumpy配列
        new_r（整数）:近傍数（中心候補数が真のクラスタ数よりも小さかった場合に更新される）

    '''
    
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

        rep_data = rep.copy()
        centers = np.unique(rep).astype(int)
        
        '''
        ------------------------------変更箇所--------------------------------------
        '''
        
        # if len(centers)>=real_center:
        #     new_r = r
        #     print('近傍数', new_r)
        #     break
            
        if len(centers)>real_center:
            print(len(centers))
            new_r = r
            print('近傍数', new_r)
            break
        
        '''----------------------------------------    '''
        
        print('rの値を変更しました', r)
        r = r-1
    
    return rep_data, centers, new_r

