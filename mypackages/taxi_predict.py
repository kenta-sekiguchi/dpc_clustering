import numpy as np
import pandas as pd
from typing import Tuple

# 時系列予測
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def hw_model(df_train, df_test, cluster_num, seasonal_periods):
    
    '''
    ホルツウィンターズモデルを用いて時系列予測を行う関数
    
    Args:
        df_train: 学習データ
        df_test: テストデータ
        cluster_num: クラスタ数
        seasonal_periods: 季節周期
        
    Returns:
        pred_list: 予測値
        mae: MAE
        rmse: RMSE
        r2: 決定係数
    '''
    
    pred_list = np.array([])
    
    for i in range(cluster_num):
        
        HW_model = ExponentialSmoothing(df_train[df_train['cluster_num']==i]['Given'],
                                        #trend = 'add',    #加法
                                        seasonal = 'additive', #加法
                                        seasonal_periods = seasonal_periods)
        
        HW_model_fit = HW_model.fit()
        pred = HW_model_fit.forecast(df_test[df_test['cluster_num']==i]['Given'].shape[0]) #予測
        pred_list = np.append(pred_list, pred.values)
    
    pred_list = np.where(pred_list<0, 0, pred_list)
    
    mae = mean_absolute_error(df_test['Given'].values, pred_list)
    rmse = np.sqrt(mean_squared_error(df_test['Given'].values, pred_list))
    
    # 決定係数を求める
    r2 = r2_score(df_test['Given'].values, pred_list)
        
    return pred_list, mae, rmse, r2



def prophet_model(df_train, df_test, cluster_num):
    
    '''
    プロフェットモデルを用いて時系列予測を行う関数
    
    Args:
        df_train: 学習データ
        df_test: テストデータ
        cluster_num: クラスタ数
        
    Returns:
        pred_list: 予測値
        mae: MAE
        rmse: RMSE
        r2: 決定係数
    '''
    
    pred_list = np.array([])
    
    for i in range(cluster_num):
        
        df_train_new = df_train[df_train['cluster_num']==i]
        df_test_new = df_test[df_test['cluster_num']==i]
        
        df_new = pd.concat([df_train_new, df_test_new])

        # 予測モデル構築
        df_prophet = df_new['Given'].reset_index()
        df_prophet.columns = ['ds', 'y']

        test_length = len(df_test_new)

        df_train_prophet = df_prophet.iloc[:-test_length]
        df_test_prophet = df_prophet.iloc[-test_length:]
        
        model = Prophet()
        model.fit(df_train_prophet)
        
        df_prophet_future = pd.DataFrame(np.array(df_prophet['ds']))
        df_prophet_future.columns = ['ds']
        
        df_pred = model.predict(df_prophet_future)
        
        test_pred = df_pred['yhat'][-test_length:].values

        pred_list = np.append(pred_list, test_pred)
    
    pred_list = np.where(pred_list<0, 0, pred_list)

    mae = mean_absolute_error(df_test['Given'].values, pred_list)
    rmse = np.sqrt(mean_squared_error(df_test['Given'].values, pred_list))
    # mape = mean_absolute_percentage_error(df_test['Given'].values, pred_list)
    # smape = 100/len(df_test) * np.sum(2 * np.abs(pred_list - df_test['Given'].values) / (np.abs(pred_list) + np.abs(df_test['Given'].values)))
    
    # 決定係数を求める
    r2 = r2_score(df_test['Given'].values, pred_list)
    
    return pred_list, mae, rmse, r2


'''
LSTMモデル
'''

# 上の一連の流れを実行するクラス

class LSTM(nn.Module):
    def __init__(self, hidden_size=100):
        super().__init__()
        self.hidden_size = hidden_size
        # input_sizeは入力する次元数
        self.lstm = nn.LSTM(input_size=1, hidden_size=self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x_last = x[-1]
        x = self.linear(x_last)

        return x
    
class LSTMModel:
    def __init__(self, train_X, train_Y, test_X, test_Y, hidden_size=100, epochs=100, batch_size=20, lr=0.001):
        self.model = LSTM(hidden_size=hidden_size)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.train_X = train_X
        self.train_Y = train_Y
        self.test_X = test_X
        self.test_Y = test_Y
        self.epochs = epochs
        self.batch_size = batch_size
        self.losses = []
        
    def mkRandomBatch(self, batch_size=10):
        
        '''
        train_X, train_Yを受け取ってbatch_X, batch_Yを返す。
        '''
        batch_X = []
        batch_Y = []

        for _ in range(batch_size):
            idx = np.random.randint(0, len(self.train_X) - 1)
            batch_X.append(self.train_X[idx])
            batch_Y.append(self.train_Y[idx])
        
        return np.array(batch_X), np.array(batch_Y)

    def fit(self):
        training_size = len(self.train_X)
        for epoch in range(self.epochs):
            for i in range(int(training_size / self.batch_size)):
                self.optimizer.zero_grad()

                data, label = self.mkRandomBatch(self.batch_size)
                data = torch.FloatTensor(data).permute(1, 0, 2)
                label = torch.FloatTensor(label).permute(1, 0, 2)

                output = self.model(data)

                loss = self.criterion(output, label)
                loss.backward()
                self.optimizer.step()

                self.losses.append(loss.item())

            # print((epoch + 1,'：', loss.item()))

    def predict(self):
        tensor_test_X = torch.FloatTensor(self.test_X).permute(1, 0, 2)
        predictions = self.model(tensor_test_X).detach().numpy()

        return predictions

    def evaluate(self):
        predictions = self.predict()
        test_Y_new = self.test_Y.reshape(len(self.test_Y), 1)
        rmse = mean_squared_error(test_Y_new, predictions, squared=False)
        mae = mean_absolute_error(test_Y_new, predictions)
        r2 = r2_score(test_Y_new, predictions)

        return predictions, rmse, mae, r2
    
def make_sequence_data(data: np.ndarray, sequence_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """データをsequence_sizeに指定したサイズのシーケンスに分けてシーケンスとその答えをarrayで返す
    Args:
        data (np.ndarray): 入力データ
        sequence_size (int): シーケンスサイズ
    Returns:
        seq_arr: sequence_sizeに指定した数のシーケンスを格納するarray
        target_arr: シーケンスに対応する答えを格納するarray
    """

    num_data = len(data)
    seq_data = []
    target_data = []
    for i in range(num_data - sequence_size):
        seq_data.append(data[i:i+sequence_size])
        target_data.append(data[i+sequence_size:i+sequence_size+1])
    seq_arr = np.array(seq_data)
    target_arr = np.array(target_data)

    return seq_arr, target_arr

def lstm_model(train_data, test_data, cluster_num, seq_length=24):
    
    '''
    lstmモデル
    
    Args:
        train_data: 学習データ
        test_data: テストデータ
        cluster_num: クラスタ数
        seq_length: シーケンス長（今は感覚を1時間にしているので24。30分にしたら48）
        
    Returns:
        pred_list: 予測値
        mae: MAE
        rmse: RMSE
        r2: 決定係数
    '''
        
    pred_list =np.array([])
    
    for i in range(cluster_num):
        
        print(i)
        
        train_data_lstm = train_data[train_data['cluster_num']==i][['Given']]
        test_data_lstm = test_data[test_data['cluster_num']==i][['Given']]

        train_X, train_Y = make_sequence_data(train_data_lstm, seq_length)
        test_X, test_Y = make_sequence_data(test_data_lstm, seq_length)
        
        lstm_model = LSTMModel(train_X, train_Y, test_X, test_Y, hidden_size=20, epochs=100, batch_size=20, lr=0.001)
        lstm_model.fit()
        predictions, rmse, mae, r2 = lstm_model.evaluate()
        
        pred_list = np.append(pred_list, predictions)
        
    return pred_list, rmse, mae, r2    