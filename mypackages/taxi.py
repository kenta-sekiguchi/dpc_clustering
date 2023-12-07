import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import datetime

# 時系列予測
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


def convert_to_unix(s):
    '''
    文字列をUNIX時間に変換する関数（単位は分）
    '''
    
    # UNIX時間に変換する（単位は秒）
    return time.mktime(datetime.datetime.strptime(s, '%Y-%m-%dT%H:%M:%S.%f').timetuple())


def return_with_trip_times(df_taxi):
    '''
    新たなデータフレームを返す関数
    '''
    
    duration = df_taxi[['lpep_pickup_datetime','lpep_dropoff_datetime']]
    #pickups and dropoffs to unix time
    duration_pickup = [convert_to_unix(x) for x in duration['lpep_pickup_datetime'].values]
    duration_drop = [convert_to_unix(x) for x in duration['lpep_dropoff_datetime'].values]
    #calculate duration of trips
    durations = (np.array(duration_drop) - np.array(duration_pickup))/float(60)

    #append durations of trips and speed in miles/hr to a new dataframe
    new_frame = df_taxi.copy()
    new_frame = new_frame[['passenger_count','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']]
    
    new_frame['lpep_pickup_datetime'] = df_taxi['lpep_pickup_datetime'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%f'))
    new_frame['lpep_dropoff_datetime'] = df_taxi['lpep_dropoff_datetime'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%f'))
    
    new_frame['pickup_times_unix'] = duration_pickup

    
    return new_frame

def return_with_trip_times_2(df_taxi):
    '''
    新たなデータフレームを返す関数
    ダウンロードしたデータver
    '''
    
    df_taxi['lpep_pickup_datetime'] = pd.to_datetime(df_taxi['lpep_pickup_datetime'])
    df_taxi['lpep_dropoff_datetime'] = pd.to_datetime(df_taxi['lpep_dropoff_datetime'])

    new_frame = df_taxi.copy()
    new_frame = new_frame[['passenger_count','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude', 
                           'lpep_pickup_datetime', 'lpep_dropoff_datetime']]
        
    new_frame['pickup_times_unix'] = new_frame['lpep_pickup_datetime'].apply(lambda x: x.timestamp())

    return new_frame

def return_unq_pickup_bins(frame, cluster_num):
    
    '''
    各地点、各時刻でピックアップがあった時間を返す関数
    
    frame：データフレーム
    cluster_num：クラスタ数
    '''
    values = []
    for i in range(0, cluster_num):
        new = frame[frame['pickup_cluster'] == i]
        list_unq = list(set(new['pickup_bins']))
        list_unq.sort()
        values.append(list_unq)
    return values

def fill_missing(count_values, values, days, cluster_num, split):
    
    '''
    該当の地区、時間に乗車データがない場合に0を追加する関数
    '''
    
    smoothed_regions=[]
    ind=0
    for r in range(0, cluster_num):
        smoothed_bins=[]
        for i in range(24*split*days):
            if i in values[r]:
                smoothed_bins.append(count_values[ind])
                ind+=1
            else:
                smoothed_bins.append(0)
        smoothed_regions.extend(smoothed_bins)
        
    return smoothed_regions

def return_mesh(lat, lon, lat_min, lat_max, lon_min, lon_max, lat_split_num = 3, lon_split_num = 3):
    
    '''
    標準メッシュの地域を返す関数
    '''
    
    one_lat = (lat_max - lat_min)/lat_split_num
    one_lon = (lon_max - lon_min)/lon_split_num
    
    # 左上
    if (lat > lat_max - one_lat) & (lon < lon_min + one_lon):
        return 0
    
    # 真ん中上
    elif (lat > lat_max - one_lat) & (lon >= lon_min + one_lon) & (lon < lon_min + 2 * one_lon):
        return 1
    
    # 右上
    elif (lat > lat_max - one_lat) & (lon >= lon_min + 2 * one_lon):
        return 2
    
    # 左真ん中
    elif (lat <= lat_max - one_lat) & (lat > lat_max - 2 * one_lat) & (lon < lon_min + one_lon):
        return 3
    
    # 中央
    elif (lat <= lat_max - one_lat) & (lat > lat_max - 2 * one_lat) & (lon >= lon_min + one_lon) & (lon < lon_min + 2 * one_lon):
        return 4
    
    # 右真ん中
    elif (lat <= lat_max - one_lat) & (lat > lat_max - 2 * one_lat) & (lon >= lon_min + 2 * one_lon):
        return 5
    
    # 左下
    elif (lat <= lat_max - 2 * one_lat) & (lon < lon_min + one_lon):
        return 6
        
    # 中央下
    elif (lat <= lat_max - 2 * one_lat) & (lon >= lon_min + one_lon) & (lon < lon_min + 2 * one_lon):
        return 7
    
    # 右下
    elif (lat <= lat_max - 2 * one_lat) & (lon >= lon_min + 2 * one_lon):
        return 8