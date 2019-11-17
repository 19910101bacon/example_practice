from abc import ABC, abstractmethod,ABCMeta
import sys
import os
import math
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from data_fetcher import downloader
from datetime import datetime
from collections import OrderedDict,Set
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
import operator
from datetime import datetime, date

def companies():
    dataset = pd.read_csv(os.path.join("data","dow30.csv"))
    return dataset

def symbol_list():
    dataset = pd.read_csv(os.path.join("data","dow30.csv"))
    return dataset['Symbol'].values.tolist()

class BaseData(object):
    def __init__(self,symbol:str):
        self.__symbol = symbol
    
    @property
    def symbol(self):
        return self.__symbol

    def save(self,file_dir:str,file_name:str,data:pd.DataFrame):
        try:
            if data is None:
                return
            full_path = os.path.join(file_dir,file_name)
            include_index = False if data.index.name == None else True
            if os.path.isdir(file_dir):
                data.to_csv(full_path,index=include_index)
            else:
                os.makedirs(file_dir)
                data.to_csv(full_path,index=include_index)
        except OSError as err:
            print("OS error for symbol {} : {}".format(self.symbol,err))
        except:
            print("Unexpected error for symbol {} : {}".format(self.symbol, sys.exc_info()[0]))


class Downloader(BaseData):
    def __init__(self,symbol:str,start_date:str, end_date:str):
        try:
            BaseData.__init__(self,symbol)
            self.__start_date = datetime.strptime(start_date,'%Y%m%d')
            self.__end_date = datetime.strptime(end_date,'%Y%m%d')
            self.__data = None

            #Download data from Yahoo.
            yah = downloader.load_yahoo_quote(symbol,start_date,end_date)
            header = yah[0].split(',')
            table = []
            for i in yah[1:]:
                quote = i.split(',')
                if len(quote)>1:
                    d = dict()
                    d[header[0]] = quote[0]
                    d[header[1]] = quote[1]
                    d[header[2]] = quote[2]
                    d[header[3]] = quote[3]
                    d[header[4]] = quote[4]
                    d[header[5]] = quote[5]
                    d[header[6]] = quote[6]
                    table.append(d)
            self.__data = pd.DataFrame(table)
            self.__size = len(self.__data)
        except OSError as err:
            print("OS error for symbol {} : {}".format(symbol,err))
    
    def save(self):
        file_dir = os.path.join("./data",self.symbol)
        BaseData.save(self,file_dir,"quotes.csv",self.__data)

    @property
    def start_date(self):
        return self.__start_date

    @property
    def end_date(self):
        return self.__end_date

    @property
    def data(self):
        return self.__data
    
    @property
    def size(self):
        return self.__size


class Feature_Selection(BaseData):
    def __init__(self,symbol:str,data:pd.DataFrame,mfi_days=14):
        BaseData.__init__(self,symbol)
        self.__days = mfi_days
        self.__data = None
        self.__data_normal = None

        cols = data.columns.values
        cols_check = "Date,Open,High,Low,Close,Adj Close,Volume".split(',')
        missing = False
        for col in cols:
            found = False
            for name in cols_check:
                if col == name:
                    found = True
                    break
            if not found:
                print("The column {} is missing.".format(col))
                missing = True
                break
        if not missing:
            self.__data = data
            self.__data['Date'] = pd.to_datetime(self.__data['Date'])
            self.__data.sort_values('Date',inplace=True)
            self.__data.reset_index(drop=True,inplace=True)
            self.__data.index.name = 'index'
    
    @classmethod
    def read_csv(cls,symbol:str,file_loc:str):
        try:
            data = pd.read_csv(file_loc)
            return cls(symbol,data)
        except OSError as err:
            print("OS error {}".format(err))
            return None

    @property
    def data(self):
        return self.__data
    
    @property
    def data_normal(self):
        return self.__data_normal

    def calculate_features(self):
        self.__cal_log_return("Adj Close")
        self.__cal_mfi()

    def __scale_data(self,col_Name:str):
        values = self.__data[col_Name].iloc[self.__days:].values.reshape(-1,1)
        scaler = MinMaxScaler(feature_range=(-1,1))
        return scaler.fit_transform(values).flatten()

    def __flatten_data(self,col_Name:str):
        return self.__data[col_Name].iloc[self.__days:].values.flatten()

    def normalize_data(self):
        scaler = MinMaxScaler(feature_range=(-1,1))
        index = self.__data.index.values[self.__days:]
        table = OrderedDict()
        table['date'] = self.__flatten_data('Date')
        table['open'] = self.__flatten_data('Open')
        table['high'] = self.__flatten_data('High')
        table['low'] = self.__flatten_data('Low')
        table['volume'] = self.__flatten_data('Volume')
        table['close'] = self.__flatten_data('Adj Close') 
        table['returns'] = self.__flatten_data('Adj Close_log_returns')
        table['mfi'] = self.__flatten_data('mfi_index')
        table['close_diff_day1'] = [0] + list(map(operator.sub, self.__flatten_data('Adj Close')[1:], self.__flatten_data('Adj Close')[:-1])) 
        table['close_diff_day2'] = [0,0] + list(map(operator.sub, self.__flatten_data('Adj Close')[2:], self.__flatten_data('Adj Close')[:-2]))   
        table['close_diff_day3'] = [0,0,0] + list(map(operator.sub, self.__flatten_data('Adj Close')[3:], self.__flatten_data('Adj Close')[:-3]))   
#         table['normal_open_'] = self.__scale_data('Open')
#         table['normal_high'] = self.__scale_data('High')
#         table['normal_low'] = self.__scale_data('Low')
        table['normal_amplitude'] = scaler.fit_transform(np.reshape(table['high'] - table['low'], (-1,1))).flatten()
        table['normal_volume'] = self.__scale_data('Volume')
        table['normal_close'] = self.__scale_data('Adj Close')
        table['normal_close_diff_day1'] = scaler.fit_transform(np.reshape(table['close_diff_day1'], (-1,1))).flatten()
        table['normal_close_diff_day2'] = scaler.fit_transform(np.reshape(table['close_diff_day2'], (-1,1))).flatten()
        table['normal_close_diff_day3'] = scaler.fit_transform(np.reshape(table['close_diff_day3'], (-1,1))).flatten()
        
        table['normal_returns'] = self.__scale_data('Adj Close_log_returns')
        table['normal_mfi'] = self.__scale_data('mfi_index')
        table['normal_weekday'] = [datetime.utcfromtimestamp(date_.astype(int) * 1e-9).weekday()  + 1 for date_ in table['date']]
            
        table['rate_day1'] = [1] + list(map(operator.truediv, table['close_diff_day1'][1:], table['close'][:-1]))
        table['rate_day2'] = [1,1] + list(map(operator.truediv, table['close_diff_day2'][2:], table['close'][:-2]))
        table['rate_day3'] = [1,1,1] + list(map(operator.truediv, table['close_diff_day3'][3:], table['close'][:-3]))
    
#         print(len(table['date']))
#         print(len(table['normal_close_diff_day3']))
        self.__data_normal = pd.DataFrame(table,index=index)
#         print(self.__data_normal)

        
        self.__data_normal['bummpy_day1_g000'] = np.where(self.__data_normal['rate_day1'] > 0, 1, 0)
        self.__data_normal['bummpy_day1_g0005'] = np.where(self.__data_normal['rate_day1'] > 0.005, 1, 0)
        self.__data_normal['bummpy_day1_g001'] = np.where(self.__data_normal['rate_day1'] > 0.01, 1, 0)
        self.__data_normal['bummpy_day1_g002'] = np.where(self.__data_normal['rate_day1'] > 0.02, 1, 0)
        self.__data_normal['bummpy_day1_g003'] = np.where(self.__data_normal['rate_day1'] > 0.03, 1, 0)
        self.__data_normal['bummpy_day1_s000'] = np.where(self.__data_normal['rate_day1'] < 0, 1, 0)
        self.__data_normal['bummpy_day1_s0005'] = np.where(self.__data_normal['rate_day1'] < -0.005, 1, 0)
        self.__data_normal['bummpy_day1_s001'] = np.where(self.__data_normal['rate_day1'] < -0.01, 1, 0)
        self.__data_normal['bummpy_day1_s002'] = np.where(self.__data_normal['rate_day1'] < -0.02, 1, 0)
        self.__data_normal['bummpy_day1_s003'] = np.where(self.__data_normal['rate_day1'] < -0.03, 1, 0)
        self.__data_normal['bummpy_day2_g000'] = np.where(self.__data_normal['rate_day2'] > 0, 1, 0)
        self.__data_normal['bummpy_day2_g0005'] = np.where(self.__data_normal['rate_day2'] > 0.005, 1, 0)
        self.__data_normal['bummpy_day2_g001'] = np.where(self.__data_normal['rate_day2'] > 0.01, 1, 0)
        self.__data_normal['bummpy_day2_g002'] = np.where(self.__data_normal['rate_day2'] > 0.02, 1, 0)
        self.__data_normal['bummpy_day2_g003'] = np.where(self.__data_normal['rate_day2'] > 0.03, 1, 0)
        self.__data_normal['bummpy_day2_s000'] = np.where(self.__data_normal['rate_day2'] < 0, 1, 0)
        self.__data_normal['bummpy_day2_s0005'] = np.where(self.__data_normal['rate_day2'] < -0.005, 1, 0)
        self.__data_normal['bummpy_day2_s001'] = np.where(self.__data_normal['rate_day2'] < -0.01, 1, 0)
        self.__data_normal['bummpy_day2_s002'] = np.where(self.__data_normal['rate_day2'] < -0.02, 1, 0)
        self.__data_normal['bummpy_day2_s003'] = np.where(self.__data_normal['rate_day2'] < -0.03, 1, 0)
        self.__data_normal.index.name = 'index'

    def __cal_log_return(self,col_name:str):
        values = self.__data[col_name].values
        log_returns = np.zeros_like(values)
        for idx in range(1,len(values)):
            log_returns[idx] = math.log(values[idx]/values[idx-1])
        self.__data[col_name+"_log_returns"] = pd.Series(log_returns, index = self.__data.index)

    def save_stock_data(self):
        file_dir = os.path.join("./data",self.symbol)
        BaseData.save(self,file_dir,"quote_processed.csv",self.__data_normal)

    def save_normalized_data(self):
        file_dir = os.path.join("./data",self.symbol)
        BaseData.save(self,file_dir,"normalized.csv",self.__data_normal)

    def __cal_mfi(self):
        # money flow index
        typ_price = pd.DataFrame((self.__data["High"] + self.__data["Low"] + self.__data["Adj Close"])/3, columns =["price"] )
        typ_price['volume'] = self.__data["Volume"]
        typ_price['pos'] = 0
        typ_price['neg'] = 0
        typ_price['mfi_index'] = 0.0
        for idx in range(1,len(typ_price)):
            if typ_price['price'].iloc[idx] > typ_price['price'].iloc[idx-1]:
                typ_price.at[idx,'pos' ] = typ_price['price'].iloc[idx] * typ_price['volume'].iloc[idx]   
            else:
                typ_price.at[idx,'neg'] = typ_price['price'].iloc[idx] * typ_price['volume'].iloc[idx]

        pointer = 1
        for idx in range(self.__days,len(typ_price)):
            pos = typ_price['pos'].iloc[pointer:idx + 1].sum()
            neg = typ_price['neg'].iloc[pointer:idx + 1].sum()
            
            if neg != 0:
                base = (1.0 + (pos/neg))
            else:
                base = 1.0
            typ_price.at[idx,'mfi_index'] = 100.0 - (100.0/base )
            pointer += 1

        self.__data["mfi_index"] = pd.Series(typ_price["mfi_index"].values, index = typ_price.index)
    

class Volatility(object):
    def __init__(self,symbol:str):
        try:
            path_norm_data = "./data/{}/normalized.csv".format(symbol)
            dataset = pd.read_csv(path_norm_data,index_col='index')
            self.__volatility = dataset['returns'].std() * math.sqrt(252)
        except:
            self.__volatility = -1

    @property
    def annual(self):
        return self.__volatility

class SequenceBase(ABC):
    def __init__(self,symbol:str,window_size:int,target_length:int,target_theme:str,column:list):
        try:
            self.__window_size = window_size
            self.__target_length = target_length
            self.__target_name = 'bummpy_day{0}_{1}'.format(str(target_length),target_theme)
            select_column = [self.__target_name, 'close', 'date'] + column
            path_norm_data = "./data/{}/all_normalized.csv".format(symbol)
            df = pd.read_csv(path_norm_data).loc[3:,:].reset_index(drop=True)
            self.__data_normal = df[[c for c in df.columns if c in select_column]]
        except:
            print("Unexpected error for symbol {} : {}".format(symbol,sys.exc_info()[0]))
    
    @property
    def data(self):
        return self.__data_normal.drop(columns=['close', 'date',self.__target_name])

    @property
    def origin_close(self):
         return self.__data_normal['close'].values
        
    @property
    def ans(self):
         return self.__data_normal[self.__target_name].values
    
    @property
    def all_date(self):
        return self.__data_normal['date'].values

    @property
    def window_size(self):
        return self.__window_size

    @property
    def target_length(self):
        return self.__target_length

    @property
    @abstractmethod
    def X(self):
        pass
     
    @property
    @abstractmethod
    def y(self):
        pass

class SimpleSequence(SequenceBase):
    def __init__(self,symbol:str,window_size:int,target_length:int,target_theme:str,column:list):
        SequenceBase.__init__(self,symbol,window_size,target_length,target_theme,column)
        self.__sequence_data()
    
    def __sequence_data(self):
        data = self.data
        true_ans = self.ans
        X=[]
        y=[]
        pointer = 0
        data_length = len(close)
        while (pointer+self.window_size+self.target_length)<=data_length:
            X.append(data[pointer:pointer+self.window_size])
            y.append(true_ans[pointer+self.window_size:pointer+self.window_size+self.target_length])
            pointer+=1
        self.__X = np.asarray(X)
        self.__X = self.__X.reshape((-1,self.__X.shape[-1],1))
        self.__y = np.asarray(y)

    @property
    def X(self):
        return self.__X
     
    @property
    def y(self):
        return self.__y

class MultiSequence(SequenceBase):
    def __init__(self,symbol:str,window_size:int,target_length:int,target_theme:str,column:list):
        SequenceBase.__init__(self,symbol, window_size, target_length,target_theme,column)
        self.__sequence_data()

    def __sequence_data(self):
        dates = self.all_date
        data = self.data
        true_ans = self.ans
        
        X = []
        Xpred = []
        y = []
        date_ = []
        pointer = 0
        data_length = len(true_ans)
        while (pointer + self.window_size) <= data_length:
            if (pointer + self.window_size + self.target_length) <= data_length:
                part_data = data.loc[pointer:pointer + self.window_size - 1,] 
#                 part_data_value = list(chain(*part_data.values.tolist())) #np.array([list(chain(*part_data.values.tolist()))]).tolist()
                part_data_value = part_data.values
                X.append(part_data_value)

                y.append(true_ans[pointer+self.window_size+self.target_length-1:pointer+self.window_size+self.target_length])
                
            if (pointer + self.window_size) <= data_length:
                part_data = data.loc[pointer:pointer + self.window_size - 1,] 
#                 part_data_value = list(chain(*part_data.values.tolist())) #np.array([list(chain(*part_data.values.tolist()))]).tolist()
                part_data_value = part_data.values
                Xpred.append(part_data_value)
                date_.append(dates[pointer + self.window_size - 1])

            pointer += 1

        self.__X = np.asarray(X)
        self.__Xpred = np.asarray(Xpred)
        self.__y = np.asarray(y)
        self.__date = np.asarray(date_)

    @property
    def X(self):
        return self.__X

    @property
    def Xpred(self):
        return self.__Xpred
     
    @property
    def y(self): 
        return self.__y
    
    @property
    def date(self): 
        return self.__date
    
    
def split_data(seq_obj:SequenceBase,split_rate=0.2):
    split = int(len(seq_obj.X) * (1-split_rate))
    X_train = seq_obj.X[:split,:]
    y_train = seq_obj.y[:split,:]

    X_test = seq_obj.X[split:,:]
    y_test = seq_obj.y[split:]
    return X_train,y_train,X_test,y_test

def graph_prediction(trained_model,X_train,X_test,original,window_size):
    train_predict = trained_model.predict(X_train)
    test_predict = trained_model.predict(X_test)
    plt.plot(original,color='k')
    split = len(X_train)
    split_pt = split + window_size
    train_in = np.arange(window_size,split_pt,1)
    plt.plot(train_in,train_predict,color='b')
    test_in = np.arange(split_pt,split_pt+len(test_predict),1)
    plt.plot(test_in,test_predict,color='r')

    plt.xlabel('day')
    plt.ylabel('(normalized) price of stock')
    plt.legend(['original series','training fit','testing fit'],loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()