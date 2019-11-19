from data_creater import *
import numpy as np
import pandas as pd
from re import search
import os
import sys
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout,Bidirectional
from keras.optimizers import RMSprop
from keras.models import model_from_json
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import time
import operator
import pickle
import warnings
from sklearn.svm import SVC
from sklearn import metrics
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=196)

class Predictor(object):
    def __init__(self,symbol:str,target_length:int,target_theme:str):
        self.__symbol = symbol
        file_loc = "./data/{0:}/all_normalized.csv".format(self.__symbol)
        if os.path.isfile(file_loc):
            data = pd.read_csv(file_loc).loc[3:,:].reset_index(drop=True)
            self.__closing_prices = data["close"].values
            self.__max_price = max(self.__closing_prices)
            self.__min_price = min(self.__closing_prices)
            self.__target_length = target_length
            self.__target_theme = target_theme
        else:
            print("File does not exist : ",file_loc)
    
    @property
    def symbol(self):
        return self.__symbol
    
    @staticmethod
    def download_prep(symbol:str,start_date:str,end_date:str):
        download = Downloader(symbol,start_date,end_date)
        download.save()
        file_path = "./data/{}/quotes.csv"
        if os.path.isfile(file_path.format(symbol)):
            feature = Feature_Selection.read_csv(symbol,file_path.format(symbol))
            feature.calculate_features()
            feature.normalize_data()
            feature.save_stock_data()
            feature.save_normalized_data()
        else:
            print("File does not exist : ",file_path.format(symbol))

    def select_model(self,verbose=0):
        __prop_path = "./model/{0}/{1}_{2}_train_props.json"
        with open(__prop_path.format(self.__symbol, self.__symbol+str(self.__target_length), self.__target_theme), 'r') as json_file:
                loaded_model_json = json_file.read()
                loaded_model_json = json.loads(loaded_model_json)
                colname = loaded_model_json['colname']
                
        loaded_model = ModelLoader(self.__symbol, self.__target_length, self.__target_theme)
#         seq_obj = MultiSequence(self.__symbol,loaded_model.window_size,self.__target_length,self.__target_theme,column=colname)
#         testing_result = loaded_model.model.evaluate(seq_obj.X,seq_obj.y, verbose=0)
#         print("> Now checking model: {0:<5}  Test loss result: {1:.4f} Test accuracy result: {2:.4f}".format(self.__symbol, testing_result[0], testing_result[1]))
        self.__best_model = loaded_model
#         self.__loss = testing_result[0]   
#         self.__accuracy = testing_result 
#         print("==> Model ticker {0:} with accuracy of {1:.4f}".format(self.__symbol, self.__accuracy))

#         tickers = []
#         tickers = tickers + [self.__symbol]
#         print(tickers)
        
#         best_model = None
#         lowest_test_error = 2.0
#         for idx,ticker in enumerate(tickers,1):
#             __prop_path = "./model/{0}/{1:}_train_props.json"
#             with open(__prop_path.format(ticker, ticker+str(self.__target_length)), 'r') as json_file:
#                 loaded_model_json = json_file.read()
#                 loaded_model_json = json.loads(loaded_model_json)
#                 colname = loaded_model_json['colname']
#             try:
#                 loaded_model = ModelLoader(ticker, self.__target_length)
#                 seq_obj = MultiSequence(self.symbol,loaded_model.window_size,self.__target_length,all_day=False,column=colname)
#                 testing_error = loaded_model.model.evaluate(seq_obj.X,seq_obj.y, verbose=0)
#                 if verbose==1:
#                     print(">{0:>3}) Now checking model: {1:<5}  Test error result: {2:.4f}".format(idx,ticker, testing_error))
#                 if lowest_test_error > testing_error:
#                     best_model = loaded_model
#                     lowest_test_error = testing_error
#             except:
#                 pass
#         self.__best_model = best_model
#         self.__test_error = lowest_test_error
#         if verbose in [1,2]:
#             print("==> Best model ticker {0:} with error of {1:.4f}".format(self.__best_model.ticker,self.__test_error))
    
    def db_return(self,show:bool):
        __prop_path = "./model/{0}/{1}_{2}_train_props.json"
        with open(__prop_path.format(self.symbol, self.symbol+str(self.__target_length), self.__target_theme), 'r') as json_file:
            loaded_model_json = json_file.read()
            loaded_model_json = json.loads(loaded_model_json)
            colname = loaded_model_json['colname']
            
        seq_obj = MultiSequence(self.symbol, self.__best_model.window_size,self.__target_length,self.__target_theme,column=colname)
        orig_prices = seq_obj.origin_close
        orig_date = seq_obj.all_date
        ans = [ '{0:.3f}%'.format(100*round(value, 4)) for value in seq_obj.rate]   
        orig_price_df = pd.DataFrame.from_dict({'date': orig_date, 'close_price': orig_prices, 'ans': ans})
        orig_price_df.index = range(len(orig_price_df))
#         predict_prob = self.__best_model.model.predict_proba(seq_obj.Xpred)
#         print(predict_prob)
        predict_prob = self.__best_model.model.predict(seq_obj.Xpred)
        pred_prob_df = pd.DataFrame(predict_prob, columns = [self.symbol+str(self.__target_length) + '_' + self.__target_theme])
        pred_prob_df.index = range(self.__target_length + self.__best_model.window_size - 1, len(pred_prob_df) + self.__target_length + self.__best_model.window_size - 1)  
        df = orig_price_df.merge(pred_prob_df, how = 'outer', left_index=True, right_index=True)
        if show is True:
            display(df)
        return df
     
    def graph_forcast(self):
        seq_obj = MultiSequence(self.symbol, self.__best_model.window_size,5)
        test_predict = self.__best_model.model.predict(seq_obj.X)
        
    def graph_tail(self, index=0):
        __prop_path = "./model/{0}/{1:}_train_props.json"
        with open(__prop_path.format(self.symbol, self.symbol+str(self.__target_length)), 'r') as json_file:
            loaded_model_json = json_file.read()
            loaded_model_json = json.loads(loaded_model_json)
            colname = loaded_model_json['colname']
            
        seq_obj = MultiSequence(self.symbol, self.__best_model.window_size,self.__target_length,all_day=False,column=colname)
        scaler = MinMaxScaler(feature_range=(self.__min_price ,self.__max_price))
        orig_data = seq_obj.original_data.reshape(-1,1)
        orig_prices = orig_data
#         orig_prices = scaler.fit_transform(orig_data).flatten()

        test_predict = self.__best_model.model.predict(seq_obj.Xpred)
        pred_prices = scaler.fit_transform(test_predict) 
        pred_close = pd.DataFrame(pred_prices, columns = ['y' + str(self.__target_length)])
        pred_close['close'] = orig_prices[self.__best_model.window_size-1:]
        pred_close['y' + str(self.__target_length)] =  pred_close['y' + str(self.__target_length)] + pred_close['close'] 
        pred_close = pred_close.set_index(seq_obj.date)     
        row_number = pred_close.shape[0]
        plot_table = pred_close.iloc[row_number+1-index-10:row_number+1-index]
        
        loss = 0
        for i in range(len(pred_prices) - self.__target_length):
            close_value = pred_close['close'].tolist()
            predclose_value = pred_close['y' + str(self.__target_length)].tolist()
            loss += np.abs(close_value[i+self.__target_length]- predclose_value[i])
        print(loss)
#         pred_close.iloc
        plt.figure(figsize=(12, 3))
        day10ago = plot_table.iloc[0, 0:1].tolist()

        plt.plot(range(-8 + self.__target_length - 1, -7 + self.__target_length - 1), day10ago, marker='o', color="#17becf")
        
        day9ago = plot_table.iloc[1, 0:1].tolist()
        plt.plot(range(-7 + self.__target_length - 1,-6 + self.__target_length - 1), day9ago, marker='o', color="#bcdb22")
        
        day8ago = plot_table.iloc[2, 0:1].tolist()
        plt.plot(range(-6 + self.__target_length - 1,-5 + self.__target_length - 1), day8ago, marker='o', color="#7f7f7f")
        
        day7ago = plot_table.iloc[3, 0:1].tolist()
        plt.plot(range(-5 + self.__target_length - 1,-4 + self.__target_length - 1), day7ago, marker='o', color="#e377c2")
        
        day6ago = plot_table.iloc[4, 0:1].tolist()
        plt.plot(range(-4 + self.__target_length - 1,-3 + self.__target_length - 1), day6ago, marker='o', color="#8c564b")
        
        day5ago = plot_table.iloc[5, 0:1].tolist()
        plt.plot(range(-3 + self.__target_length - 1,-2 + self.__target_length - 1), day5ago, marker='o', color="#9467bd")
        
        day4ago = plot_table.iloc[6, 0:1].tolist()
        plt.plot(range(-2 + self.__target_length - 1,-1 + self.__target_length - 1), day4ago, marker='o', color="#d62728")
        
        day3ago = plot_table.iloc[7, 0:1].tolist()
        plt.plot(range(-1 + self.__target_length - 1,0 + self.__target_length - 1), day3ago, marker='o', color="#2ca02c")
        
        day2ago = plot_table.iloc[8, 0:1].tolist()
        plt.plot(range(0 + self.__target_length - 1, 1 + self.__target_length - 1), day2ago, marker='o', color="#ff7f0e")
       
        day1ago = plot_table.iloc[9, 0:1].tolist()
        plt.plot(range(1 + self.__target_length - 1, 2 + self.__target_length - 1), day1ago, marker='o', color="#1f77b4")
        
        true = plot_table['close'].tolist()
        plt.plot(range(-9,1), true, marker='o', color="k")
        
def forcast_ploting(plot_table):
    plt.figure(figsize=(12, 3))
    colnum = plot_table.shape[1]
    day10ago = plot_table.iloc[0, 1:colnum].tolist()
    plt.plot(range(-8 , -8 + colnum - 1), day10ago, marker='o', color="#17becf")

    day9ago = plot_table.iloc[1, 1:colnum].tolist()
    plt.plot(range(-7 ,-7 + colnum - 1), day9ago, marker='o', color="#bcdb22")

    day8ago = plot_table.iloc[2, 1:colnum].tolist()
    plt.plot(range(-6 ,-6 + colnum - 1), day8ago, marker='o', color="#7f7f7f")

    day7ago = plot_table.iloc[3, 1:colnum].tolist()
    plt.plot(range(-5 ,-5 + colnum - 1), day7ago, marker='o', color="#e377c2")

    day6ago = plot_table.iloc[4, 1:colnum].tolist()
    plt.plot(range(-4 ,-4 + colnum - 1), day6ago, marker='o', color="#8c564b")

    day5ago = plot_table.iloc[5, 1:colnum].tolist()
    plt.plot(range(-3 ,-3 + colnum - 1), day5ago, marker='o', color="#9467bd")

    day4ago = plot_table.iloc[6, 1:colnum].tolist()
    plt.plot(range(-2 ,-2 + colnum - 1), day4ago, marker='o', color="#d62728")

    day3ago = plot_table.iloc[7, 1:colnum].tolist()
    plt.plot(range(-1 ,-1 + colnum - 1), day3ago, marker='o', color="#2ca02c")

    day2ago = plot_table.iloc[8, 1:colnum].tolist()
    plt.plot(range(0 , 0 + colnum - 1), day2ago, marker='o', color="#ff7f0e")

    day1ago = plot_table.iloc[9, 1:colnum].tolist()
    plt.plot(range(1 , 1 + colnum - 1), day1ago, marker='o', color="#1f77b4")
    
    true = plot_table['close'].tolist()
    plt.plot(range(-9,1), true, marker='o', color="k")

def final_model(X:np.array,y:np.array,learn_rate:float,dropout:float):
    model = Sequential()
    model.add(Bidirectional(LSTM(X.shape[1],return_sequences=False),input_shape=(X.shape[1:])))
    model.add(Dense(X.shape[1]))
    model.add(Dropout(dropout))
    model.add(Dense(y.shape[1],activation='tanh'))
    optimizer = RMSprop(lr=learn_rate)
    model.compile(loss='binary_crossentropy',optimizer=optimizer, metrics=['accuracy'])
    return model

def model_selector(ticker:str,window_sizes:list,target_length:int,target_theme:str,verbose=0,column = []):
    best_model = None
    best_training_F1 = 0
    best_training_precision = 0
    best_training_recall = 0
    best_test_F1 = 0
    best_test_precision = 0
    best_test_recall = 0              
    best_window_size = 0
    best_score = 0
    counter = 1
    if verbose==1:
        print("*** Best Model Selection for {} ***".format(ticker))
        print("=" * 60)

    for window_size in window_sizes:
        if verbose == 1:
            print("\nWindow size: {}".format(window_size))
            print('-' * 60)
        
        seq_obj = MultiSequence(ticker,window_size,target_length,target_theme, column)
        X_train,y_train,X_test,y_test = split_data(seq_obj)
        

        model = SVC(probability=True, gamma='auto', class_weight={1: 10})  
        y_train_true = list(chain(*y_train.tolist()))
#         print(len(X_train))
#         print(len(y_train_true))
#         print(y_train_true)
        y_test_true = list(chain(*y_test.tolist()))
        model.fit(X_train,y_train_true)
        y_train_pred = model.predict(X_train)
#         print(len(y_train_pred))
#         print(y_train_pred)
#         print('\n')
        y_test_pred = model.predict(X_test)
#         print(len(y_test_pred))
#         print(y_test_pred)
        df = pd.DataFrame({'y_test_true' : y_test_true, 'y_test_pred' : y_test_pred})
#         print(df)
#         print(metrics.precision_score(y_test_true, y_test_pred, average=None))
        
        F1_train = metrics.f1_score(y_train_true, y_train_pred, average='weighted')
        precision_train = metrics.precision_score(y_train_true, y_train_pred, average='micro')
        recall_train = metrics.recall_score(y_train_true, y_train_pred, average='micro')
        
        F1_test = metrics.f1_score(y_test, y_test_pred, average='weighted')
        precision_test = metrics.precision_score(y_test_true, y_test_pred, average='micro')
        recall_test = metrics.recall_score(y_test_true, y_test_pred, average='micro')
       
        score = (F1_train + F1_test)/2

        
        if verbose==1:
            msg = " > Window size: {0:} Score: {1:.4f}"
            msg+= "training_F1: {2:.4f} training_precision: {3:.4f}  training_recall: {4:.4f} test_F1: {5:.4f} testing_precision: {6:.4f}  testing_recall: {7:.4f}"
#             msg = " > Learn rate: {0:.4f} Dropout: {1:.2f}"
#             msg+= " Epoch: {2:} Training loss: {3:.4f} Training acc: {4:.4f} Testing loss: {5:.4f} Testing acc: {6:.4f} Score : {7:.4f}"
            msg = "{0:2}".format(str(counter))+"  "+msg.format(window_size,score,F1_train,precision_train,recall_train,F1_test,precision_test,recall_test)
            print(msg)

        if score > best_score :
            best_model = model
            best_window_size = window_size
            best_score = score
            best_training_F1 = F1_train
            best_training_precision = precision_train
            best_training_recall = recall_train
            best_test_F1 = F1_test
            best_test_precision = precision_test
            best_test_recall = recall_test                 
        counter+=1
        
    if verbose in [1,2]:
        print("\nFinal model selection summary for {0} with window size of {1}. Score = {2:.4f}:".format(ticker,best_window_size,best_score))
        print('-' * 60)
        msg = "training_F1: {0:.4f} training_precision: {1:.4f}  training_recall: {2:.4f} test_F1: {3:.4f} testing_precision: {4:.4f}  testing_recall: {5:.4f}"
        msg = msg.format(best_training_F1,best_training_precision,best_training_recall,best_test_F1,best_test_precision,best_test_recall)
#         msg = " Epoch: {2:} Training loss: {3:.4f} Training acc: {4:.4f} Testing loss: {5:.4f} Testing acc: {6:.4f} Score : {7:.4f}"
#         msg = msg.format(rate,dropout, epoch, lowest_training_loss, best_training_acc, lowest_test_loss, best_test_acc, best_score)
        print(msg)
    
    best_dict = {}
    best_dict["ticker"] = ticker
    best_dict['window_size'] = best_window_size
    best_dict['score'] = float("{0:.4f}".format(best_score))
    best_dict['training_F1'] = float("{0:.4f}".format(best_training_F1))
    best_dict['training_precision'] = float("{0:.4f}".format(best_training_precision))
    best_dict['training_recall'] = float("{0:.4f}".format(best_training_recall))
    best_dict['test_F1'] = float("{0:.4f}".format(best_test_F1))
    best_dict['test_precision'] = float("{0:.4f}".format(best_test_precision))
    best_dict['test_recall'] = float("{0:.4f}".format(best_test_recall))
    best_dict['target_length'] = target_length
    best_dict['target_theme'] = target_theme
    best_dict['colname'] = column

    return (best_model,best_dict)

class ModelLoader(object):
    __sub_folder = "./model/{0:}"
    __model_path = "./model/{0}/{1}_{2}_model.pickle"
#     __weights_path = "./model/{0}/{1}_{2}_weights.h5"
    __prop_path = "./model/{0}/{1}_{2}_train_props.json"
    
    __step_sub_folder = "./model_step/{0:}"
    __step_model_path = "./model_step/{0}/{1}_{2}_model.pickle"
#     __step_weights_path = "./model_step/{0}/{1}_{2}_weights.h5"
    __step_prop_path = "./model_step/{0}/{1}_{2}_train_props.json"

    def __init__(self,symbol: str,target_length:int, target_theme:str):
#         try:         
            if not os.path.isfile(ModelLoader.__model_path.format(symbol, symbol+str(target_length), target_theme)):
                print("No model exist for {}".format(symbol))
                return
#             if not os.path.isfile(ModelLoader.__weights_path.format(symbol, symbol+str(target_length), target_theme)):
#                 print("No weigths file exist for {}".format(symbol))
#                 return
            if not os.path.isfile(ModelLoader.__prop_path.format(symbol, symbol+str(target_length), target_theme)):
                print("No training properties file exist for {}".format(symbol))
                return            
            with open(ModelLoader.__model_path.format(symbol, symbol+str(target_length), target_theme), 'rb') as pickle_file:
                loaded_model = pickle.load(pickle_file)
#                 print(ModelLoader.__model_path.format(symbol, symbol+str(target_length), target_theme))
#                 loaded_model_json = json_file.read()
#                 loaded_model = model_from_json(loaded_model_json)
#                 loaded_model.load_weights(ModelLoader.__weights_path.format(symbol, symbol+str(target_length), target_theme))
#                 optimizer = RMSprop(lr=loaded_model_json['learn_rate'])
#                 loaded_model.compile(loss='binary_crossentropy',optimizer='rmsprop')
#                 loaded_model.compile(loss='mean_squared_error', optimizer='rmsprop')
                self.__model = loaded_model
            with open(ModelLoader.__prop_path.format(symbol, symbol+str(target_length), target_theme), 'r') as prop_file:
                self.__train_prop = json.load(prop_file)
#         except OSError as err:
#             print("OS error for symbol {}: {}".format(symbol, err))
#         except:
#             print("Unexpected error for symbol {}:{}".format(symbol, sys.exc_info()[0]))        


    @staticmethod
    def root_path():
        return "./model"

    @property
    def model(self):
        return self.__model

    @property
    def ticker(self):
        return self.__train_prop["ticker"]

    @property
    def window_size(self):
        return self.__train_prop["window_size"]

    @property
    def train_prop(self):
        return self.__train_prop
    
    @staticmethod
    def save(symbol: str,model,train_props: dict,force_overwrite:bool):
#         try:   
            if not os.path.isdir(ModelLoader.__sub_folder.format(symbol)):
                os.makedirs(ModelLoader.__sub_folder.format(symbol))
            if not os.path.isdir(ModelLoader.__step_sub_folder.format(symbol)):
                os.makedirs(ModelLoader.__step_sub_folder.format(symbol))
#             model_json = model.to_json()

#             timestr = time.strftime("%Y%m%d_%H%M%S", time.localtime())
#             with open(ModelLoader.__step_model_path.format(symbol, symbol + str(train_props['target_length']) + timestr, train_props['target_theme']), "wb") as pickle_file:
#                 pickle.dump(model, pickle_file)
#                 json_file.write(model_json)
#             model.save_weights(ModelLoader.__step_weights_path.format(symbol, symbol + str(train_props['target_length'])+ timestr, train_props['target_theme']))
#             with open(ModelLoader.__step_prop_path.format(symbol, symbol + str(train_props['target_length']) + timestr, train_props['target_theme']), 'w') as prop_file:
#                 json.dump(train_props, prop_file)
            
            if force_overwrite is not True:
                if os.path.isfile(ModelLoader.__model_path.format(symbol, symbol+str(train_props['target_length']), train_props['target_theme'])):
                    with open(ModelLoader.__prop_path.format(symbol, symbol+str(train_props['target_length']), train_props['target_theme']), 'r') as json_file:
                        loaded_model_json = json_file.read()
                        loaded_model_json = json.loads(loaded_model_json)
                        past_score = loaded_model_json['score']
                    if past_score >= train_props['score']:
                        print('Score is worse than the past')
                        return None
#                     if past_testing_score <= train_props['score']:
#                         print('Score is lower than the past')
#                         return 'Accuracy is worse than the past'
                    else :
                        print('Score is better than the past')

                    with open(ModelLoader.__model_path.format(symbol, symbol + str(train_props['target_length']), train_props['target_theme']), "wb") as pickle_file:
                        pickle.dump(model, pickle_file)
#                     model.save_weights(ModelLoader.__weights_path.format(symbol, symbol + str(train_props['target_length']), train_props['target_theme']))
                    with open(ModelLoader.__prop_path.format(symbol, symbol + str(train_props['target_length']), train_props['target_theme']), 'w') as prop_file:
                        json.dump(train_props, prop_file)
                        
                else :
                    with open(ModelLoader.__model_path.format(symbol, symbol + str(train_props['target_length']), train_props['target_theme']), "wb") as pickle_file:
                        pickle.dump(model, pickle_file)
#                     model.save_weights(ModelLoader.__weights_path.format(symbol, symbol + str(train_props['target_length']), train_props['target_theme']))
                    with open(ModelLoader.__prop_path.format(symbol, symbol + str(train_props['target_length']), train_props['target_theme']), 'w') as prop_file:
                        json.dump(train_props, prop_file)
                    
                    
#                 if not os.path.isfile(ModelLoader.__model_path.format(symbol, symbol+str(train_props['target_length']), train_props['target_theme'])) 
#                     with open(ModelLoader.__model_path.format(symbol, symbol + str(train_props['target_length']), train_props['target_theme']), "w") as json_file:
#                         json_file.write(model_json)
#                     model.save_weights(ModelLoader.__weights_path.format(symbol, symbol + str(train_props['target_length']), train_props['target_theme']))
#                     with open(ModelLoader.__prop_path.format(symbol, symbol + str(train_props['target_length']), train_props['target_theme']), 'w') as prop_file:
#                         json.dump(train_props, prop_file)
                

                
                

#         except OSError as err:
#             print("OS error for symbol {}: {}".format(symbol, err))
#         except:
#             print("Unexpected error for symbol {}:{}".format(symbol, sys.exc_info()[0]))