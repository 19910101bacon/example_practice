from data_creater import *
from model import *
import re, copy
import gc 
import sys

theme = sys.argv[1]
symbol_chunk = int(sys.argv[2]) 

window_sizes = [7]

stocks = companies()
symbols = stocks['Symbol'].values.tolist()

def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

all_chunk_symbols = chunkIt(symbols, 10)
nowchunk_symbols = all_chunk_symbols[symbol_chunk]


for symbol in nowchunk_symbols:
    print('\n')
    print('---------- {0} ---------'.format(symbol))
    try:
        other_symbols = [symbol_tem for symbol_tem in symbols if symbol_tem != symbol]
        df = pd.read_csv('./data/{0}/all_normalized.csv'.format(symbol), index_col=[0], parse_dates=[0])
        symbol_columns = [symbol_tem for symbol_tem in list(df.columns.values) if bool(re.match('normal.*', symbol_tem))]

        box = []
        for symbol_tem in other_symbols:
            box_tem = []
            for col in list(df.columns.values):
                if bool(re.match(symbol_tem + '_normal_.*', col)):
                    box_tem.append(col)
            box.append(box_tem)

        all_combination = []
        all_combination.append(symbol_columns)
        for box_tem in box:
            all_combination.append(symbol_columns + box_tem)

        for i in range(len(all_combination)):
            result = model_selector(symbol, window_sizes, target_length=1,target_theme=theme,verbose=1,column=all_combination[i])
            ModelLoader.save(result[1]['ticker'],result[0],result[1],force_overwrite=False)
    except Exception as e: 
        print(e)
        with open('./rerun_symbol_model.txt', 'a') as f:
            f.write(symbol)
        pass





