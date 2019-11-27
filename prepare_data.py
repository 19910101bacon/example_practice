from data_creater import *
import re, time
import sys

symbol_chunk = int(sys.argv[1])  ## 如果小於 0，則不跑 all_normalized

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

all_chunk_symbols = chunkIt(symbols, 5)
nowchunk_symbols = all_chunk_symbols[symbol_chunk]


if symbol_chunk >= 0:
    for symbol in nowchunk_symbols:
        print(symbol)
        target_dfs = pd.read_csv('./data/{0}/normalized.csv'.format(symbol),  index_col=[0], parse_dates=[0])
        dfs = []
        for symbol_tem in symbols:
            if symbol_tem == symbol or symbol_tem == 'BTC-USD':
                pass
            df = pd.read_csv('./data/{0}/normalized.csv'.format(symbol_tem), index_col=[0], parse_dates=[0]) 
            df = df.set_index(df['date'])
            specific_col = [col for col in df.columns if re.match('normal.*', col)]
            specific_col = [col for col in specific_col if col != 'normal_weekday']
            df_part = df[specific_col]
            df_part.columns = [symbol_tem + '_' + col for col in df_part.columns]   # if not bool(re.match('.*delta.*', col)) 
            dfs.append(df_part)

        finaldf_ = pd.concat(dfs, axis=1, join='outer')
        finaldf_['date'] = finaldf_.index
    #     finaldf_ = finaldf_.reset_index(['date'])
        finaldf = target_dfs.merge(finaldf_, on='date', how='left')
        finaldf.to_csv('./data/{0}/all_normalized.csv'.format(symbol), index = False)