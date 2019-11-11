from data_creater import *
from model import *
import re, copy
import gc 
import sys
stocks = companies()

index = int(sys.argv[1])

symbol = sys.argv[2]
symbols = stocks['Symbol'].values.tolist()
other_symbols = [symbol_tem for symbol_tem in symbols if symbol_tem != symbol]

target_length = int(sys.argv[3])

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

print('------ have {0} combinations ------'.format(str(len(all_combination))))



# other_columns = [symbol_tem for symbol_tem in list(df.columns.values) if not bool(re.match('normal.*', symbol_tem))]
# other_columns = [symbol_tem for symbol_tem in other_columns if bool(re.match('.*_.*', symbol_tem))]
# other_columns = [symbol_tem for symbol_tem in other_columns if bool(re.match('.*volume|.*close', symbol_tem))]

# all_combination = []
# for other_col in other_columns:
#     tem = copy.deepcopy(base_columns)
#     tem.append(other_col)
#     all_combination.append(tem)

# all_combination.append(base_columns)
if index == 0:
    window_sizes = [1,2,3,4] 
    dropouts =  [0.4]
    learn_rates = [0.0005,0.00001]
    epochs = [100,200,300,500]
    batch_size = 100
else:
    window_sizes = [1,2,3,4] 
    dropouts =  [0.4]
    learn_rates = [0.0005,0.00001]
    epochs = [50,100,250,400]
    batch_size = 100

print(all_combination[index])
result = model_selector(symbols[2], window_sizes, learn_rates, dropouts, epochs, batch_size,target_length=target_length,all_day=False,verbose=1,column=all_combination[index])

print("\nResults : ")
print("-"*60)
print(result[0])

print(result[1])
ModelLoader.save(result[1]['ticker'],result[0],result[1],force_overwrite=False)
gc.collect()