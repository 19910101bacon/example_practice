from data_creater import *
from model import *
import re, copy
import gc 
import sys
stocks = companies()

index = int(sys.argv[1])

symbol = sys.argv[2]
#symbols = stocks[stocks.Industry == 'Technology']['Symbol'].values.tolist()
symbols = stocks['Symbol'].values.tolist()

if os.path.isfile('./data/{0}/important_symbol.json'.format(symbol)):
    with open('./data/{0}/important_symbol.json'.format(symbol), 'r') as json_file:
        loaded_model_json = json_file.read()
        loaded_model_json = json.loads(loaded_model_json)
        symbols = list(loaded_model_json['score'].keys())

other_symbols = [symbol_tem for symbol_tem in symbols if symbol_tem != symbol]

target_length = int(sys.argv[3])
target_theme = sys.argv[4]

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
    window_sizes = [2,3,4,5] 
    dropouts =  [0.1]
    learn_rates = [0.00005,0.0001]
    epochs = [10,30,50,100]
    batch_size = 50
else:
    window_sizes = [2,3,4,5] 
    dropouts =  [0.1]
    learn_rates = [0.0005,0.0001]
    epochs = [10,30,50,100]
    batch_size = 50

print(all_combination[index])
result = model_selector(symbol, window_sizes, learn_rates, dropouts, epochs, batch_size,target_length=target_length,target_theme=target_theme,verbose=1,column=all_combination[index])

print("\nResults : ")
print("-"*60)
print(result[0])

print(result[1])
ModelLoader.save(result[1]['ticker'],result[0],result[1],force_overwrite=False)
gc.collect()
