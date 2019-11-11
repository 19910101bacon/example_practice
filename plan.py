import os
import sys
symbol = 'AAPL'
target_length = 1

for i in range(5):
    command = 'python3.7 training_model.py 0 {0} {1}'.format(symbol, target_length)
    print('-'*80)
    print('\n')
    print('-'*30 + command + '-'*30)
    print('\n')
    os.system(command)

for i in range(15,31):
    command = 'python3.7 training_model.py {0} {1} {2}'.format(i, symbol, target_length)
    print('-'*80)
    print('\n')
    print('-'*30 + command + '-'*30)
    print('\n')
    os.system(command)
