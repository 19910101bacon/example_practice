import os
import sys
symbol = sys.argv[1]
target_length = int(sys.argv[2])
target_theme = sys.argv[3]

for i in range(5):
    command = 'python3.7 training_model.py 0 {0} {1} {2}'.format(symbol, target_length, target_theme)
    print('-'*80)
    print('\n')
    print('-'*30 + command + '-'*30)
    print('\n')
    os.system(command)

for i in range(500):
    command = 'python3.7 training_model.py {0} {1} {2} {3}'.format(i, symbol, target_length, target_theme)
    print('-'*80)
    print('\n')
    print('-'*30 + command + '-'*30)
    print('\n')
    os.system(command)
