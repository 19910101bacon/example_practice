import os
import sys

target_theme = sys.argv[1]
symbol_chunk = sys.argv[2]
command = 'python3.7 training_model.py {0} {1}'.format(target_theme, symbol_chunk)
print(command)
os.system(command)
