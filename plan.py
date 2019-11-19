import os
import sys

target_theme = sys.argv[1]
command = 'python3.7 training_model.py {0}'.format(target_theme)
print(command)
os.system(command)
