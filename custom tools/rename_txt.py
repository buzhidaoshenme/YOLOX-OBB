import os

source_dir = '/home/lyy/gxw/result/txt'
target_dir = '/home/lyy/gxw/result/tijiao'

file_list = os.listdir(source_dir)

for filename in file_list:
    rename_filename = 'Task1_' + filename
    source_file_path = os.path.join(source_dir, filename)
    target_file_path = os.path.join(target_dir, rename_filename)
    os.rename(source_file_path, target_file_path)