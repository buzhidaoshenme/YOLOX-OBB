import os
import shutil
from tqdm import tqdm

source_path = '/home/yangyang/yangyang/DATA/gxw/dataset/DOTA_split/train/hbbxml'
target_path = '/home/yangyang/yangyang/DATA/gxw/dataset/DOTA_HBB_1/VOC2012/Annotations'

filelist = os.listdir(source_path)
for filename in filelist:
    source_file_path = os.path.join(source_path, filename)
    target_file_path = os.path.join(target_path, filename)
    shutil.copy(source_file_path, target_file_path)
    #break