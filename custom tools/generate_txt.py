# import os
#
# # path = '/home/yangyang/yangyang/DATA/gxw/dataset/DOTA_split/train'
# # label_file_name = 'labelTxt'
#
# path = '/home/yangyang/yangyang/DATA/gxw/dataset/DOTA_demo/VOC2012'
# label_file_name = 'Annotations'
#
# label_file_path = os.path.join(path, label_file_name)
# filelist = os.listdir(label_file_path)
#
# # txt_path = os.path.join(path, 'train.txt')
# txt_path = os.path.join(path, 'train.txt')
# f = open(txt_path, 'a')
#
# for filename in filelist:
#     txt = filename.split('.')[0]
#     f.write('{}\n'.format(txt))
#
# f.close()


import os



label_file_path = '/home/lyy/gxw/DOTA_OBB_1_5/VOC2012/JPEGImages-test'
filelist = os.listdir(label_file_path)

path = '/home/lyy/gxw'
txt_path = os.path.join(path, 'test.txt')
f = open(txt_path, 'a')

for filename in filelist:
    txt = filename.split('.')[0]
    f.write('{}\n'.format(txt))

f.close()