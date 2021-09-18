import os

# classnames = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
# 'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter', 'container-crane']
# For DOTA-v1.0
classnames = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship',
              'tennis-court',
              'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool',
              'helicopter']


target_dir = '/home/lyy/gxw/result/txt'



for classname in classnames:
    txt_name = classname + '.txt'
    txt_path = os.path.join(target_dir, txt_name)
    file = open(txt_path, 'w')
    file.close()