# from pycocotools.coco import COCO
# import numpy as np
# import skimage.io as io
# import matplotlib.pyplot as plt
#
# annFile = 'instances_val2017.json'
# coco=COCO(annFile)
#
# catIds = coco.getCatIds(catNms=['dog', 'person', 'bicycle'])
# print(catIds)
#
# imgIds = coco.getImgIds(catIds=catIds)
# print(imgIds)
#
# img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]
# print(img)
#
# I = io.imread(img['coco_url'])
# # plt.axis('off')
# # plt.imshow(I)
# # plt.show()
#
# plt.imshow(I)
# plt.axis('off')
#
# annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
# print(annIds)
#
# anns = coco.loadAnns(annIds)
# coco.showAnns(anns)
#
# plt.show()

'''
from yolox.data import (
            DOTAOBBDetection,
            TrainTransformOBB,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetectionOBB,
        )
from torch.utils.data import DataLoader
import numpy as np
import cv2
import torch

dataset = DOTAOBBDetection(
            #data_dir=os.path.join(get_yolox_datadir(), "VOCdevkit"), #delete
            data_dir = '/home/yangyang/yangyang/DATA/gxw/dataset/DOTA_OBB_1_5',
            #data_dir = '/home/yangyang/yangyang/DATA/gxw/dataset/DOTA_demo',
            #image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
            image_sets=[('2012', 'train')],
            img_size=(1024, 1024),
            preproc=TrainTransformOBB(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=200,
            )
        )

dataset = MosaicDetectionOBB(
    dataset,
    mosaic=True,
    img_size=(1024, 1024),
    preproc=TrainTransformOBB(
        rgb_means=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        max_labels=800,
    ),
    degrees=0.0,
    translate=0.1,
    scale=(0.8, 1.6),
    shear=2.0,
    perspective=0.0,
    enable_mixup=False,
)

train_dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=1)
print(len(train_dataloader))

# one_batch = next(iter(train_dataloader))
# img, target, img_info, img_id = one_batch[0]
max_angle = []
min_angle = []
for i, data in enumerate(train_dataloader):
    img, target, img_info, img_id = data
    img = img[0]
    img = img.numpy()
    img[0] *= 0.229
    img[1] *= 0.224
    img[2] *= 0.225
    img[0] += 0.485
    img[1] += 0.456
    img[2] += 0.406
    img *= 255.0
    img = np.uint8(img)
    #print(img.shape)
    img=img.transpose(1, 2, 0)
    img = np.ascontiguousarray(img, dtype=np.uint8) #
    #print(img.shape)
    target = target[0]
    if target[:, 5].max() > 90:
        break
    max_angle.append(target[:, 5].max())
    #print(target[:, 5].max())
    min_angle.append(target[:, 5].min())
    #print(target[:, 5].max())
    #print(target[0])
    #print(target[1])
    '''
'''
    for rect in target:
        label, x_c, y_c, w, h, angle = rect
        # print(type(label))
        # print('label is {}'.format(label))
        # print('hh')
        if -90 < angle <= 0:
            h, w = w, h
            angle = angle + 90.0
        if angle == -90.0:
            angle = angle + 180
        x_c = float(x_c)
        y_c = float(y_c)
        w = float(w)
        h = float(h)
        angle = float(angle)
        ret = [[x_c, y_c], [w, h], angle]
        #print(ret)
        #print(type(ret[2]))
        box = cv2.boxPoints(ret)
        box = np.int0(box)
        #print(box)
        [x1, y1], [x2, y2], [x3, y3], [x4, y4] = box
        #print(type(x1))
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 2, cv2.LINE_AA)
        cv2.line(img, (x2, y2), (x3, y3), (0, 255, 255), 2, cv2.LINE_AA)
        cv2.line(img, (x3, y3), (x4, y4), (0, 255, 255), 2, cv2.LINE_AA)
        cv2.line(img, (x4, y4), (x1, y1), (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('dst', img)
    cv2.waitKey(1000)
    #break
    '''
'''
    if i % 100 == 0:
        print(i)
    # if i > 1000:
    #     break

max_angle = torch.tensor(max_angle)
print('max angle is')
print(max_angle.max())

min_angle = torch.tensor(min_angle)
print('min angle is')
print(min_angle.min())
'''

import torch
def gaussian_label(labels, u=0, sig=4.0, raduius=2):
    '''
    转换成CSL Labels：
        用高斯窗口函数根据角度θ的周期性赋予gt labels同样的周期性，使得损失函数在计算边界处时可以做到“差值很大但loss很小”；
        并且使得其labels具有环形特征，能够反映各个θ之间的角度距离
    @param label: 当前box的θ类别  shape(1)
    @param num_class: θ类别数量=180
    @param u: 高斯函数中的μ
    @param sig: 高斯函数中的σ
    @return: 高斯离散数组:将高斯函数的最高值设置在θ所在的位置，例如label为45，则将高斯分布数列向右移动直至x轴为45时，取值为1 shape(180)
    '''
    # floor()返回数字的下舍整数   ceil() 函数返回数字的上入整数  range(-90,90)
    # 以num_class=180为例，生成从-90到89的数字整形list  shape(180)
    smooth_labels = []
    x = torch.tensor(range(-90, 90, 1))
    y_sig = torch.exp(-(x - u) ** 2 / (2 * sig ** 2))  # shape(180) 为-90到89的经高斯公式计算后的数值
    y_sig[0:90-raduius] = 0
    y_sig[91 + raduius:] = 0
    print(y_sig)
    for label in labels:
        smooth_labels.append(torch.cat([y_sig[90 - int(label.item()):], y_sig[:90 - int(label.item())]], dim=0).unsqueeze(0))
    # 将高斯函数的最高值设置在θ所在的位置，例如label为45，则将高斯分布数列向右移动直至x轴为45时，取值为1
    return torch.cat(smooth_labels, dim=0)


label = torch.zeros(1, dtype=int)
a = gaussian_label(label, u=0, sig=4.0)
print(a)
print(a.shape)
