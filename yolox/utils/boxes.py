#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import numpy as np
import cv2
from DOTA_devkit_YOLO import polyiou

import torch
import torchvision


__all__ = [
    "filter_box",
    "postprocess",
    "bboxes_iou",
    "matrix_iou",
    "adjust_box_anns",
    "xyxy2xywh",
    "xyxy2cxcywh",
    'bboxes_iou_obb',#add
    'py_cpu_nms_poly',#add
    'postprocessobb',
    'postprocessobb_kld'
]


def filter_box(output, scale_range):
    """
    output: (N, 5+class) shape
    """
    min_scale, max_scale = scale_range
    w = output[:, 2] - output[:, 0]
    h = output[:, 3] - output[:, 1]
    keep = (w * h > min_scale * min_scale) & (w * h < max_scale * max_scale)
    return output[keep]


def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45): #[batch, n_anchors_all, 4 + 1 +  80]
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True) #[n_anchors_all, 1]

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        nms_out_index = torchvision.ops.batched_nms(
            detections[:, :4],
            detections[:, 4] * detections[:, 5],
            detections[:, 6],
            nms_thre,
        )
        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output


def postprocessobb(prediction, num_classes, conf_thre=0.7, nms_thre=0.45): # [batch, n_anchors_all, 4 + 1 + 180 + 80]
    # box_corner = prediction.new(prediction.shape)
    # box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    # box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    # box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    # box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    # prediction[:, :, :4] = box_corner[:, :, :4] # xywh to xyxy

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 185: 185 + num_classes], 1, keepdim=True) # [n_anchors_all, 1]
        _, angle_pred = torch.max(image_pred[:, 5: 185], 1, keepdim=True)
        angle_pred = angle_pred - 90 #(-90, 89)
        #print(angle_pred)


        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x, y, w, h, angle_pred, class_conf, class_pred, obj_conf)
        detections = torch.cat((image_pred[:, :4], angle_pred.float(), class_conf, class_pred.float(), image_pred[:, 4].unsqueeze(1)), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        detections = detections.numpy()
        boxes = np.zeros((0, 10))
        boxes_class = np.zeros((0, 9))

        #print(detections[0:5, :])

        for j in range(detections.shape[0]):
            if -90 < detections[j][4] <= 0:
                detections[j][2], detections[j][3] = detections[j][3], detections[j][2]
                detections[j][4] = detections[j][4] + 90.0
            if detections[j][4] == -90.0:
                detections[j][4] = 90.0
            rect = ((detections[j][0], detections[j][1]), (detections[j][3], detections[j][2]), detections[j][4])

            #rect = ((detections[j][0], detections[j][1]), (detections[j][2], detections[j][3]), detections[j][4])
            box = cv2.boxPoints(rect) # (x1,y1,x2,y2,x3,y3,x4,y4)
            box = np.int0(box)
            box = box.reshape(-1)
            box = np.append(box, detections[j][5]*detections[j][7])
            box = np.append(box, detections[j][6])
            # (x1,y1,x2,y2,x3,y3,x4,y4, class_conf*obj_conf, class_pred)
            box_class = np.copy(box[0:9])
            box_class[0:8] = box_class[0:8] + 4000 * detections[j][6]
            # (x1,y1,x2,y2,x3,y3,x4,y4, class_conf*obj_conf)
            box_class = box_class.reshape(1, 9)
            box = box.reshape(1, 10)
            boxes_class = np.append(box_class, boxes_class, axis=0)
            boxes = np.append(box, boxes, axis=0)

        nms_out_index = py_cpu_nms_poly(boxes_class, nms_thre)
        boxes = boxes[nms_out_index]
        boxes = torch.from_numpy(boxes) #(x1,y1,x2,y2,x3,y3,x4,y4, class_conf*obj_conf, class_pred)

        if output[i] is None:
            output[i] = boxes
        else:
            output[i] = torch.cat((output[i], boxes))
    return output


def postprocessobb_kld(prediction, num_classes, conf_thre=0.7, nms_thre=0.45): # [batch, n_anchors_all, 5 + 1 + 80]
    # box_corner = prediction.new(prediction.shape)
    # box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    # box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    # box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    # box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    # prediction[:, :, :4] = box_corner[:, :, :4] # xywh to xyxy

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):
        #(x,y,w,h,angle,obj_conf,n_class)
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 6: 6 + num_classes], 1, keepdim=True) # [n_anchors_all, 1]
        conf_mask = (image_pred[:, 5] * class_conf.squeeze() >= conf_thre).squeeze()
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float(), image_pred[:, 5].unsqueeze(1)), dim=1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        detections = detections.numpy()
        boxes = np.zeros((0, 10))
        boxes_class = np.zeros((0, 9))

        #print(detections[0:5, :])

        for j in range(detections.shape[0]):
            if -90 < detections[j][4] <= 0:
                detections[j][2], detections[j][3] = detections[j][3], detections[j][2]
                detections[j][4] = detections[j][4] + 90.0
            if detections[j][4] == -90.0:
                detections[j][4] = 90.0
            rect = ((detections[j][0], detections[j][1]), (detections[j][2], detections[j][3]), detections[j][4])

            # rect = ((detections[j][0], detections[j][1]), (detections[j][2], detections[j][3]), detections[j][4])
            box = cv2.boxPoints(rect)  # (x1,y1,x2,y2,x3,y3,x4,y4)
            box = np.int0(box)
            box = box.reshape(-1)
            box = np.append(box, detections[j][5] * detections[j][7])
            box = np.append(box, detections[j][6])
            # (x1,y1,x2,y2,x3,y3,x4,y4, class_conf*obj_conf, class_pred)
            box_class = np.copy(box[0:9])
            box_class[0:8] = box_class[0:8] + 4000 * detections[j][6]
            # (x1,y1,x2,y2,x3,y3,x4,y4, class_conf*obj_conf)
            box_class = box_class.reshape(1, 9)
            box = box.reshape(1, 10)
            boxes_class = np.append(box_class, boxes_class, axis=0)
            boxes = np.append(box, boxes, axis=0)

        nms_out_index = py_cpu_nms_poly(boxes_class, nms_thre)
        boxes = boxes[nms_out_index]
        boxes = torch.from_numpy(boxes)  # (x1,y1,x2,y2,x3,y3,x4,y4, score, class_pred)

        if output[i] is None:
            output[i] = boxes
        else:
            output[i] = torch.cat((output[i], boxes))
    return output


def py_cpu_nms_poly(dets, thresh):
    """
    任意四点poly nms.取出nms后的边框的索引
    @param dets: shape(detection_num, [poly, confidence1]) 原始图像中的检测出的目标数量
    @param thresh:
    @return:
            keep: 经nms后的目标边框的索引
    """
    scores = dets[:, 8]
    polys = []
    areas = []
    for i in range(len(dets)):
        tm_polygon = polyiou.VectorDouble([dets[i][0], dets[i][1],
                                            dets[i][2], dets[i][3],
                                            dets[i][4], dets[i][5],
                                            dets[i][6], dets[i][7]])
        polys.append(tm_polygon)

    # argsort将元素小到大排列 返回索引值 [::-1]即从后向前取元素
    order = scores.argsort()[::-1]  # 取出元素的索引值 顺序为从大到小
    keep = []
    while order.size > 0:
        ovr = []
        i = order[0]  # 取出当前剩余置信度最大的目标边框的索引
        keep.append(i)
        for j in range(order.size - 1):  # 求出置信度最大poly与其他所有poly的IoU
            iou = polyiou.iou_poly(polys[i], polys[order[j + 1]])
            ovr.append(iou)
        ovr = np.array(ovr)
        inds = np.where(ovr <= thresh)[0]  # 找出iou小于阈值的索引
        order = order[inds + 1]
    return keep



def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)


def matrix_iou(a, b):
    """
    return iou of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_i / (area_a[:, np.newaxis] + area_b - area_i + 1e-12)


def adjust_box_anns(bbox, scale_ratio, padw, padh, w_max, h_max):
    bbox[:, 0::2] = np.clip(bbox[:, 0::2] * scale_ratio + padw, 0, w_max)
    bbox[:, 1::2] = np.clip(bbox[:, 1::2] * scale_ratio + padh, 0, h_max)
    return bbox


def xyxy2xywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return bboxes


def xyxy2cxcywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5
    return bboxes


def iou_rotate_calculate(boxes1, boxes2):
    area1 = boxes1[2] * boxes1[3]
    area2 = boxes2[2] * boxes2[3]
    r1 = ((boxes1[0], boxes1[1]), (boxes1[2], boxes1[3]), boxes1[4])
    r2 = ((boxes2[0], boxes2[1]), (boxes2[2], boxes2[3]), boxes2[4])

    int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
    if int_pts is not None:
        order_pts = cv2.convexHull(int_pts, returnPoints=True)

        int_area = cv2.contourArea(order_pts)

        inter = int_area * 1.0 / (area1 + area2 - int_area)
        return inter
    else:
        return 0.0


def bboxes_iou_obb(gt_bboxes_per_image, bboxes_preds_per_image):
    ious = np.zeros((gt_bboxes_per_image.shape[0], bboxes_preds_per_image.shape[0]), dtype=np.float32)
    for i in range(gt_bboxes_per_image.shape[0]):
        for j in range(bboxes_preds_per_image.shape[0]):
            ious[i][j] = iou_rotate_calculate(gt_bboxes_per_image[i], bboxes_preds_per_image[j])
    return ious

def bboxes_iou_obb_cuda(gt_bboxes_per_image, bboxes_preds_per_image):
    ious = np.zeros((gt_bboxes_per_image.shape[0], bboxes_preds_per_image.shape[0]), dtype=np.float32)
    for i in range(gt_bboxes_per_image.shape[0]):
        for j in range(bboxes_preds_per_image.shape[0]):
            ious[i][j] = iou_rotate_calculate(gt_bboxes_per_image[i], bboxes_preds_per_image[j])
    return ious