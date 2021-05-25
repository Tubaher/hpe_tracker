# from human_pose_estimation_demo_eddy import *
from deployable import *
from model import *

import logging
import sys
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from openvino.inference_engine import IECore

def draw_poses(img, poses, point_score_threshold, skeleton=default_skeleton, draw_ellipses=False, deploy=False  ):
    if poses.size == 0:
        return img, [], []
    stick_width = 4

    img_limbs = np.copy(img)
    #print("poses",poses.shape)
    bbox_list = []
    scores = []
    for pose in poses:
        points = pose[:, :2].astype(int).tolist()
        points_scores = pose[:, 2]
        points_scores_mean=np.mean(points_scores)
        # print(f'Shape of points_scores_mean : {points_scores_mean.shape}')
        
        # Draw joints.
        totalx=0
        totaly=0
        
        x=[]
        y=[]
        nocero=np.count_nonzero(np.array(points_scores))
        #print(nocero)
        for i, (p, v) in enumerate(zip(points, points_scores)):
            if v >0: # point_score_threshold:
                if deploy==False:
                    cv2.circle(img, tuple(p), 1, colors[i], 2)
                #totalx += p[0]
                #totaly += p[1]
                x.append(p[0])
                y.append(p[1])
                #print(p[0])
                #print(i)
                #print(len(p))
        #print(i)
       
        x_left = int(min(np.array(x)))
        x_right = int(max(np.array(x)))
        y_top = int(min(np.array(y)))
        y_bottom = int(max(np.array(y)))

        h = (y_bottom - y_top)
        w = (x_right - x_left)

        # print("points",len(points))
        #print("totalx",len(totalx))
        #totalx=totalx/nocero
        #totaly=totaly/nocero
        #cv2.circle(img, (int(totalx),int(totaly)), 10, colors[0],4)
        area = (((x_right-x_left)) * ((y_bottom-y_top)))
        # print('Area: ', area)
        minArea = 2000 
        if area > minArea:
            if deploy==False:
                cv2.rectangle(img, (x_left,y_top), (x_right, y_bottom), (255, 0, 0), 5)
            x_center = x_left + w/2  
            y_center = y_top + h/2
            bbox_list.append([x_center, y_center, w, h])
            scores.append(points_scores_mean)

        if deploy==False:

            # Draw limbs.
            for i, j in skeleton:
                if points_scores[i] > point_score_threshold and points_scores[j] > point_score_threshold:
                    if draw_ellipses:
                        middle = (points[i] + points[j]) // 2
                        vec = points[i] - points[j]
                        length = np.sqrt((vec * vec).sum())
                        angle = int(np.arctan2(vec[1], vec[0]) * 180 / np.pi)
                        polygon = cv2.ellipse2Poly(tuple(middle), (int(length / 2), min(int(length / 50), stick_width)),
                                                angle, 0, 360, 1)
                        cv2.fillConvexPoly(img_limbs, polygon, colors[j])
                    else:
                        cv2.line(img_limbs, tuple(points[i]), tuple(points[j]), color=colors[j], thickness=stick_width)
    
    if deploy==False:
        cv2.addWeighted(img, 0.4, img_limbs, 0.6, 0, dst=img)

    if len(scores)==0:
        scores = []
    if len(bbox_list)==0:
        bbox_list = []

    return img, bbox_list, scores


def print_raw_results(poses, scores):
    log.info('Poses:')
    for pose, pose_score in zip(poses, scores):
        pose_str = ' '.join('({:.2f}, {:.2f}, {:.2f})'.format(p[0], p[1], p[2]) for p in pose)
        log.info('{} | {:.2f}'.format(pose_str, pose_score))

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result