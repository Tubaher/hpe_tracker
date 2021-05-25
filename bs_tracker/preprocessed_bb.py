from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np
from deep_sort_pytorch.deep_sort import build_tracker
from deep_sort_pytorch.utils.draw import draw_boxes
from deep_sort_pytorch.utils.parser import get_config
from lc_logic import Line_cross, Person

#Ubicacion line crossing



parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='test.mp4')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='KNN')
parser.add_argument('--output', type=str, help='Path for the output video.', default='output_processed.mp4')
parser.add_argument('--config', type=str, help='Path a yaml config file for tracker.', default='./deep_sort_pytorch/configs/deep_sort.yaml')

args = parser.parse_args()
if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
if args.algo == 'CNT':
    backSub = cv.bgsegm.BackgroundSubtractorCNT()
if args.algo == 'LSBP':
    backSub = cv.bgsegm.createBackgroundSubtractorLSBP()
else:
    backSub = cv.createBackgroundSubtractorKNN()
    backSub.setDetectShadows(False) #Set shadows detection
    backSub.setHistory(70) #Sets the number of last frames that affect the background mode
    #backSub.setDist2Threshold(100)
    backSub.setShadowThreshold(0.5)
capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))
if not capture.isOpened():
    print('Unable to open: ' + args.input)
    exit(0)

frame_width = int(capture.get(3))
frame_height = int(capture.get(4))

out = cv.VideoWriter(args.output,cv.VideoWriter_fourcc('m','p','4','v'), 30, ( frame_width ,frame_height), 1)
cfg = get_config()
cfg.merge_from_file("./deep_sort_pytorch/configs/yolov3.yaml")
cfg.merge_from_file(args.config)
deepsort_person = build_tracker(cfg, use_cuda=1)

#line_pos = ((37,368),(37,493))
line_pos = ((104,368),(104,493))
lc = Line_cross(line_pos)
length = int(capture.get(cv.CAP_PROP_FRAME_COUNT)) #
frames = 0
while True:
    ret, frame = capture.read()
    #frame_width = int(capture.get(3))
    #frame_height = int(capture.get(4))
    if frame is None:
        break
    frames += 1    
    print(f'status: {frames}/{length}\n')
    #Definir dimensiones ROI
    #roiy = 136 #Y inicial
    #roih = 480 #Altura
    #roix = 0   #X inicial
    #roiw = 595 #Ancho
    #print('size frame antes', frame.dtype)
    #frame = frame[roiy:roiy+roih, roix:roix+roiw]

    fgMask = backSub.apply(frame)
    
    cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(4,4))

    #fgMask = cv.morphologyEx(fgMask, cv.MORPH_CLOSE, kernel)# 
    cnts = cv.findContours(fgMask, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_KCOS)[0]
    
    
    #cv.imshow('Frame', frame)
    #cv.imshow('FG Mask', fgMask)

    
    bbox_list = []
    for i,cnt in enumerate(cnts):
        if cv.contourArea(cnt) > 4500 and cv.contourArea(cnt) < 50000: #en 13000 no genera bb para senorita claro. 
            x, y, w, h = cv.boundingRect(cnt)
            # print(f'Contorno {i} con puntos: {x},{y},{w},{h}')
            x_center = x+(w/2)
            y_center = y+(h/2)
            bbox_xywh = [x_center,y_center,w,h]
            bbox_list.append(bbox_xywh)
            #print bbox center
            #cx = int(x+(w/2))
            #cy = int(y+(h/2))
            #cv.circle(frame,(cx,cy), 5, (255, 255, 255), -1)    
            #cv.putText(frame, str((lc.track[] ,(cx,cy)), (15, 15),
            #   cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,255))
            
    
            # print(bbox_xywh)
            cv.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0) , 5)
            cv.putText(frame, str(cv.contourArea(cnt)), (x+w,y+h),
            cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    #print line of linecrossing
    cv.line(frame,line_pos[0],line_pos[1],(0,0,255),5)

    # print(np.array(bbox_list).shape)    
    if len(bbox_list) > 0:
        outputs_person = deepsort_person.update(
                            np.array(bbox_list), [0.5]*len(bbox_list), frame)

        if len(outputs_person) > 0:
            bbox_xyxy_person = outputs_person[:, :4]
            identities_person = outputs_person[:, -1]
            frame = draw_boxes(
                        frame, bbox_xyxy_person, identities_person)

            center_point_x = (bbox_xyxy_person[:, 0] + ((bbox_xyxy_person[:, 2]-bbox_xyxy_person[:, 0])/2))
            center_point_y = (bbox_xyxy_person[:, 1] + ((bbox_xyxy_person[:, 3]-bbox_xyxy_person[:, 1])/2))
            # print(f'values in center_point_x: {center_point_x}')
            center_point = list(zip(center_point_x, center_point_y))
            # print(f'values in center_point: {center_point}')
            # print(f'values of identities: {identities_person}')

            directions = lc.get_ids_directions(center_point, identities_person)
            
            #print('center point', center_point)
            for item in center_point:
                #print('par-------',par )
                cx = int(item[0])
                cy = int(item[1])
                cv.circle(frame,(cx,cy), 5, (255, 255, 255), -1)      
                cv.putText(frame, str((cx,cy)), (cx+10, cy),cv.FONT_HERSHEY_SIMPLEX, 1 , (255,255,255), thickness=3) 
            # print(directions)
            for id in identities_person:
                if not id in lc.track:
                    lc.track[id] = []
                
                if (directions[id] not in lc.track[id]) and (directions[id] != None):
                    lc.track[id].append(directions[id])
                
            
            #cv.putText(frame, str()), (15, 15),
            #   cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
                  
    #Line_cross(((99,368),(99,493)))    
    lc.count()   
    cv.putText(frame, str(('Input: ',str(lc.count_entrada))), line_pos[0],cv.FONT_HERSHEY_SIMPLEX, 1 , (255,255,255), thickness=3) 
    cv.putText(frame, str(('Output: ',str(lc.count_salida))), line_pos[1],cv.FONT_HERSHEY_SIMPLEX, 1 , (255,255,255), thickness=3) 

    out.write(frame)
    #cv.imshow('cnts', fgMask)
    #cv.imshow('cnts', frame)
    keyboard = cv.waitKey(30)
    
    if keyboard == 'q' or keyboard == 27:
        break
# lc.count()
lc.get_results(args.output)
print(f'Number of salida:{lc.count_salida}  and Number of entrada: {lc.count_entrada}') 
print(lc.track)