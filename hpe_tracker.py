import logging
import sys
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from openvino.inference_engine import IECore

from utils import *
from model import *

from bs_tracker.deep_sort_pytorch.deep_sort import build_tracker
from bs_tracker.deep_sort_pytorch.utils.draw import draw_boxes
from bs_tracker.deep_sort_pytorch.utils.parser import get_config
from bs_tracker.lc_logic import Line_cross, Person

sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python'))
import models
import monitors
from images_capture import open_images_capture
from pipelines import AsyncPipeline
from performance_metrics import PerformanceMetrics

logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.INFO, stream=sys.stdout)
log = logging.getLogger()


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument('-m', '--model', help='Required. Path to an .xml file with a trained model.',
                      required=True, type=Path)
    args.add_argument('-at', '--architecture_type', help='Required. Specify model\' architecture type.',
                      type=str, required=True, choices=('ae', 'openpose'))
    args.add_argument('-i', '--input', required=True,
                      help='Required. An input to process. The input must be a single image, '
                           'a folder of images, video file or camera id.')
    args.add_argument('--loop', default=False, action='store_true',
                      help='Optional. Enable reading the input in a loop.')
    args.add_argument('-o', '--output', required=False,
                      help='Optional. Name of output to save.')
    args.add_argument('-limit', '--output_limit', required=False, default=0, type=int,
                       help='Optional. Number of frames to store in output. '
                            'If 0 is set, all frames are stored.')
    args.add_argument('-d', '--device', default='CPU', type=str,
                      help='Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is '
                           'acceptable. The sample will look for a suitable plugin for device specified. '
                           'Default value is CPU.')
    parser.add_argument('--config', type=str, help='Path a yaml config file for tracker.', default='./bs_tracker/deep_sort_pytorch/configs/deep_sort.yaml')

    common_model_args = parser.add_argument_group('Common model options')
    common_model_args.add_argument('-t', '--prob_threshold', default=0.1, type=float,
                                   help='Optional. Probability threshold for poses filtering.')
    common_model_args.add_argument('--tsize', default=None, type=int,
                                   help='Optional. Target input size. This demo implements image pre-processing '
                                        'pipeline that is common to human pose estimation approaches. Image is first '
                                        'resized to some target size and then the network is reshaped to fit the input '
                                        'image shape. By default target image size is determined based on the input '
                                        'shape from IR. Alternatively it can be manually set via this parameter. Note '
                                        'that for OpenPose-like nets image is resized to a predefined height, which is '
                                        'the target size in this case. For Associative Embedding-like nets target size '
                                        'is the length of a short first image side.')

    infer_args = parser.add_argument_group('Inference options')
    infer_args.add_argument('-nireq', '--num_infer_requests', help='Optional. Number of infer requests',
                            default=1, type=int)
    infer_args.add_argument('-nstreams', '--num_streams',
                            help='Optional. Number of streams to use for inference on the CPU or/and GPU in throughput '
                                 'mode (for HETERO and MULTI device cases use format '
                                 '<device1>:<nstreams1>,<device2>:<nstreams2> or just <nstreams>).',
                            default='', type=str)
    infer_args.add_argument('-nthreads', '--num_threads', default=None, type=int,
                            help='Optional. Number of threads to use for inference on CPU (including HETERO cases).')

    io_args = parser.add_argument_group('Input/output options')
    io_args.add_argument('-no_show', '--no_show', help="Optional. Don't show output.", action='store_true')
    io_args.add_argument('-u', '--utilization_monitors', default='', type=str,
                         help='Optional. List of monitors to show initially.')

    debug_args = parser.add_argument_group('Debug options')
    debug_args.add_argument('-r', '--raw_output_message', help='Optional. Output inference results raw values showing.',
                            default=False, action='store_true')
    return parser


def main():
    pos_lc  = ((47,35),(47,239))
    pos_lc1 = ((375,40),(375,290))
    pos_lc2 = ((350,40),(350,290))
    # Variables for crop and rotation
    ang=-24
    w=640
    h=360
    y1=65
    x1=85

    args = build_argparser().parse_args()
    metrics = PerformanceMetrics()

    log.info('Initializing Inference Engine...')
    ie = IECore()

    plugin_config = get_plugin_configs(args.device, args.num_streams, args.num_threads)

    cap = open_images_capture(args.input, args.loop)

    start_time = perf_counter()
    frame = cap.read()

    if frame is None:
        raise RuntimeError("Can't read an image from the input")

    frame=rotate_image(frame,ang)
    frame=frame[y1:y1+h, x1:x1+w]
    
    log.info('Loading network...')
    #top, bottom, left, right = 300,700,0,500
    #print('O Ratio', frame.shape[1] / frame.shape[0])
    #frame = frame[300:700, 0:711]
    #print('F Ratio', frame.shape[1] / frame.shape[0])
    #cv2.imwrite('/docker/crop.jpg', frame)
    model = get_model(ie, args, frame.shape[1] / frame.shape[0])
    hpe_pipeline = AsyncPipeline(ie, model, plugin_config, device=args.device, max_num_requests=args.num_infer_requests)

    log.info('Starting inference...')
    hpe_pipeline.submit_data(frame, 0, {'frame': frame, 'start_time': start_time})
    next_frame_id = 1
    next_frame_id_to_show = 0

    presenter = monitors.Presenter(args.utilization_monitors, 55,
                                   (round(frame.shape[1] / 4), round(frame.shape[0] / 8)))
    video_writer = cv2.VideoWriter()
    #print('FRAME SHAPE: ', frame.shape)
    #out = cv2.VideoWriter(args.output,cv2.VideoWriter_fourcc('m','p','4','v'), 30, ( frame_width ,frame_height), 1)
    if args.output and not video_writer.open(args.output, cv2.VideoWriter_fourcc(*'MJPG'), cap.fps(),
        (frame.shape[1], frame.shape[0])):
        raise RuntimeError("Can't open video writer")

    print("To close the application, press 'CTRL+C' here or switch to the output window and press ESC key")

    cfg = get_config()
    cfg.merge_from_file("./bs_tracker/deep_sort_pytorch/configs/yolov3.yaml")
    cfg.merge_from_file(args.config)
    deepsort_person = build_tracker(cfg, use_cuda=1)
    
    #line_pos = ((104,368),(104,493))
    #lc = Line_cross(line_pos)

    lc = Line_cross(pos_lc)
    lc1 = Line_cross(pos_lc1)
    lc2 = Line_cross(pos_lc2)
    num_frame = 0
    while True:
        if hpe_pipeline.callback_exceptions:
            print('hpe_pipeline.callback_exceptions[0]')
            print(hpe_pipeline.callback_exceptions)
            raise hpe_pipeline.callback_exceptions[0]
        # Process all completed requests

        # print('*************************before model')
        results = hpe_pipeline.get_result(next_frame_id_to_show)
        if results:
            (poses, scores), frame_meta = results
            frame = frame_meta['frame']
            num_frame +=1

            print(f'Number of frames: {num_frame}')

            orig_frame = frame.copy()

            # print(f'Len of frame: {frame.shape}')
            # print(f'Len of orig: {orig_frame.shape}')
            start_time = frame_meta['start_time']

            if len(poses) and args.raw_output_message: 
                print_raw_results(poses, scores)

            presenter.drawGraphs(frame)
            # print(f'Len of draw: {len(draw_poses(frame, poses, args.prob_threshold))}')
            frame, bbox_list, scores_box = draw_poses(frame, poses, args.prob_threshold)
            # print(f'frame shape: {frame.shape}')
            # print(f'Bbox list values: {bbox_list}')
            #print('frame: ', frame)
            #print('poses: ', poses)
            #frame, bbox_list = draw_poses(frame, poses)
            array_bbox = np.array(bbox_list)

            
            cv2.line(frame,pos_lc[0],pos_lc[1],(0,0,255),2)
            cv2.line(frame,pos_lc1[0],pos_lc1[1],(0,255,0),2)
            cv2.line(frame,pos_lc2[0],pos_lc2[1],(255,0,0),2)

            if bbox_list:
                if len(bbox_list) > 0:
                    # print(sys.path)
                    #print(f'Array bbox shape: {array_bbox.shape}' )
                    #print(f'Length of orig_frame after crop: {orig_frame.shape}')
                    outputs_person = deepsort_person.update(
                                        array_bbox, scores_box, orig_frame)

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
                        directions1 = lc1.get_ids_directions(center_point, identities_person)
                        directions2 = lc2.get_ids_directions(center_point, identities_person)

                        print(f'Values of directions: {directions}')
                        print(f'Values of directions1: {directions1}')
                        print(f'Values of directions2: {directions2}')
                        
                        #print('center point', center_point)
                        for item in center_point:
                            #print('par-------',par )
                            cx = int(item[0])
                            cy = int(item[1])
                            cv2.circle(frame,(cx,cy), 5, (255, 255, 255), -1)      
                            #cv2.putText(frame, str((cx,cy)), (cx+10, cy),cv2.FONT_HERSHEY_SIMPLEX, 1 , (255,255,255), thickness=3) 
                            

                        # print(directions)
                        for id in identities_person:
                            if not id in lc.track:
                                lc.track[id] = []
                            
                            if not id in lc1.track:
                                lc1.track[id] = []
                            
                            if not id in lc2.track:
                                lc2.track[id] = []
                                
                            
                            if (directions[id] not in lc.track[id]) and (directions[id] != None):
                                lc.track[id].append(directions[id])

                            if (directions1[id] not in lc1.track[id]) and (directions1[id] != None):
                                lc1.track[id].append(directions1[id])
                            
                            if (directions2[id] not in lc2.track[id]) and (directions2[id] != None):
                                lc2.track[id].append(directions2[id])
                #print(poses)
                metrics.update(start_time, frame)
                if video_writer.isOpened(): #and (args.output_limit <= 0 or next_frame_id_to_show <= args.output_limit-1):
                    
                    lc.count()
                    lc1.count()
                    lc2.count()
                    #lc
                    cv2.putText(frame, str(('Input: ',str(lc.count_entrada))), pos_lc[0],cv2.FONT_HERSHEY_SIMPLEX, 1 , (0,0,255), thickness=2) 
                    cv2.putText(frame, str(('Output: ',str(lc.count_salida))), pos_lc[1],cv2.FONT_HERSHEY_SIMPLEX, 1 , (0,0,255), thickness=2) 
                    #lc1
                    cv2.putText(frame, str(('Input: ',str(lc1.count_entrada))), pos_lc1[0],cv2.FONT_HERSHEY_SIMPLEX, 1 , (0,255,0), thickness=2) 
                    cv2.putText(frame, str(('Output: ',str(lc1.count_salida))), pos_lc1[1],cv2.FONT_HERSHEY_SIMPLEX, 1 , (0,255,0), thickness=2) 
                    #lc2 ((350,90),(350,290))
                    cv2.putText(frame, str(('Input: ',str(lc2.count_entrada))), (350,55),cv2.FONT_HERSHEY_SIMPLEX, 1 , (255,0,0), thickness=2) 
                    cv2.putText(frame, str(('Output: ',str(lc2.count_salida))), (350,325),cv2.FONT_HERSHEY_SIMPLEX, 1 , (255,0,0), thickness=2)     
                    video_writer.write(frame) ##WRITER
                if not args.no_show:
                    cv2.imshow('Pose estimation results', frame)
                    key = cv2.waitKey(1)

                    ESC_KEY = 27
                    # Quit.
                    if key in {ord('q'), ord('Q'), ESC_KEY}:
                        print('BREAK KEY = Q')
                        break
                    presenter.handleKey(key)
            next_frame_id_to_show += 1
            continue

        if hpe_pipeline.is_ready():
            # Get new image/frame
            start_time = perf_counter()
            frame = cap.read()

            #if frame is not None:
                #frame = frame[300:700, 0:711]
            if frame is None:
                # print('BREAK hpe_pipeline.is_ready()')
                break

            frame=rotate_image(frame,ang)
            frame=frame[y1:y1+h, x1:x1+w]

            # Submit for inference
            hpe_pipeline.submit_data(frame, next_frame_id, {'frame': frame, 'start_time': start_time})
            next_frame_id += 1

        else:
            # Wait for empty request
            # print('hpe_pipeline.await_any()')
            hpe_pipeline.await_any()

    # print('hpe_pipeline.await_all()')
    hpe_pipeline.await_all()
    # Process completed requests
    '''
    while hpe_pipeline.has_completed_request():
        results = hpe_pipeline.get_result(next_frame_id_to_show)
        if results:
            (poses, scores), frame_meta = results
            frame = frame_meta['frame']

            orig_frame = frame.copy()
            start_time = frame_meta['start_time']

            if len(poses) and args.raw_output_message:
                print_raw_results(poses, scores)

            presenter.drawGraphs(frame)
            frame, bbox_list, scores_box = draw_poses(frame, poses, args.prob_threshold)
            array_bbox = np.array(bbox_list)

            for i,box in enumerate(array_bbox):
                x1,y1,x2,y2 = [int(i) for i in box]
                x1 += offset[0]
                x2 += offset[0]
                y1 += offset[1]
                y2 += offset[1]
                # box text and bar
                id = int(identities[i]) if identities is not None else 0    
                color = compute_color_for_labels(id)
                #label = '{}{:d}'.format("", id)
                #label_score = '{}{:f}'.format("", scores_box[i])
                label_score=str(int(scores_box[i]))
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
                cv2.rectangle(frame,(x1, y1),(x2,y2),color,3)
                cv2.rectangle(frame,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
                cv2.putText(frame,label_score,(x1,y1+t_size[1]-15), cv2.FONT_HERSHEY_PLAIN, 2, [255,0,255], 2)


            
            if bbox_list:
                if len(bbox_list) > 0:
                    print(f'Array bbox shape: {array_bbox.shape}' )
                    outputs_person = deepsort_person.update(
                            array_bbox, scores_box, orig_frame)

                    if len(outputs_person) > 0:
                        bbox_xyxy_person = outputs_person[:, :4]
                        identities_person = outputs_person[:, -1]
                        frame = draw_boxes(
                            frame, bbox_xyxy_person, identities_person)

                metrics.update(start_time, frame)
                if video_writer.isOpened() and (args.output_limit <= 0 or next_frame_id_to_show <= args.output_limit-1):
                    print('Enters here --------')
                    lc.count()
                    lc1.count()
                    lc2.count()
                    video_writer.write(frame)
                if not args.no_show:
                    cv2.imshow('Pose estimation results', frame)
                    key = cv2.waitKey(1)

                    ESC_KEY = 27
                    # Quit.
                    if key in {ord('q'), ord('Q'), ESC_KEY}:
                        break
                    presenter.handleKey(key)
                next_frame_id_to_show += 1
        else:
            break
    '''


    metrics.print_total()
    print(presenter.reportMeans())
    lc.get_results(args.output)
    print(f'Number of salida:{lc.count_salida}  and Number of entrada: {lc.count_entrada}') 
    lc1.get_results(args.output)
    print(f'Number of salida:{lc1.count_salida}  and Number of entrada: {lc1.count_entrada}') 
    lc2.get_results(args.output)
    print(f'Number of salida:{lc2.count_salida}  and Number of entrada: {lc2.count_entrada}') 


if __name__ == '__main__':
    sys.exit(main() or 0)
