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

logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.CRITICAL, stream=sys.stdout)
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

    model = get_model(ie, args, frame.shape[1] / frame.shape[0])
    hpe_pipeline = AsyncPipeline(ie, model, plugin_config, device=args.device, max_num_requests=args.num_infer_requests)

    log.info('Starting inference...')
    hpe_pipeline.submit_data(frame, 0, {'frame': frame, 'start_time': start_time})
    next_frame_id = 1
    next_frame_id_to_show = 0

    presenter = monitors.Presenter(args.utilization_monitors, 55,
                                   (round(frame.shape[1] / 4), round(frame.shape[0] / 8)))

    cfg = get_config()
    cfg.merge_from_file("./bs_tracker/deep_sort_pytorch/configs/yolov3.yaml")
    cfg.merge_from_file(args.config)
    deepsort_person = build_tracker(cfg, use_cuda=1)

    lc = Line_cross(pos_lc)
    lc1 = Line_cross(pos_lc1)
    lc2 = Line_cross(pos_lc2)
    num_frame = 0
    while True:
        if hpe_pipeline.callback_exceptions:
            raise hpe_pipeline.callback_exceptions[0]
        results = hpe_pipeline.get_result(next_frame_id_to_show)
        if results:
            (poses, scores), frame_meta = results
            frame = frame_meta['frame']
            num_frame +=1

            # print(f'Number of frames: {num_frame}')

            orig_frame = frame.copy()
            start_time = frame_meta['start_time']

            if len(poses) and args.raw_output_message: 
                print_raw_results(poses, scores)

            presenter.drawGraphs(frame)
            frame, bbox_list, scores_box = draw_poses(frame, poses, args.prob_threshold, deploy=True)
            array_bbox = np.array(bbox_list)

            if bbox_list:
                if len(bbox_list) > 0:
                    outputs_person = deepsort_person.update(
                                        array_bbox, scores_box, orig_frame)

                    if len(outputs_person) > 0:
                        bbox_xyxy_person = outputs_person[:, :4]
                        identities_person = outputs_person[:, -1]

                        center_point_x = (bbox_xyxy_person[:, 0] + ((bbox_xyxy_person[:, 2]-bbox_xyxy_person[:, 0])/2))
                        center_point_y = (bbox_xyxy_person[:, 1] + ((bbox_xyxy_person[:, 3]-bbox_xyxy_person[:, 1])/2))
                        center_point = list(zip(center_point_x, center_point_y))


                        directions = lc.get_ids_directions(center_point, identities_person)
                        directions1 = lc1.get_ids_directions(center_point, identities_person)
                        directions2 = lc2.get_ids_directions(center_point, identities_person)
                       
                        for item in center_point:
                            cx = int(item[0])
                            cy = int(item[1])
                            cv2.circle(frame,(cx,cy), 5, (255, 255, 255), -1)      
                            
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
                
                    
                lc.count()
                lc1.count()
                lc2.count()

            next_frame_id_to_show += 1
            continue

        if hpe_pipeline.is_ready():

            start_time = perf_counter()
            frame = cap.read()

            if frame is None:

                break

            frame=rotate_image(frame,ang)
            frame=frame[y1:y1+h, x1:x1+w]

            # Submit for inference
            hpe_pipeline.submit_data(frame, next_frame_id, {'frame': frame, 'start_time': start_time})
            next_frame_id += 1

        else:
            hpe_pipeline.await_any()

    hpe_pipeline.await_all()


    metrics.print_total()
    print(presenter.reportMeans())
    lc.get_results(args.input)
    print(f'Number of salida:{lc.count_salida}  and Number of entrada: {lc.count_entrada}') 
    lc1.get_results(args.input)
    print(f'Number of salida:{lc1.count_salida}  and Number of entrada: {lc1.count_entrada}') 
    lc2.get_results(args.input)
    print(f'Number of salida:{lc2.count_salida}  and Number of entrada: {lc2.count_entrada}') 


if __name__ == '__main__':
    sys.exit(main() or 0)
