import argparse
from track import *
import logging

#Parsers
parser = argparse.ArgumentParser()
parser.add_argument('-mp','--model_path',help='path to model',type=str)
parser.add_argument('-vp','--video_path',help='path to the video',type=str)
parser.add_argument('-od','--output_dir',help='path to save the video',type=str)

logging.basicConfig(format='[%(levelname)s]: %(message)s', level=logging.INFO) 


if __name__=='__main__':
    args = parser.parse_args()
    out_dir = args.output_dir ## VIdeo output dir
    model_path = args.model_path #Weights path
    video_path = args.video_path #INput video path
    
    dl = datasets.LoadVideo(video_path, (1088,608))##Load video, get video info, rotate video, crop video, resize, normalize
    #Load all the initial configurations of the model.
    opt = opts().init()
    opt.load_model = model_path
    show_image = False
    output_dir = out_dir

    # Executes the model according with the configurations.
    eval_seq(opt, dl, 'mot',save_dir=output_dir, show_image=show_image)
