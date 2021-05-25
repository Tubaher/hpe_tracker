# bs-tracker



usage: preprocessed_bb.py [-h] [--input INPUT] [--algo ALGO] [--output OUTPUT] [--config CONFIG]

This program shows how to use background subtraction methods provided by OpenCV. You can process both videos and images.

optional arguments:
  -h, --help       show this help message and exit
  --input INPUT    Path to a video or a sequence of image.
  --algo ALGO      Background subtraction method (KNN, MOG2).
  --output OUTPUT  Path for the output video.
  --config CONFIG  Path a yaml config file for tracker.


## Datasets

The input videos are inside [data](stuff/data) folder (path= `./stuff/dataset/`). 

## Output videos folder

The output videos folder is [output_videos](stuff/output_videos/). (path= `./stuff/output_videos/`). 
 
## Requirements

Install the requirements inside a virtualenv with the following command: `pip install -r requirements.txt` .