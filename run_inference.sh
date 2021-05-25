#!/bin/bash
dir="/docker/input_videos/*"
for video in $dir
do
  echo $video
  IFS="/" read -r -a array <<< "$video" 
  echo "${array[3]}"
  python3 eddy_prueba.py -i $video -o /docker/output_videos/"${array[3]}" -t 0.2 -no_show -m  /opt/intel/openvino_2021.3.394/deployment_tools/open_model_zoo/tools/downloader/intel/human-pose-estimation-0001/FP16/human-pose-estimation-0001.xml -at openpose -d CPU
done

