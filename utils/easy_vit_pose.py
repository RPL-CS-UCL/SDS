import argparse
import json
import os
from PIL import Image
import cv2
import numpy as np

from easy_ViTPose.vit_utils.inference import NumpyEncoder, VideoReader
from easy_ViTPose.inference import VitInference
from easy_ViTPose.vit_utils.visualization import joints_dict

from utils.vid_utils import read_video
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import shutil

ROOT_DIR = f"{os.getcwd()}/.."
ViTPOSE_CPT = f"{ROOT_DIR}/easy_ViTPose/checkpoints/vitpose-h-ap10k.pth"
MODEL_TYPE = "h"
DATASET = "ap10k"
# YOLO_CPT = f"{ROOT_DIR}/easy_ViTPose/checkpoints/yolov8x.pt"
YOLO_CPT = f"{ROOT_DIR}/easy_ViTPose/checkpoints/yolov8x-oiv7.pt"
SINGLE_POSE = True
# YOLO_SIZE = 256
YOLO_SIZE = 96

class default_args():
    yolo_step = 1
    save_img = False
    save_video = True
    show_yolo = False
    show_raw_yolo = False
    conf_threshold = 0
    save_json = False
    is_video=True
    

def vitpose_inference(video_path,output_path,args=default_args):
    
    reader = read_video(video_path,0,False)
    
    try:
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
    except:
        print("Failed to remove directory")

    # Attempt to create the directory
    try:
        os.makedirs(output_path+"/kp",exist_ok=True)
    except OSError as e:
        pass


    # Initialize model
    model = VitInference(ViTPOSE_CPT, YOLO_CPT, MODEL_TYPE, DATASET,
                         YOLO_SIZE, is_video=args.is_video,
                         single_pose=SINGLE_POSE,
                         yolo_step=args.yolo_step)  # type: ignore

    keypoints = []
    frames = []
    # fps = []
    # tot_time = 0.
    for (ith, img) in enumerate(reader):
        # t0 = time.time()
        
        # img = crop_to_dim(img)
        # Run inference
        frame_keypoints = model.inference(img)
        keypoints.append(frame_keypoints)

        img = model.draw(args.show_yolo, args.show_raw_yolo, args.conf_threshold)[..., ::-1]
        
        frames.append(img)
            
        if args.save_img: 
            if args.save_img:
                cv2.imwrite(f"{output_path}/kp/res_{ith}.png", img)

    # if is_video:
    #     tot_poses = sum(len(k) for k in keypoints)
    #     print(f'>>> Mean inference FPS: {1 / np.mean(fps):.2f}')
    #     print(f'>>> Total poses predicted: {tot_poses} mean per frame: '
    #           f'{(tot_poses / (ith + 1)):.2f}')
    #     print(f'>>> Mean FPS per pose: {(tot_poses / tot_time):.2f}')
    
    out = {'keypoints': keypoints,
                   'skeleton': joints_dict()["ap10k"]['keypoints']}
    
    with open(f"{output_path}/res.json", 'w') as f:
        json.dump(out, f, cls=NumpyEncoder)
        
    if args.save_video:
        height, width, layers = frames[0].shape
        fps = 30  # Frames per second
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
        out = cv2.VideoWriter(f'{output_path}/annotated_video.mp4', fourcc, fps, (width, height))

        # Write each image to the video
        for image in frames:
            out.write(image)
            
        out.release()
    
    return f'{output_path}/annotated_video.mp4'
