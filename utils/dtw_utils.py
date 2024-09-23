import cv2
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import os
from easy_vit_pose import vitpose_inference
import json
from scipy.spatial import procrustes
from sklearn.neighbors import NearestNeighbors

def resize_video(video_path, output_size):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame, output_size)
        frames.append(resized_frame)
    cap.release()
    return frames

def icp(a, b, max_iterations=10, tolerance=1e-6):
    src = np.copy(a)
    dst = np.copy(b)
    for i in range(max_iterations):
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(dst)
        distances, indices = nbrs.kneighbors(src)
        T = np.mean(dst[indices.flatten()], axis=0) - np.mean(src, axis=0)
        src += T
        if np.mean(distances) < tolerance:
            break
    return src

def align_icp(series1, series2):
    aligned_series2 = []
    for t in range(len(series1)):
        aligned_series2_t = icp(series1[t], series2[t])
        aligned_series2.append(aligned_series2_t)
    return np.array(aligned_series2)

def construct_keypoint_sequence(keypoint_data):
    keypoints = []
    for frame in keypoint_data["keypoints"]:
        frame_keypoints = []
        for joint in frame["0"]:
            frame_keypoints.append((joint[0], joint[1]))
        keypoints.append(frame_keypoints)
    return np.array(keypoints)

def normalize_sequence(sequence):
    # Normalize based on the centroid and scale
    sequence = np.array(sequence)
    centroids = np.mean(sequence, axis=1, keepdims=True)
    max_dist = np.max(np.linalg.norm(sequence - centroids, axis=2, keepdims=True), axis=1, keepdims=True)
    normalized_sequence = (sequence - centroids) / max_dist
    return normalized_sequence

def get_keypoint_data(video_path, agent):
    _, res = vitpose_inference(video_path, "./testing/" + agent)
    with open(res, 'r') as file:
        keypoint_data = json.load(file)
    return keypoint_data

def compute_dtw_cost_between_videos(video_path_1, video_path_2):
    output_size = (240, 368)  # Resize both videos to the same resolution
    resize_video(video_path_1, output_size)
    resize_video(video_path_2, output_size)
    
    keypoint_data_1 = get_keypoint_data(video_path_1, 'real')

    keypoint_data_2 = get_keypoint_data(video_path_2, 'robot')
    
    keypoints_sequence_1 = construct_keypoint_sequence(keypoint_data_1)
    keypoints_sequence_2 = construct_keypoint_sequence(keypoint_data_2)

    # Use ICP for alignment instead of Procrustes
    aligned_sequence_2 = align_icp(keypoints_sequence_1, keypoints_sequence_2)

    keypoints_sequence_1 = normalize_sequence(keypoints_sequence_1)
    aligned_sequence_2 = normalize_sequence(aligned_sequence_2)

    # Compute DTW using Euclidean distance
    distance, path = fastdtw(keypoints_sequence_1.reshape(-1, 2), aligned_sequence_2.reshape(-1, 2), dist=euclidean)
    
    return distance

if __name__ == "__main__":
    cwd = os.getcwd()
    video_path_1s = [
        os.path.join(cwd, "..", "videos", "running.mp4"),
    ]
    
    video_path_2s = [
        os.path.join(cwd, "trained_run.gif"),

    ]
    
    names = ["run"]

    for i in range(len(video_path_1s)):
        for j in range(len(video_path_2s)):
            vid1 = video_path_1s[i]
            vid2 = video_path_2s[j]
            dtw_cost = compute_dtw_cost_between_videos(vid1, vid2)
            print(f"GT: {names[i]}  Trained: {names[j]}")
            print("DTW alignment cost between the two videos:", dtw_cost)
