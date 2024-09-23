import cv2
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean,cdist
import os
from utils.easy_vit_pose import vitpose_inference
import json
from munkres import Munkres
from scipy.spatial import procrustes
from sklearn.neighbors import NearestNeighbors
 
def icp(a, b, max_iterations=20, tolerance=1e-6):
    src = np.copy(a)
    dst = np.copy(b)
    for i in range(max_iterations):
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(dst)
        distances, indices = nbrs.kneighbors(src)
        T = np.mean(dst[indices], axis=0) - np.mean(src, axis=0)
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
 
def align_procrustes(series1, series2):
    aligned_series2 = []
    for t in range(len(series1)):
        mtx1, mtx2, disparity = procrustes(series1[t], series2[t])
        aligned_series2.append(mtx2)
    return np.array(aligned_series2)
 
def align_joints(series1, series2):
    aligned_series2 = []
    m = Munkres()
    for t in range(len(series1)):
        cost_matrix = cdist(series1[t], series2[t], 'euclidean')
        indexes = m.compute(cost_matrix)
        reordered_series2_t = [series2[t][j] for i, j in indexes]
        aligned_series2.append(reordered_series2_t)
    return np.array(aligned_series2)
 
def construct_kepoint_sequence(keypoint_data):
    keypoints = []
   
    for frame in keypoint_data["keypoints"]:
        frame_keypoints = []
        for joint in frame["0"]:
            frame_keypoints.append((joint[0],joint[1]))
        keypoints.append(frame_keypoints)
    return np.array(keypoints)
 
# Normalization function for sequences
def normalize_sequence(sequence):
    sequence = np.array(sequence)
    min_val = np.min(sequence)
    max_val = np.max(sequence)
    normalized_sequence = (sequence - min_val) / (max_val - min_val)
    return normalized_sequence
 
def get_keypoint_data(video_path):
    _,res = vitpose_inference(video_path,"./testing")
   
    with open(res, 'r') as file:
        keypoint_data = json.load(file)
   
    return keypoint_data
   
# Main function to compute DTW cost between two videos
def compute_dtw_cost_between_videos(video_path_1, video_path_2):
   
    keypoint_data_1 = get_keypoint_data(video_path_1)
    input()
    keypoint_data_2 = get_keypoint_data(video_path_2)
   
    keypoints_sequence_1 = construct_kepoint_sequence(keypoint_data_1)
    keypoints_sequence_2 = construct_kepoint_sequence(keypoint_data_2)
   
    aligned_sequence_2 = align_icp(keypoints_sequence_1, keypoints_sequence_2)
   
    keypoints_sequence_1 = keypoints_sequence_1.reshape(-1, 2)
    aligned_sequence_2 = aligned_sequence_2.reshape(-1, 2)
 
    keypoints_sequence_1 = normalize_sequence(keypoints_sequence_1)
    aligned_sequence_2 = normalize_sequence(aligned_sequence_2)
 
    distance, path = fastdtw(keypoints_sequence_1, aligned_sequence_2, dist=euclidean)
 
    return distance
 
 
if __name__ == "__main__":
    cwd = (os.getcwd())
    # Example usage:
    video_path_1s = [
        os.path.join(cwd,"..","videos","running.mp4"),
        # os.path.join(cwd,"..","videos","hop.mp4"),
        # os.path.join(cwd,"..","videos","pace.mp4")
    ]
   
    video_path_2s = [
        os.path.join(cwd,"..","results","bound","031143.029789","videos","00961.mp4"),
        # os.path.join(cwd,"..","results","checkpoints","hop","videos","play.mp4"),
        # os.path.join(cwd,"..","results","checkpoints","pace","videos","play.mp4"),
    ]
   
    names = ["run"]
 
    for i in range(len(video_path_1s)):
        for j in range(len(video_path_2s)):
            vid1 = video_path_1s[i]
            vid2 = video_path_2s[j]
            dtw_cost = compute_dtw_cost_between_videos(vid1, vid2)
            print(f"GT: {names[i]}  Trained: {names[j]}")
            print("DTW alignment cost between the two videos:", dtw_cost)

print('DONE')