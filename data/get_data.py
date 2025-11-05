import cv2
import numpy as np
import os 
from pathlib import Path
def load_y4m_video(filepath, max_frames = None):
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {filepath}")
    frames = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        y_channel = yuv_frame[:,:,0]
        frames.append(y_channel)
        frame_count += 1
        if max_frames and frame_count >= max_frames:
            break
    cap.release()
    return np.array(frames)
def extract_blocks(frames,block_size=8):
    num_frames,height,width = frames.shape
    blocks_h = height // block_size
    blocks_w = width // block_size
    height_trim = blocks_h * block_size
    width_trim = blocks_w * block_size
    frames = frames[:,:height_trim,:width_trim]
    blocks = []
    for frame in frames:
        for i in range(0,height_trim,block_size):
            for j in range(0,width_trim,block_size):
                block = frame[i:i+block_size,j:j+block_size]
                blocks.append(block)
    return np.array(blocks)
def load_training_data(video_dir,video_names,max_frames_per_video=300):
    all_blocks = []
    for video_name in video_names:
        filepath = Path(video_dir) / f"{video_name}_cif.y4m"
        print(f"Loading {video_name}...")
        frames = load_y4m_video(filepath,max_frames_per_video)
        blocks=extract_blocks(frames,block_size=8)
        all_blocks.append(blocks)
    all_blocks=np.vstack(all_blocks)
    return all_blocks
if __name__ == "__main__":
    VIDEO_DIR = "./cif_videos"
    TRAINING_VIDEOS = ["foreman","akiyo","news","mobile"]
    training_blocks = load_training_data(VIDEO_DIR,TRAINING_VIDEOS,max_frames_per_video = 300)
    training_vectors = training_blocks.reshape(len(training_blocks),-1)
    print(f"Training vectors shape: {training_vectors.shape}")
    np.save("training_blocks.npy",training_blocks)
    np.save("training_vectors.npy",training_vectors)
    print("\nSaved training data")