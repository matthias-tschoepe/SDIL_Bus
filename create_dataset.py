import os
import numpy as np
import cv2
import h5py

from glob import glob
from tqdm import tqdm

def extract_sliding_windows(video_path, window_size=16, step_size=8):

    video_frames = []
    vid_cap = cv2.VideoCapture(video_path)
    suc, img = vid_cap.read()
    while suc:
        video_frames.append(img)
        suc, img = vid_cap.read()
    vid_cap.release()

    windows = []
    # for i in range(0, len(video_frames) - window_size + 1, step_size):
    win_start_frame = 0
    win_end_frame = window_size
    while win_end_frame <= len(video_frames):
        window = video_frames[win_start_frame:win_end_frame]
        windows.append(window)

        win_start_frame += step_size
        win_end_frame = win_start_frame + window_size

    return windows

# extract_sliding_windows(video_path=root_path + "VIOLENCE\\VIOLENCE_183.mp4")

if __name__ == '__main__':
    root_path = "\\\\169.254.213.20\\NAS_Data\\Datasets\\SDIL (Bus-Videos)\\"
    save_path = root_path + "Sliding_Windows" + os.sep

    video_paths = [elem for elem in glob(root_path + "*\\*.mp4")]
    video_paths = sorted(video_paths, reverse=True)

    train_txt = root_path + "train.txt"
    test_txt = root_path + "test.txt"

    train_video_names = list(np.loadtxt(train_txt, dtype=str))
    test_video_names = list(np.loadtxt(test_txt, dtype=str))

    # print(train_video_names)
    # print(test_video_names)

    print("VIOLENCE_500.mp4 in train_video_names:", 'VIOLENCE_500.mp4' in train_video_names)
    print("VIOLENCE_500.mp4 in test_video_names:", 'VIOLENCE_500.mp4' in test_video_names)

    if not os.path.isdir(save_path + "Train"):
        os.makedirs(save_path + "Train")

    if not os.path.isdir(save_path + "Test"):
        os.makedirs(save_path + "Test")

    WIN_SIZE = 16
    STEP_SIZE = 8
    data = []
    labels = []

    for video_path in tqdm(video_paths):
        video_name = '.'.join(video_path.split(os.sep)[-1].split('.')[:-1])
        video_name_with_extension = video_path.split(os.sep)[-1]
        class_name = video_path.split(os.sep)[-2]
        windows = extract_sliding_windows(video_path=video_path, window_size=WIN_SIZE, step_size=STEP_SIZE)

        # data.extend(windows)
        # labels.extend([class_name] * len(windows))

        train_or_test = "Train" if video_name_with_extension in train_video_names else "Test"
        with h5py.File(save_path + train_or_test + os.sep + video_name + ".hdf5", "w") as hf:
            # hf.create_dataset('data', data=windows, compression="gzip", compression_opts=9, chunks=True)
            hf.create_dataset('data', data=windows, compression=9)
            hf.create_dataset('labels', data=class_name)








"""
import cv2
import os
import h5py
import numpy as np

def extract_sliding_windows(video_path, window_size=16, step_size=8):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()

    windows = []
    for i in range(0, len(frames) - window_size + 1, step_size):
        window = frames[i:i+window_size]
        windows.append(window)

    return windows

def process_videos(videos_dir, output_hdf5_file, window_size=16, step_size=8):
    data = []
    labels = []
    
    for label, class_dir in enumerate(['vio', 'non_vio']):
        videos_path = os.path.join(videos_dir, class_dir)
        for video_file in os.listdir(videos_path):
            video_path = os.path.join(videos_path, video_file)
            windows = extract_sliding_windows(video_path, window_size, step_size)
            data.extend(windows)
            labels.extend([label] * len(windows))
    
    data = np.array(data)
    labels = np.array(labels)
    
    with h5py.File(output_hdf5_file, 'w') as hf:
        hf.create_dataset('data', data=data)
        hf.create_dataset('labels', data=labels)

# Example usage:
videos_dir = '/path/to/videos'
output_hdf5_file = 'sliding_windows.h5'
process_videos(videos_dir, output_hdf5_file)
"""


"""
import cv2
import os
import h5py
import numpy as np

def extract_frames(video_path, num_frames=16, step_size=8):
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(0, total_frames, step_size):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (224, 224))  # Resize frames if necessary
            frames.append(frame)
        if len(frames) == num_frames:
            break

    cap.release()
    return frames

def process_videos(video_dir, output_file, num_frames=16, step_size=8):
    data = []
    labels = []

    for class_name in os.listdir(video_dir):
        class_dir = os.path.join(video_dir, class_name)
        if os.path.isdir(class_dir):
            for video_file in os.listdir(class_dir):
                video_path = os.path.join(class_dir, video_file)
                frames = extract_frames(video_path, num_frames, step_size)
                if len(frames) == num_frames:
                    data.append(frames)
                    labels.append(class_name)

    data = np.array(data)
    labels = np.array(labels)

    with h5py.File(output_file, 'w') as hf:
        hf.create_dataset('data', data=data)
        hf.create_dataset('labels', data=labels)

if __name__ == '__main__':
    video_dir = '/path/to/videos'
    output_file = 'sliding_windows.h5'
    process_videos(video_dir, output_file)

"""