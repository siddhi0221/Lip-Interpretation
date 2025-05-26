import tensorflow as tf
from typing import List
import cv2
import os 


vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

def load_video(path:str) -> List[float]:
    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    
    # Read all frames first
    for _ in range(total_frames):
        ret, frame = cap.read()
        if ret:  # Only process if frame was read successfully
            frame = tf.image.rgb_to_grayscale(frame)
            frames.append(frame[190:236,80:220,:])
    cap.release()

    # Select exactly 75 frames uniformly spaced from the original
    if len(frames) > 75:
        step = len(frames) / 75
        indices = [int(i * step) for i in range(75)]
        frames = [frames[i] for i in indices if i < len(frames)]
    elif len(frames) < 75:
        # Pad with last frame if video is too short (unlikely case)
        last_frame = frames[-1]
        frames += [last_frame] * (75 - len(frames))
    
    # Normalize as before
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, dtype=tf.float32))
    return tf.cast((frames-mean), dtype=tf.float32) / std

def load_alignments(path:str) -> List[str]:
    with open(path, 'r') as f:
        lines = f.readlines()
    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != 'sil':
            tokens = [*tokens,' ',line[2]]
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]

def load_data(path: str):
    path = bytes.decode(path.numpy())
    #file_name = path.split('/')[-1].split('.')[0]
    # File name splitting for windows
    file_name = path.split('\\')[-1].split('.')[0]
    video_path = os.path.join('data','s1',f'{file_name}.mpg')
    alignment_path = os.path.join('data','alignments','s1',f'{file_name}.align')
    frames = load_video(video_path)
    alignments = load_alignments(alignment_path)

    return frames, alignments