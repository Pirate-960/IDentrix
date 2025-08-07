# pages/2_🎥_Video_Processor.py
"""
Streamlit page for person tracking in video files.

This advanced feature demonstrates a complete pipeline:
1.  Loads a user-uploaded video.
2.  Uses YOLOv8 for robust person detection in each frame.
3.  Crops each detected person and generates a Re-ID embedding.
4.  Implements a simple tracking algorithm to assign a consistent ID to each
    person across frames based on embedding similarity.
5.  Outputs a new video with tracking IDs and bounding boxes drawn.

NOTE: This is computationally intensive and may be slow.
"""

import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
from collections import defaultdict

# --- Local Imports ---
# Assumes we are running from the root directory
from model import DualBackboneNet
from utils import preprocess_image

# --- Resource Loading (Cached) ---

@st.cache_resource
def load_models():
    """Loads both the Re-ID and YOLOv8 models."""
    # Load Re-ID model
    reid_model_path = "checkpoints/best_model.pth"
    if not os.path.exists(reid_model_path):
        return None, None
    reid_model = DualBackboneNet()
    reid_model.load_state_dict(torch.load(reid_model_path, map_location='cpu', weights_only=True))
    reid_model.eval()
    
    # Load YOLO model
    yolo_model = YOLO('yolov8n.pt')  # 'n' is the smallest, fastest model
    
    return reid_model, yolo_model

# --- Tracking Logic ---
class SimpleTracker:
    """A simple tracker based on embedding similarity."""
    def __init__(self, match_threshold=0.8):
        self.next_track_id = 0
        self.tracks = {}  # {track_id: last_embedding}
        self.match_threshold = match_threshold

    def update(self, detection_embeddings):
        matched_ids = []
        unmatched_detections = list(range(len(detection_embeddings)))

        if not self.tracks: # First frame
            for emb in detection_embeddings:
                self.tracks[self.next_track_id] = emb
                matched_ids.append(self.next_track_id)
                self.next_track_id += 1
            return matched_ids

        # Create similarity matrix
        track_embs = np.array(list(self.tracks.values()))
        sim_matrix = np.dot(detection_embeddings, track_embs.T)

        # Match detections to existing tracks
        for i, det_emb in enumerate(detection_embeddings):
            if sim_matrix.size > 0:
                best_match_idx = np.argmax(sim_matrix[i])
                if sim_matrix[i, best_match_idx] > self.match_threshold:
                    track_id = list(self.tracks.keys())[best_match_idx]
                    matched_ids.append(track_id)
                    # Update track with new embedding for smoothness
                    self.tracks[track_id] = (self.tracks[track_id] + det_emb) / 2.0
                    if i in unmatched_detections:
                        unmatched_detections.remove(i)
                    # Prevent this track from being matched again
                    sim_matrix[:, best_match_idx] = -1 
                else:
                    matched_ids.append(-1) # No match
            else:
                 matched_ids.append(-1) # No match

        # Handle unmatched detections as new tracks
        for idx in unmatched_detections:
            new_id = self.next_track_id
            self.tracks[new_id] = detection_embeddings[idx]
            matched_ids[idx] = new_id
            self.next_track_id += 1

        return matched_ids

# --- Main Page UI and Logic ---

st.set_page_config(page_title="Video Processor", layout="wide")
st.title("🎥 Video-Based Person Tracking")
st.warning("This is an experimental feature. Processing can be slow and requires significant CPU/GPU resources.")

reid_model, yolo_model = load_models()

if not reid_model or not yolo_model:
    st.error("Models could not be loaded. Ensure `checkpoints/best_model.pth` exists.")
    st.stop()

uploaded_file = st.file_uploader("Upload a short video file (.mp4, .mov)", type=['mp4', 'mov'])

if uploaded_file:
    # Save the uploaded file to a temporary path for OpenCV
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_video_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.video(temp_video_path)
    
    if st.button("🚀 Process Video to Track People"):
        tracker = SimpleTracker()
        
        cap = cv2.VideoCapture(temp_video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Define the codec and create VideoWriter object
        output_path = os.path.join(temp_dir, f"tracked_{uploaded_file.name}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        progress_bar = st.progress(0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_count = 0
        with st.spinner("Processing video frame by frame..."):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 1. Person Detection with YOLO
                # We only care about class 0, which is 'person' in COCO dataset
                results = yolo_model(frame, classes=[0], verbose=False)
                
                detections = results[0].boxes.xyxy.cpu().numpy()
                detection_embeddings = []
                
                if len(detections) > 0:
                    # 2. Get Re-ID Embedding for each detection
                    for x1, y1, x2, y2 in detections:
                        crop = frame[int(y1):int(y2), int(x1):int(x2)]
                        pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        with torch.no_grad():
                            emb = reid_model(preprocess_image(pil_crop)).detach().cpu().numpy()[0]
                        detection_embeddings.append(emb / np.linalg.norm(emb)) # Normalize

                # 3. Update Tracker
                track_ids = tracker.update(np.array(detection_embeddings))

                # 4. Draw results on frame
                for i, (x1, y1, x2, y2) in enumerate(detections):
                    track_id = track_ids[i]
                    color = (int(track_id) * 30 % 255, int(track_id) * 50 % 255, int(track_id) * 70 % 255)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(frame, f"ID: {track_id}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                out.write(frame)
                frame_count += 1
                progress_bar.progress(frame_count / total_frames)

        cap.release()
        out.release()
        
        st.success("Video processing complete!")
        st.video(output_path)