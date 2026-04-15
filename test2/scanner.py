import streamlit as st
import cv2
import tempfile
import os
import numpy as np
from ultralytics import YOLO

# 1. Page Configuration & Professional UI Cleanup
st.set_page_config(page_title="Mask Monitor Pro", page_icon="😷", layout="centered")

# Custom CSS to hide the Streamlit header, footer, and menu
hide_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none;}
    </style>
    """
st.markdown(hide_style, unsafe_allow_html=True)

# 2. Load the Custom Model
# Ensure 'best.pt' is in the same folder as this script
model = YOLO('best (1).pt')

st.title("🛡️ Real-Time Mask Monitoring")

# Sidebar settings for user control
st.sidebar.header("Settings")
conf_threshold = st.sidebar.slider("Detection Confidence", 0.0, 1.0, 0.5, 0.05)
st.sidebar.divider()
st.sidebar.info("System tracking for:\n- with_mask\n- mask_weared_incorrect\n- without_mask")

# 3. Input Selection
input_type = st.radio("Select Input Source:", ("Image", "Video", "Webcam"))

uploaded_file = None
if input_type in ["Image", "Video"]:
    uploaded_file = st.file_uploader(f"Upload {input_type}", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"])

# ==========================
#      IMAGE PROCESSING
# ==========================
if input_type == "Image" and uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    results = model.predict(source=img, conf=conf_threshold)
    
    for r in results:
        frame_rgb = cv2.cvtColor(r.plot(), cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, channels="RGB", use_container_width=True)
        
        # Static counting for images
        counts = {name: 0 for name in model.names.values()}
        if r.boxes.cls is not None:
            for cls_id in r.boxes.cls.int().cpu().tolist():
                counts[model.names[cls_id]] += 1
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Safe", counts.get('with_mask', 0))
        c2.metric("Incorrect", counts.get('mask_weared_incorrect', 0))
        c3.metric("Danger", counts.get('without_mask', 0))

# ==========================
#   VIDEO & WEBCAM LOGIC
# ==========================
elif (input_type == "Video" and uploaded_file) or (input_type == "Webcam"):
    
    start_processing = False
    source = None
    
    if input_type == "Video":
        start_time = st.number_input("Start at (seconds):", min_value=0.0, value=0.0)
        if st.button("Analyze Video"):
            # Create a named temp file with a suffix for Windows compatibility
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                tfile.write(uploaded_file.read())
                source = tfile.name
            start_processing = True
    else:
        # Webcam Logic toggle
        run_webcam = st.toggle("Start Webcam")
        if run_webcam:
            source = 0 
            start_processing = True

    if start_processing:
        # Create UI Placeholders for live updates
        st_frame = st.empty()
        st.write("---")
        
        # Fixed layout for the live counters
        c1, c2, c3 = st.columns(3)
        count_safe = c1.empty()
        count_warn = c2.empty()
        count_danger = c3.empty()

        # Initialize the voting system to handle "flickering" labels
        # Structure: { track_id: { 'with_mask': 10, 'without_mask': 2 } }
        id_votes = {}

        cap = cv2.VideoCapture(source)
        
        if input_type == "Video" and source:
            cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)

        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                # Track objects across frames
                results = model.track(frame, persist=True, conf=conf_threshold)

                for r in results:
                    if r.boxes.id is not None:
                        ids = r.boxes.id.int().cpu().tolist()
                        clss = r.boxes.cls.int().cpu().tolist()
                        
                        for track_id, cls_id in zip(ids, clss):
                            label = model.names[cls_id]
                            
                            # Initialize vote counter for new track_id
                            if track_id not in id_votes:
                                id_votes[track_id] = {name: 0 for name in model.names.values()}
                            
                            # Add a vote for the current detected class
                            id_votes[track_id][label] += 1

                    # Calculate live tallies based on the "Winner" (Majority Vote) for each ID
                    final_tallies = {name: 0 for name in model.names.values()}
                    for t_id, votes in id_votes.items():
                        winner = max(votes, key=votes.get)
                        final_tallies[winner] += 1

                    # Update live Video Frame
                    frame_rgb = cv2.cvtColor(r.plot(), cv2.COLOR_BGR2RGB)
                    st_frame.image(frame_rgb, channels="RGB", use_container_width=True)

                    # Update live Counters Underneath
                    count_safe.metric("Live Safe", final_tallies.get('with_mask', 0))
                    count_warn.metric("Live Incorrect", final_tallies.get('mask_weared_incorrect', 0))
                    count_danger.metric("Live No Mask", final_tallies.get('without_mask', 0))

                # Stop loop if webcam toggle is turned off
                if input_type == "Webcam" and not run_webcam:
                    break

        finally:
            cap.release()
            if input_type == "Video" and source and os.path.exists(source):
                os.remove(source)