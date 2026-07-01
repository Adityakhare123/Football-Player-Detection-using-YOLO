import os
import cv2
import time
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from ultralytics import YOLO


st.set_page_config(
    page_title="Football Player Detection System",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)


CUSTOM_CSS = """
<style>
    .main {
        background: #020617;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(34,197,94,0.16), transparent 35%),
            radial-gradient(circle at top right, rgba(59,130,246,0.14), transparent 35%),
            #020617;
        color: #e5e7eb;
    }

    .hero-card {
        padding: 32px;
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(15,23,42,0.96), rgba(2,6,23,0.96));
        border: 1px solid rgba(148,163,184,0.22);
        box-shadow: 0 24px 70px rgba(0,0,0,0.45);
        margin-bottom: 24px;
    }

    .hero-title {
        font-size: 46px;
        font-weight: 800;
        letter-spacing: -1.4px;
        margin-bottom: 12px;
        color: #ffffff;
    }

    .hero-subtitle {
        font-size: 18px;
        color: #cbd5e1;
        max-width: 900px;
        line-height: 1.6;
    }

    .metric-card {
        padding: 20px;
        border-radius: 18px;
        background: rgba(15,23,42,0.88);
        border: 1px solid rgba(148,163,184,0.18);
        text-align: center;
    }

    .metric-value {
        font-size: 30px;
        font-weight: 800;
        color: #22c55e;
    }

    .metric-label {
        font-size: 14px;
        color: #94a3b8;
        margin-top: 4px;
    }

    .section-card {
        padding: 24px;
        border-radius: 20px;
        background: rgba(15,23,42,0.86);
        border: 1px solid rgba(148,163,184,0.18);
        margin-bottom: 20px;
    }

    .stButton > button {
        width: 100%;
        border-radius: 14px;
        border: 1px solid rgba(34,197,94,0.6);
        background: linear-gradient(135deg, #16a34a, #2563eb);
        color: white;
        font-weight: 700;
        padding: 0.8rem 1rem;
    }

    .stButton > button:hover {
        border: 1px solid rgba(255,255,255,0.8);
        color: white;
    }

    div[data-testid="stSidebar"] {
        background: rgba(2,6,23,0.96);
        border-right: 1px solid rgba(148,163,184,0.18);
    }

    .small-note {
        color: #94a3b8;
        font-size: 14px;
        line-height: 1.6;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


@st.cache_resource
def load_yolo_model(model_path: str):
    return YOLO(model_path)


def save_uploaded_file(uploaded_file):
    suffix = Path(uploaded_file.name).suffix

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.read())
        return temp_file.name


def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps and fps > 0 else 0

    cap.release()

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration": duration
    }


def process_video(
    model,
    input_path,
    output_path,
    confidence,
    iou_threshold,
    enable_tracking,
    class_zero_only,
    show_detection_table
):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise RuntimeError("Could not open the uploaded video file.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 25

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    progress_bar = st.progress(0)
    status_text = st.empty()
    preview_placeholder = st.empty()

    detection_rows = []
    frame_index = 0
    start_time = time.time()

    selected_classes = [0] if class_zero_only else None

    while True:
        success, frame = cap.read()

        if not success:
            break

        if enable_tracking:
            results = model.track(
                frame,
                persist=True,
                conf=confidence,
                iou=iou_threshold,
                classes=selected_classes,
                tracker="bytetrack.yaml",
                verbose=False
            )
        else:
            results = model.predict(
                frame,
                conf=confidence,
                iou=iou_threshold,
                classes=selected_classes,
                verbose=False
            )

        result = results[0]
        annotated_frame = result.plot()
        writer.write(annotated_frame)

        if show_detection_table and result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes

            if boxes.id is not None:
                track_ids = boxes.id.cpu().numpy().astype(int).tolist()
            else:
                track_ids = [None] * len(boxes)

            for box_index, box in enumerate(boxes):
                xyxy = box.xyxy[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                score = float(box.conf[0].cpu().numpy())

                detection_rows.append({
                    "frame": frame_index,
                    "track_id": track_ids[box_index],
                    "class_id": class_id,
                    "confidence": round(score, 4),
                    "x1": round(float(xyxy[0]), 2),
                    "y1": round(float(xyxy[1]), 2),
                    "x2": round(float(xyxy[2]), 2),
                    "y2": round(float(xyxy[3]), 2)
                })

        frame_index += 1

        if total_frames > 0:
            progress = min(frame_index / total_frames, 1.0)
            progress_bar.progress(progress)

        if frame_index % 15 == 0:
            elapsed = time.time() - start_time
            status_text.markdown(
                f"<p class='small-note'>Processing frame {frame_index} of {total_frames} | Elapsed time: {elapsed:.1f}s</p>",
                unsafe_allow_html=True
            )

            preview_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            preview_placeholder.image(preview_rgb, caption="Live detection preview", use_container_width=True)

    cap.release()
    writer.release()

    total_time = time.time() - start_time

    progress_bar.progress(1.0)
    status_text.markdown(
        f"<p class='small-note'>Processing complete in {total_time:.1f} seconds.</p>",
        unsafe_allow_html=True
    )

    return {
        "frames_processed": frame_index,
        "processing_time": total_time,
        "detections": detection_rows
    }


st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">Football Player Detection System</div>
        <div class="hero-subtitle">
            Upload a football match video, run YOLO-based player detection, enable tracking,
            preview the annotated output, and download the processed video.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


with st.sidebar:
    st.header("Model Settings")

    model_choice = st.radio(
        "Select model",
        ["Pretrained YOLOv8 Large", "Custom model path"],
        index=0
    )

    if model_choice == "Pretrained YOLOv8 Large":
        model_path = "yolov8l.pt"
    else:
        model_path = st.text_input("Custom model path", value="models/best.pt")

    confidence = st.slider(
        "Confidence threshold",
        min_value=0.10,
        max_value=0.95,
        value=0.35,
        step=0.05
    )

    iou_threshold = st.slider(
        "IOU threshold",
        min_value=0.10,
        max_value=0.95,
        value=0.45,
        step=0.05
    )

    enable_tracking = st.toggle("Enable player tracking", value=True)

    class_zero_only = st.checkbox(
        "Detect class 0 only",
        value=True,
        help="For COCO YOLO, class 0 is person. For custom football dataset, class 0 is usually player."
    )

    show_detection_table = st.checkbox(
        "Generate detection table",
        value=True
    )

    st.markdown("---")
    st.markdown(
        """
        <p class="small-note">
        Use a custom trained model when you have a football-specific best.pt file.
        Keep large .pt files outside GitHub or upload them through releases/cloud storage.
        </p>
        """,
        unsafe_allow_html=True
    )


col1, col2 = st.columns([1.05, 0.95], gap="large")

with col1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Upload Football Video")

    uploaded_video = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_video is not None:
        input_video_path = save_uploaded_file(uploaded_video)
        video_info = get_video_info(input_video_path)

        if video_info:
            m1, m2, m3 = st.columns(3)

            with m1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-value">{video_info["frame_count"]}</div>
                        <div class="metric-label">Frames</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with m2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-value">{video_info["fps"]:.1f}</div>
                        <div class="metric-label">FPS</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with m3:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-value">{video_info["duration"]:.1f}s</div>
                        <div class="metric-label">Duration</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.video(input_video_path)

        run_button = st.button("Run Detection")

    else:
        input_video_path = None
        run_button = False
        st.info("Upload a football video to start detection.")

    st.markdown('</div>', unsafe_allow_html=True)


with col2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Output")

    if uploaded_video is None:
        st.markdown(
            """
            <p class="small-note">
            The processed video will appear here after detection is complete.
            </p>
            """,
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)


if run_button and input_video_path:
    if model_choice == "Custom model path" and not Path(model_path).exists():
        st.error(f"Custom model not found: {model_path}")
        st.stop()

    try:
        with st.spinner("Loading YOLO model..."):
            model = load_yolo_model(model_path)

        output_video_path = OUTPUT_DIR / f"{Path(uploaded_video.name).stem}_detected.mp4"

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Processing")

        summary = process_video(
            model=model,
            input_path=input_video_path,
            output_path=output_video_path,
            confidence=confidence,
            iou_threshold=iou_threshold,
            enable_tracking=enable_tracking,
            class_zero_only=class_zero_only,
            show_detection_table=show_detection_table
        )

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Processed Video")

        st.video(str(output_video_path))

        with open(output_video_path, "rb") as file:
            st.download_button(
                label="Download Processed Video",
                data=file,
                file_name=output_video_path.name,
                mime="video/mp4"
            )

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Detection Summary")

        s1, s2, s3 = st.columns(3)

        with s1:
            st.metric("Frames Processed", summary["frames_processed"])

        with s2:
            st.metric("Processing Time", f"{summary['processing_time']:.1f}s")

        with s3:
            st.metric("Detection Rows", len(summary["detections"]))

        if show_detection_table and len(summary["detections"]) > 0:
            detections_df = pd.DataFrame(summary["detections"])
            st.dataframe(detections_df, use_container_width=True)

            csv_data = detections_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Detection CSV",
                data=csv_data,
                file_name="football_detections.csv",
                mime="text/csv"
            )

        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as error:
        st.error(f"Something went wrong: {error}")