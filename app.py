import time
import tempfile
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st
from ultralytics import YOLO


st.set_page_config(
    page_title="Football Vision Studio",
    layout="wide",
    initial_sidebar_state="expanded"
)


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


CUSTOM_CSS = """
<style>
    :root {
        --bg: #060914;
        --panel: rgba(13, 18, 31, 0.92);
        --panel-soft: rgba(17, 24, 39, 0.72);
        --line: rgba(148, 163, 184, 0.16);
        --text: #f8fafc;
        --muted: #94a3b8;
        --green: #22c55e;
        --blue: #3b82f6;
        --amber: #f59e0b;
    }

    .stApp {
        background:
            radial-gradient(circle at 12% 8%, rgba(34, 197, 94, 0.10), transparent 28%),
            radial-gradient(circle at 88% 12%, rgba(59, 130, 246, 0.12), transparent 26%),
            linear-gradient(180deg, #060914 0%, #020617 100%);
        color: var(--text);
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    div[data-testid="stToolbar"] {
        visibility: hidden;
        height: 0%;
        position: fixed;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    div[data-testid="stSidebar"] {
        background: rgba(2, 6, 23, 0.96);
        border-right: 1px solid rgba(148, 163, 184, 0.16);
    }

    div[data-testid="stSidebar"] h1,
    div[data-testid="stSidebar"] h2,
    div[data-testid="stSidebar"] h3 {
        color: #f8fafc;
    }

    .topbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 16px;
        margin-bottom: 18px;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 14px;
    }

    .brand-mark {
        width: 42px;
        height: 42px;
        border-radius: 14px;
        background:
            linear-gradient(135deg, rgba(34, 197, 94, 0.95), rgba(59, 130, 246, 0.95));
        box-shadow: 0 12px 34px rgba(34, 197, 94, 0.18);
    }

    .brand-title {
        font-size: 16px;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #e2e8f0;
    }

    .brand-subtitle {
        font-size: 13px;
        color: var(--muted);
        margin-top: 2px;
    }

    .status-pill {
        padding: 9px 14px;
        border-radius: 999px;
        border: 1px solid rgba(34, 197, 94, 0.24);
        background: rgba(34, 197, 94, 0.08);
        color: #bbf7d0;
        font-size: 13px;
        font-weight: 650;
        white-space: nowrap;
    }

    .hero {
        padding: 34px;
        border-radius: 28px;
        border: 1px solid var(--line);
        background:
            linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(2, 6, 23, 0.90)),
            radial-gradient(circle at top right, rgba(34, 197, 94, 0.10), transparent 30%);
        box-shadow: 0 24px 90px rgba(0, 0, 0, 0.38);
        margin-bottom: 22px;
    }

    .eyebrow {
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: #86efac;
        margin-bottom: 12px;
    }

    .hero h1 {
        font-size: 48px;
        line-height: 1.02;
        letter-spacing: -0.055em;
        color: #ffffff;
        margin: 0 0 14px 0;
    }

    .hero p {
        color: #cbd5e1;
        font-size: 17px;
        line-height: 1.65;
        max-width: 900px;
        margin: 0;
    }

    .hero-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 14px;
        margin-top: 26px;
    }

    .mini-card {
        padding: 16px;
        border-radius: 18px;
        background: rgba(15, 23, 42, 0.74);
        border: 1px solid rgba(148, 163, 184, 0.13);
    }

    .mini-card-title {
        color: #94a3b8;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.10em;
        font-weight: 750;
        margin-bottom: 8px;
    }

    .mini-card-value {
        color: #f8fafc;
        font-size: 18px;
        font-weight: 800;
    }

    .panel {
        padding: 22px;
        border-radius: 24px;
        border: 1px solid var(--line);
        background: var(--panel);
        box-shadow: 0 18px 60px rgba(0, 0, 0, 0.22);
        margin-bottom: 18px;
    }

    .panel h2,
    .panel h3 {
        color: #f8fafc;
        margin-top: 0;
    }

    .panel-heading {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
        margin-bottom: 14px;
    }

    .panel-title {
        font-size: 20px;
        font-weight: 800;
        color: #f8fafc;
        letter-spacing: -0.02em;
    }

    .panel-caption {
        color: var(--muted);
        font-size: 13px;
        line-height: 1.55;
    }

    .metric-tile {
        padding: 18px;
        border-radius: 18px;
        background: rgba(2, 6, 23, 0.46);
        border: 1px solid rgba(148, 163, 184, 0.12);
    }

    .metric-number {
        font-size: 30px;
        line-height: 1;
        font-weight: 850;
        color: #ffffff;
        letter-spacing: -0.04em;
    }

    .metric-label {
        margin-top: 8px;
        color: #94a3b8;
        font-size: 13px;
        font-weight: 600;
    }

    .hint-box {
        padding: 15px 16px;
        border-radius: 16px;
        border: 1px solid rgba(59, 130, 246, 0.20);
        background: rgba(59, 130, 246, 0.07);
        color: #bfdbfe;
        font-size: 13px;
        line-height: 1.6;
    }

    .success-box {
        padding: 15px 16px;
        border-radius: 16px;
        border: 1px solid rgba(34, 197, 94, 0.22);
        background: rgba(34, 197, 94, 0.08);
        color: #bbf7d0;
        font-size: 13px;
        line-height: 1.6;
    }

    .stButton > button {
        height: 46px;
        border-radius: 14px;
        border: 1px solid rgba(34, 197, 94, 0.34);
        background: linear-gradient(135deg, #16a34a, #2563eb);
        color: white;
        font-weight: 800;
        letter-spacing: -0.01em;
        box-shadow: 0 14px 34px rgba(37, 99, 235, 0.18);
    }

    .stButton > button:hover {
        border-color: rgba(255, 255, 255, 0.70);
        color: white;
        transform: translateY(-1px);
    }

    .stDownloadButton > button {
        height: 44px;
        border-radius: 14px;
        border: 1px solid rgba(148, 163, 184, 0.22);
        background: rgba(15, 23, 42, 0.92);
        color: #f8fafc;
        font-weight: 750;
    }

    .stDownloadButton > button:hover {
        border-color: rgba(34, 197, 94, 0.45);
        color: #ffffff;
    }

    div[data-testid="stFileUploader"] section {
        border-radius: 18px;
        border: 1px dashed rgba(148, 163, 184, 0.30);
        background: rgba(2, 6, 23, 0.30);
    }

    div[data-testid="stFileUploader"] section:hover {
        border-color: rgba(34, 197, 94, 0.45);
    }

    .footer-note {
        color: #64748b;
        font-size: 12px;
        text-align: center;
        margin-top: 24px;
    }

    @media (max-width: 900px) {
        .hero h1 {
            font-size: 36px;
        }

        .hero-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }

        .topbar {
            flex-direction: column;
            align-items: flex-start;
        }
    }
</style>
"""


st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_model(model_path: str):
    return YOLO(model_path)


def save_uploaded_video(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.read())
        return temp_file.name


def read_video_metadata(video_path: str):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frames / fps if fps and fps > 0 else 0

    cap.release()

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frames": frames,
        "duration": duration
    }


def build_metric_tile(label: str, value: str):
    st.markdown(
        f"""
        <div class="metric-tile">
            <div class="metric-number">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def process_video(
    model,
    input_path: str,
    output_path: Path,
    confidence: float,
    iou_threshold: float,
    use_tracking: bool,
    class_filter_enabled: bool,
    preview_every: int,
    collect_table: bool
):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise RuntimeError("The uploaded video could not be opened.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not fps or fps <= 0:
        fps = 25

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    progress = st.progress(0)
    status = st.empty()
    live_preview = st.empty()

    rows = []
    frame_index = 0
    total_detections = 0
    unique_tracks = set()
    started_at = time.time()

    selected_classes = [0] if class_filter_enabled else None

    while True:
        success, frame = cap.read()

        if not success:
            break

        if use_tracking:
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

        boxes = result.boxes
        frame_detection_count = 0

        if boxes is not None and len(boxes) > 0:
            frame_detection_count = len(boxes)
            total_detections += frame_detection_count

            track_ids = [None] * frame_detection_count

            if boxes.id is not None:
                track_ids = boxes.id.cpu().numpy().astype(int).tolist()
                unique_tracks.update(track_ids)

            if collect_table:
                for index, box in enumerate(boxes):
                    xyxy = box.xyxy[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    score = float(box.conf[0].cpu().numpy())

                    rows.append({
                        "frame": frame_index,
                        "track_id": track_ids[index],
                        "class_id": class_id,
                        "confidence": round(score, 4),
                        "x1": round(float(xyxy[0]), 2),
                        "y1": round(float(xyxy[1]), 2),
                        "x2": round(float(xyxy[2]), 2),
                        "y2": round(float(xyxy[3]), 2)
                    })

        frame_index += 1

        if total_frames > 0:
            progress.progress(min(frame_index / total_frames, 1.0))

        if frame_index % preview_every == 0:
            elapsed = time.time() - started_at
            fps_processing = frame_index / elapsed if elapsed > 0 else 0

            status.markdown(
                f"""
                <div class="success-box">
                    Processing frame {frame_index} of {total_frames}. 
                    Current speed: {fps_processing:.2f} FPS. 
                    Detections so far: {total_detections}.
                </div>
                """,
                unsafe_allow_html=True
            )

            preview_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            live_preview.image(
                preview_frame,
                caption="Live processing preview",
                use_container_width=True
            )

    cap.release()
    writer.release()

    total_time = time.time() - started_at
    progress.progress(1.0)

    status.markdown(
        f"""
        <div class="success-box">
            Processing complete. {frame_index} frames analyzed in {total_time:.1f} seconds.
        </div>
        """,
        unsafe_allow_html=True
    )

    return {
        "frames_processed": frame_index,
        "processing_time": total_time,
        "total_detections": total_detections,
        "unique_tracks": len(unique_tracks),
        "rows": rows
    }


with st.sidebar:
    st.markdown("## Control Room")
    st.caption("Configure the model, detection thresholds, and tracking options.")

    model_mode = st.radio(
        "Model source",
        ["Pretrained YOLOv8 Large", "Custom model path"],
        index=0
    )

    if model_mode == "Pretrained YOLOv8 Large":
        model_path = "yolov8l.pt"
    else:
        model_path = st.text_input("Model path", value="models/best.pt")

    st.divider()

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

    st.divider()

    use_tracking = st.toggle("Player tracking", value=True)

    class_filter_enabled = st.checkbox(
        "Use class 0 only",
        value=True,
        help="For pretrained YOLO, class 0 is person. For a custom dataset, class 0 is usually player."
    )

    collect_table = st.checkbox("Create detection table", value=True)

    preview_every = st.select_slider(
        "Preview refresh rate",
        options=[5, 10, 15, 20, 30],
        value=15
    )

    st.divider()

    st.markdown(
        """
        <div class="hint-box">
            For best results, use a football-specific trained model. 
            The pretrained model works well for player detection because it can detect persons.
        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown(
    """
    <div class="topbar">
        <div class="brand">
            <div class="brand-mark"></div>
            <div>
                <div class="brand-title">Football Vision Studio</div>
                <div class="brand-subtitle">YOLO detection and tracking workspace</div>
            </div>
        </div>
        <div class="status-pill">Local inference dashboard</div>
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Sports computer vision</div>
        <h1>Analyze football footage with player detection and tracking.</h1>
        <p>
            Upload a match clip, run YOLO-based detection, inspect frame-level results, 
            export tracking data, and download the annotated video output.
        </p>
        <div class="hero-grid">
            <div class="mini-card">
                <div class="mini-card-title">Detection</div>
                <div class="mini-card-value">YOLOv8</div>
            </div>
            <div class="mini-card">
                <div class="mini-card-title">Tracking</div>
                <div class="mini-card-value">ByteTrack</div>
            </div>
            <div class="mini-card">
                <div class="mini-card-title">Export</div>
                <div class="mini-card-value">Video and CSV</div>
            </div>
            <div class="mini-card">
                <div class="mini-card-title">Mode</div>
                <div class="mini-card-value">Local Processing</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


left_col, right_col = st.columns([1.05, 0.95], gap="large")

with left_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="panel-heading">
            <div>
                <div class="panel-title">Input Video</div>
                <div class="panel-caption">Upload a football clip to start detection.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_video = st.file_uploader(
        "Upload video",
        type=["mp4", "avi", "mov", "mkv"],
        label_visibility="collapsed"
    )

    input_video_path = None
    run_detection = False

    if uploaded_video is not None:
        input_video_path = save_uploaded_video(uploaded_video)
        video_meta = read_video_metadata(input_video_path)

        if video_meta:
            m1, m2, m3 = st.columns(3)

            with m1:
                build_metric_tile("Frames", f"{video_meta['frames']}")

            with m2:
                build_metric_tile("FPS", f"{video_meta['fps']:.1f}")

            with m3:
                build_metric_tile("Duration", f"{video_meta['duration']:.1f}s")

            st.markdown("<br>", unsafe_allow_html=True)
            st.video(input_video_path)

        run_detection = st.button("Run analysis", use_container_width=True)

    else:
        st.markdown(
            """
            <div class="hint-box">
                Add a video file to begin. Short clips are recommended while testing locally.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)


with right_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="panel-heading">
            <div>
                <div class="panel-title">Session Setup</div>
                <div class="panel-caption">Current configuration selected from the control room.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:
        build_metric_tile("Confidence", f"{confidence:.2f}")

    with c2:
        build_metric_tile("IOU", f"{iou_threshold:.2f}")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="hint-box">
            Model: {model_path}<br>
            Tracking: {"Enabled" if use_tracking else "Disabled"}<br>
            Class filter: {"Class 0 only" if class_filter_enabled else "All detected classes"}<br>
            Detection table: {"Enabled" if collect_table else "Disabled"}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="panel-title">Output Preview</div>
        <div class="panel-caption">
            Processed video, summary metrics, and CSV export will appear after analysis.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('</div>', unsafe_allow_html=True)


if run_detection and input_video_path:
    if model_mode == "Custom model path" and not Path(model_path).exists():
        st.error(f"Custom model not found: {model_path}")
        st.stop()

    output_path = OUTPUT_DIR / f"{Path(uploaded_video.name).stem}_football_analysis.mp4"

    try:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Processing</div>', unsafe_allow_html=True)

        with st.spinner("Loading detection model"):
            model = load_model(model_path)

        summary = process_video(
            model=model,
            input_path=input_video_path,
            output_path=output_path,
            confidence=confidence,
            iou_threshold=iou_threshold,
            use_tracking=use_tracking,
            class_filter_enabled=class_filter_enabled,
            preview_every=preview_every,
            collect_table=collect_table
        )

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Analysis Summary</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        s1, s2, s3, s4 = st.columns(4)

        with s1:
            build_metric_tile("Frames", f"{summary['frames_processed']}")

        with s2:
            build_metric_tile("Time", f"{summary['processing_time']:.1f}s")

        with s3:
            build_metric_tile("Detections", f"{summary['total_detections']}")

        with s4:
            build_metric_tile("Tracks", f"{summary['unique_tracks']}")

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Processed Video</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        st.video(str(output_path))

        with open(output_path, "rb") as video_file:
            st.download_button(
                label="Download processed video",
                data=video_file,
                file_name=output_path.name,
                mime="video/mp4",
                use_container_width=True
            )

        st.markdown('</div>', unsafe_allow_html=True)

        if collect_table and len(summary["rows"]) > 0:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown('<div class="panel-title">Detection Data</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            detections_df = pd.DataFrame(summary["rows"])
            st.dataframe(detections_df, use_container_width=True, hide_index=True)

            csv_data = detections_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download detection CSV",
                data=csv_data,
                file_name="football_detection_results.csv",
                mime="text/csv",
                use_container_width=True
            )

            st.markdown('</div>', unsafe_allow_html=True)

    except Exception as error:
        st.error(f"Processing failed: {error}")


st.markdown(
    """
    <div class="footer-note">
        Football Vision Studio runs locally and keeps uploaded videos on your machine during processing.
    </div>
    """,
    unsafe_allow_html=True
)