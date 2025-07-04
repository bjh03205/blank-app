import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image

# Page config
st.set_page_config(page_title="Turtle Detection App", layout="centered")

# Dark theme styling
st.markdown(
    """
    <style>
    .stApp { background-color: #111111; color: #FFFFFF; }
    .stButton>button { width: 100%; background-color: #333333; color: #FFFFFF; border: none; }
    </style>
    """, unsafe_allow_html=True
)

# Sidebar: select between two YOLOv11 models
st.sidebar.title("Model Selection")
model_choice = st.sidebar.radio(
    "Choose a detection model:",
    ("YOLOv11 - Invasive Turtle", "YOLOv11 - Red Ear Turtle")
)

# Paths for the two YOLOv11 checkpoints
invasive_ckpt = r'C:/Users/bjh03/Desktop/project/streamlit/invasive_best.pt'
red_ear_ckpt  = r'C:/Users/bjh03/Desktop/project/streamlit/invasive2_best.pt'

# Mapping species names to info URLs
INFO_LINKS = {
    "Chelydra serpentina":   "https://en.wikipedia.org/wiki/Common_snapping_turtle",
    "Macrochelys temminckii": "https://en.wikipedia.org/wiki/Alligator_snapping_turtle",
    "Mauremys sinensis":      "https://en.wikipedia.org/wiki/Chinese_stripe-necked_turtle",
    "Pseudemys concinna":     "https://en.wikipedia.org/wiki/River_cooter",
    "Pseudemys nelsoni":      "https://en.wikipedia.org/wiki/Florida_red-bellied_cooter",
    "Trachemys scripta":      "https://en.wikipedia.org/wiki/Trachemys_scripta_elegans",
}

# Load models with caching
@st.cache_resource
def load_model(path):
    return YOLO(path)

invasive_model = load_model(invasive_ckpt)
red_ear_model  = load_model(red_ear_ckpt)
model = invasive_model if model_choice.startswith("YOLOv11 - Invasive Turtle") else red_ear_model

# Initialize session storage
if 'input_image'   not in st.session_state: st.session_state.input_image   = None
if 'capture_mode'  not in st.session_state: st.session_state.capture_mode  = False
if 'upload_mode'   not in st.session_state: st.session_state.upload_mode   = False

# ── 항상 노출되는 입력 UI ──
st.title("Turtle Detection App")
st.markdown("Capture or upload an image for detection.")

col1, col2 = st.columns([1, 1], gap="small")
with col1:
    if st.button("Capture Photo", key="btn_capture"):
        st.session_state.capture_mode = True
        st.session_state.upload_mode  = False
with col2:
    if st.button("Upload Image", key="btn_upload"):
        st.session_state.upload_mode   = True
        st.session_state.capture_mode  = False

# ── 이미지 획득 ──
if st.session_state.capture_mode:
    cam_input = st.camera_input("Capture your photo", key="cam")
    if cam_input:
        st.session_state.input_image = Image.open(cam_input)

elif st.session_state.upload_mode:
    uploaded = st.file_uploader(
        "Select an image file",
        type=["jpg","jpeg","png"],
        key="up"
    )
    if uploaded:
        st.session_state.input_image = Image.open(uploaded)

# ── 탐지 실행 & 결과 ──
if st.session_state.input_image is not None:
    img_arr   = np.array(st.session_state.input_image.convert('RGB'))
    results   = model(img_arr)
    boxes     = results[0].boxes
    annotated = results[0].plot()

    if boxes:
        # 가장 높은 확신도의 단일 결과 추출
        best     = max(boxes, key=lambda b: b.conf)
        cls_id   = int(best.cls)
        cls_name = results[0].names[cls_id]
        conf     = float(best.conf)

        # 결과 제목
        st.subheader("Detection Results")

        # 탐지된 이미지
        st.image(annotated, caption="Detected Image", use_column_width=True)

        # 단 하나의 결과 값
        st.markdown(f"- **Species:** {cls_name}")
        st.markdown(f"- **Confidence:** {conf:.2f}")

        # 상세정보 버튼
        info_url = INFO_LINKS.get(
            cls_name,
            f"https://en.wikipedia.org/wiki/{cls_name.replace(' ', '_')}"
        )
        st.markdown(
            f'<a href="{info_url}" target="_blank">'
            f'<button style="background-color:#333333;'
            f'color:#FFFFFF;border:none;padding:8px 16px;'
            f'border-radius:4px;">상세정보</button></a>',
            unsafe_allow_html=True
        )
    else:
        st.subheader("Detection Results")
        st.write("No objects detected.")
