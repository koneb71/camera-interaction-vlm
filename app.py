import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
from video_interference import load_model, generate_response
import tempfile
import time
from PIL import Image

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_analyzed = 0
        self.response = ""
        self.model = None
        self.processor = None
        self.device = "cuda" if st.session_state.get("use_cuda", False) else "cpu"
        self.checkpoint_path = st.session_state.get("checkpoint_path", None)
        self.base_model_id = st.session_state.get("base_model_id", "HuggingFaceTB/SmolVLM-Instruct")
        self.question = st.session_state.get("question", "What do you see?")
        self.max_frames = st.session_state.get("max_frames", 50)
        if self.model is None or self.processor is None:
            self.model, self.processor = load_model(self.checkpoint_path, self.base_model_id, self.device)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        now = time.time()
        # Analyze every 0.25s (snapshot mode)
        if now - self.last_analyzed > 1:
            print("Analyzing snapshot image")
            pil_img = Image.fromarray(img[..., ::-1])
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                pil_img.save(tmp_file, format="JPEG")
                image_path = tmp_file.name
            try:
                    
                self.response = generate_response(
                    self.model, self.processor, image_path, self.question, max_frames=1
                )
                print(f"Response: {self.response}")
            except Exception as e:
                print(f"Error: {e}")
                self.response = f"Error: {e}"
            self.last_analyzed = now
        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.set_page_config(page_title="Camera Interaction App", layout="centered")
st.title("Camera Interaction App")

# Sidebar for model config
st.sidebar.header("Model Configuration")
st.session_state["checkpoint_path"] = st.sidebar.text_input("Checkpoint Path (optional)", value="")
st.session_state["base_model_id"] = st.sidebar.text_input("Base Model ID", value="HuggingFaceTB/SmolVLM-Instruct")
st.session_state["max_frames"] = st.sidebar.slider("Max Frames", min_value=1, max_value=100, value=50)
st.session_state["use_cuda"] = st.sidebar.checkbox("Use CUDA (GPU)", value=True)

st.session_state["question"] = st.text_input("Instruction:", value="What do you see?")

ctx = webrtc_streamer(
    key="example",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if ctx.video_processor:
    st.markdown("**Live Response:**")
    st.text_area("", value=ctx.video_processor.response, height=100)