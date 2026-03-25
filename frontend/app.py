import requests
import streamlit as st
from PIL import Image, ImageDraw

IMAGE_TYPES = ["jpg", "jpeg", "png"]
VIDEO_TYPES = ["mp4", "mov", "avi", "mkv"]


def is_video_upload(uploaded_file) -> bool:
    return uploaded_file.name.lower().endswith(tuple(f".{ext}" for ext in VIDEO_TYPES))


# Page config
st.set_page_config(
    page_title="Face Mask Detector",
    layout="centered",
)

# CSS
st.markdown(
    """
<style>
    .result-box     { border-radius:10px; padding:18px 22px; margin-top:10px; font-size:15px; }
    .with-mask      { background:#e8f5e9; border-left:5px solid #43a047; color:#1b5e20; }
    .without-mask   { background:#ffebee; border-left:5px solid #e53935; color:#b71c1c; }
    .model-title    { font-size:14px; font-weight:600; color:#555; margin-bottom:4px; }
    .label-big      { font-size:22px; font-weight:700; }
    .conf-text      { font-size:14px; margin-top:4px; }
    .det-row        { display:flex; justify-content:space-between; padding:4px 0;
                      border-bottom:1px solid #eee; font-size:14px; }
</style>
""",
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.title("Settings")
    api_url = st.text_input("API Base URL", value="http://localhost:8000")

    model_choice = st.radio("Model", ["MobileNetV2", "YOLO"], index=0)

    st.markdown("---")
    st.markdown("**Endpoint**")
    if model_choice == "MobileNetV2":
        endpoint = f"{api_url}/predict/mobilenet"
    else:
        endpoint = f"{api_url}/predict/yolo"
    st.code(endpoint)

    st.markdown("---")
    if st.button("Check API Health"):
        try:
            response = requests.get(f"{api_url}/health", timeout=5)
            if response.status_code == 200:
                st.success("API is healthy.")
                st.json(response.json())
            else:
                st.error(f"API returned {response.status_code}")
        except Exception as exc:
            st.error(f"Cannot reach API:\n{exc}")

    st.markdown("---")
    st.markdown("**Model notes**")
    if model_choice == "MobileNetV2":
        st.info("Classification supports images only.")
    else:
        st.info("YOLO supports both images and videos.")

# Main
st.title("Face Mask Detector")
st.caption("Upload an image or a video and run inference from the sidebar model.")

allowed_types = (
    IMAGE_TYPES if model_choice == "MobileNetV2" else IMAGE_TYPES + VIDEO_TYPES
)
upload_label = (
    "Choose an image" if model_choice == "MobileNetV2" else "Choose an image or video"
)
uploaded = st.file_uploader(upload_label, type=allowed_types)

if uploaded:
    uploaded.seek(0)
    file_bytes = uploaded.read()
    video_upload = is_video_upload(uploaded)

    if model_choice == "MobileNetV2" and video_upload:
        st.error("MobileNetV2 currently supports images only.")
        st.stop()

    if video_upload and model_choice == "YOLO":
        st.subheader("Video Preview")
        st.video(file_bytes)
        st.caption(uploaded.name)

        files = {"file": (uploaded.name, file_bytes, uploaded.type or "video/mp4")}

        with st.spinner("Running YOLO on video..."):
            try:
                response = requests.post(
                    f"{api_url}/predict/yolo-video",
                    files=files,
                    timeout=300,
                )
                response.raise_for_status()
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the API. Is the backend running?")
                st.stop()
            except requests.exceptions.HTTPError as exc:
                st.error(f"API error: {exc}")
                st.stop()
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")
                st.stop()

        st.subheader("Processed Video")
        st.video(response.content)
        st.download_button(
            label="Download annotated video",
            data=response.content,
            file_name=f"annotated_{uploaded.name.rsplit('.', 1)[0]}.mp4",
            mime="video/mp4",
        )

    else:
        col_img, col_res = st.columns([1, 1.2], gap="large")

        with col_img:
            st.subheader("Image Preview")
            image = Image.open(uploaded).convert("RGB")
            img_placeholder = st.empty()
            img_placeholder.image(image, use_column_width=True)
            st.caption(f"{uploaded.name} | {image.size[0]}x{image.size[1]} px")

        with col_res:
            st.subheader("Prediction")

            files = {"file": (uploaded.name, file_bytes, uploaded.type or "image/jpeg")}

            with st.spinner("Running inference..."):
                try:
                    response = requests.post(endpoint, files=files, timeout=30)
                    response.raise_for_status()
                    data = response.json()
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to the API. Is the backend running?")
                    st.stop()
                except requests.exceptions.HTTPError as exc:
                    st.error(f"API error: {exc}")
                    st.stop()
                except Exception as exc:
                    st.error(f"Unexpected error: {exc}")
                    st.stop()

            # MobileNetV2 result
            if model_choice == "MobileNetV2":
                label = data["label"]
                conf = data["confidence"]
                probs = data["probability"]
                css_cls = "with-mask" if label == "With Mask" else "without-mask"
                icon = "[OK]" if label == "With Mask" else "[NO]"

                st.markdown(
                    f"""
                <div class="result-box {css_cls}">
                    <div class="model-title">MobileNetV2</div>
                    <div class="label-big">{icon} {label}</div>
                    <div class="conf-text">Confidence: <b>{conf}%</b></div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                st.markdown("**Class probabilities**")
                st.progress(
                    probs["With Mask"] / 100,
                    text=f"With Mask - {probs['With Mask']}%",
                )
                st.progress(
                    probs["Without Mask"] / 100,
                    text=f"Without Mask - {probs['Without Mask']}%",
                )

            # YOLO image result
            else:
                total = data["total"]
                with_mask = data["with_mask"]
                without_mask = data["without_mask"]
                detections = data["detections"]

                if total == 0:
                    st.warning("No faces detected. Try a clearer or closer image.")
                else:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Faces", total)
                    c2.metric("With Mask", with_mask)
                    c3.metric("Without Mask", without_mask)

                    st.markdown("---")

                    draw = ImageDraw.Draw(image)
                    color_map = {"With Mask": "#43a047", "Without Mask": "#e53935"}

                    for det in detections:
                        x1, y1, x2, y2 = det["box"]
                        color = color_map.get(det["label"], "yellow")
                        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                        draw.rectangle([x1, y1 - 18, x1 + 120, y1], fill=color)
                        draw.text(
                            (x1 + 3, y1 - 16),
                            f"{det['label']} {det['confidence']}%",
                            fill="white",
                        )

                    img_placeholder.image(image, use_column_width=True)

                    st.markdown("**Detections**")
                    for i, det in enumerate(detections):
                        icon = "[OK]" if det["label"] == "With Mask" else "[NO]"
                        st.markdown(
                            f'<div class="det-row">'
                            f'<span>#{i + 1} {icon} {det["label"]}</span>'
                            f'<span><b>{det["confidence"]}%</b></span>'
                            f"</div>",
                            unsafe_allow_html=True,
                        )

else:
    st.info("Upload an image or video to get started.")
