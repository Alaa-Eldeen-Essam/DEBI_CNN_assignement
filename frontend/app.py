import io
import requests
import streamlit as st
from PIL import Image

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Face Mask Detector",
    page_icon="😷",
    layout="centered",
)

# ── Minimal custom CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    .result-box {
        border-radius: 10px;
        padding: 18px 22px;
        margin-top: 10px;
        font-size: 15px;
    }
    .with-mask    { background: #e8f5e9; border-left: 5px solid #43a047; color: #1b5e20; }
    .without-mask { background: #ffebee; border-left: 5px solid #e53935; color: #b71c1c; }
    .model-title  { font-size: 14px; font-weight: 600; color: #555; margin-bottom: 4px; }
    .label-big    { font-size: 22px; font-weight: 700; }
    .conf-text    { font-size: 14px; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar config ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    api_url = st.text_input("API Base URL", value="http://localhost:8000")

    st.markdown("---")
    st.markdown("**Endpoint**")
    st.code(f"{api_url}/predict/mobilenet")

    st.markdown("---")
    # Health check
    if st.button("🔍 Check API Health"):
        try:
            r = requests.get(f"{api_url}/health", timeout=5)
            if r.status_code == 200:
                st.success("API is healthy ✅")
                st.json(r.json())
            else:
                st.error(f"API returned {r.status_code}")
        except Exception as e:
            st.error(f"Cannot reach API:\n{e}")


# ── Main page ──────────────────────────────────────────────────────────────────
st.title("😷 Face Mask Detector")
st.caption("Upload a face image to detect whether the person is wearing a mask.")

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded:
    col_img, col_res = st.columns([1, 1.2], gap="large")

    with col_img:
        st.subheader("Image Preview")
        image = Image.open(uploaded).convert("RGB")
        st.image(image, use_column_width=True)
        st.caption(f"{uploaded.name}  ·  {image.size[0]}×{image.size[1]} px")

    with col_res:
        st.subheader("Prediction")

        # Build the API request
        uploaded.seek(0)
        files = {"file": (uploaded.name, uploaded.read(), uploaded.type or "image/jpeg")}

        endpoint = f"{api_url}/predict/mobilenet"

        with st.spinner("Running inference…"):
            try:
                response = requests.post(endpoint, files=files, timeout=30)
                response.raise_for_status()
                data = response.json()
            except requests.exceptions.ConnectionError:
                st.error("❌ Could not connect to the API. Is the backend running?")
                st.stop()
            except requests.exceptions.HTTPError as e:
                st.error(f"❌ API error: {e}")
                st.stop()
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")
                st.stop()

        # ── Render results ─────────────────────────────────────────────────────
        def render_result(result: dict):
            label   = result["label"]
            conf    = result["confidence"]
            model   = result["model"]
            probs   = result["probability"]
            css_cls = "with-mask" if label == "With Mask" else "without-mask"
            icon    = "✅" if label == "With Mask" else "🚫"

            st.markdown(f"""
            <div class="result-box {css_cls}">
                <div class="model-title">{model}</div>
                <div class="label-big">{icon} {label}</div>
                <div class="conf-text">Confidence: <b>{conf}%</b></div>
            </div>
            """, unsafe_allow_html=True)

            # Probability bars
            st.markdown("**Class probabilities**")
            st.progress(probs["With Mask"] / 100,
                        text=f"With Mask — {probs['With Mask']}%")
            st.progress(probs["Without Mask"] / 100,
                        text=f"Without Mask — {probs['Without Mask']}%")

        render_result(data)

else:
    st.info("👆 Upload an image to get started.")