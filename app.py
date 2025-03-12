import streamlit as st
from PIL import Image
import numpy as np
import io
from model import run

def preprocess_image(image):
    """Resizes and normalizes images for the model."""
    image = image.resize((400, 400))
    image = np.array(image) / 255.0
    return image

st.title("🎨 Neural Style Transfer")

st.sidebar.markdown("""
## How to Use:
1. Upload a **Content Image** (e.g., your photo).
2. Upload a **Style Image** (e.g., a painting).
3. Choose the **Number of Training Epochs**.
4. Click **Generate** to stylize the image.
5. Download the generated image.
6. **Note**: To generate another image, please download the previous one first.
""")

st.sidebar.header("Upload Images")

if "content_image" not in st.session_state:
    st.session_state.content_image = None
if "style_image" not in st.session_state:
    st.session_state.style_image = None
if "generated_image" not in st.session_state:
    st.session_state.generated_image = None


content_image_file = st.sidebar.file_uploader("Choose a content image...", type=["jpg", "jpeg", "png"], key="content_uploader")
style_image_file = st.sidebar.file_uploader("Choose a style image...", type=["jpg", "jpeg", "png"], key="style_uploader")

def display_image(image_file, caption, key):
    """Displays an image and updates session state."""
    if image_file:
        image = Image.open(image_file)
        st.sidebar.image(image, caption=caption, use_container_width=True)  
        st.session_state[key] = image  
        return image
    return None



if content_image_file:
    st.session_state.content_image = display_image(content_image_file, "Content Image", "content_image")
if style_image_file:
    st.session_state.style_image = display_image(style_image_file, "Style Image", "style_image")

if content_image_file or style_image_file:
    st.session_state.generated_image = None

epochs = st.sidebar.number_input("Epochs:", min_value=1, max_value=1000, value=10, step=1)

if st.sidebar.button("Generate"):
    if st.session_state.content_image and st.session_state.style_image:
        with st.spinner("Generating Art... 🎨"):
            try:
                content_image = preprocess_image(st.session_state.content_image)
                style_image = preprocess_image(st.session_state.style_image)

                st.session_state.generated_image = run(content_image, style_image, epochs)

            except Exception as e:
                st.error(f"Error: {e}")

if st.session_state.generated_image:
    st.image(st.session_state.generated_image, caption="Stylized Image", use_container_width=True)


    buf = io.BytesIO()
    st.session_state.generated_image.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(label="Download Image 🎉", data=byte_im, file_name="styled_image.png", mime="image/png")
