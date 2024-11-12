import requests
from PIL import Image
import streamlit as st
from st_img_pastebutton import paste
from io import BytesIO
import base64


def encode_image(file):
    _, encoded = file.split(",", 1)
    binary_data = base64.b64decode(encoded)
    bytes_data = BytesIO(binary_data)
    return bytes_data


if __name__ == "__main__":
    st.set_page_config(page_title="LaTeX-OCR")
    st.title("LaTeX OCR")
    st.markdown(
        "Convert images of equations to corresponding LaTeX code.\n\nThis is based on the `pix2tex` module. Check it out [![github](https://img.shields.io/badge/LaTeX--OCR-visit-a?style=social&logo=github)](https://github.com/lukas-blecher/LaTeX-OCR)"
    )

    source = st.radio(
        "Choose the source of the image",
        options=["Paste", "Upload"],
    )

    image = None

    if source == "Upload":
        uploaded_file = st.file_uploader(
            "Upload an image of an equation",
            type=["png", "jpg"],
        )

        if uploaded_file is not None:
            st.image(Image.open(uploaded_file))
            image = uploaded_file.getvalue()

    if source == "Paste":
        pasted_file = paste("Paste an image of an equation")

        if pasted_file is not None:
            image = encode_image(pasted_file)
            st.image(image)

    if st.button("Convert"):
        if image is not None:
            with st.spinner("Computing"):
                response = requests.post(
                    "http://127.0.0.1:8502/predict/", files={"file": image}
                )
            if response.ok:
                st.session_state.latex_code = response.json()
                
    if 'latex_code' in st.session_state:

        st.subheader("Preview")
        st.markdown(f"$\\displaystyle {st.session_state.latex_code}$")
        
        format_option = st.selectbox(
            "Select format to copy:",
            options=[
                "Raw LaTeX",
                "Inline Math ($....$)",
                "Display Math ($$....$$)"
            ]
        )
        
        formatted_code = st.session_state.latex_code
        if format_option == "Inline Math ($....$)":
            formatted_code = f"${st.session_state.latex_code}$"
        elif format_option == "Display Math ($$....$$)":
            formatted_code = f"$${st.session_state.latex_code}$$"
        
        st.code(formatted_code, language="latex")
                
    elif image is None and st.session_state.get('_button_clicked', False):
        st.error("No image selected")