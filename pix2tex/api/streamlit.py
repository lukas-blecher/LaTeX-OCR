import requests
from PIL import Image
import streamlit

if __name__ == '__main__':
    streamlit.set_page_config(page_title='LaTeX-OCR')
    streamlit.title('LaTeX OCR')
    streamlit.markdown('Convert images of equations to corresponding LaTeX code.\n\nThis is based on the `pix2tex` module. Check it out [![github](https://img.shields.io/badge/LaTeX--OCR-visit-a?style=social&logo=github)](https://github.com/lukas-blecher/LaTeX-OCR)')

    uploaded_file = streamlit.file_uploader(
        'Upload an image an equation',
        type=['png', 'jpg'],
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        streamlit.image(image)
    else:
        streamlit.text('\n')

    if streamlit.button('Convert'):
        if uploaded_file is not None and image is not None:
            with streamlit.spinner('Computing'):
                response = requests.post('http://127.0.0.1:8502/predict/', files={'file': uploaded_file.getvalue()})
            if response.ok:
                latex_code = response.json()
                streamlit.code(latex_code, language='latex')
                streamlit.markdown(f'$\\displaystyle {latex_code}$')
            else:
                streamlit.error(response.text)
        else:
            streamlit.error('Please upload an image.')
