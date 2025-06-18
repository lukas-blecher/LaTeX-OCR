import gradio as gr
from pix2tex.cli import LatexOCR
from PIL import Image


custom_css = """
footer {
    display: none !important;
}
"""


model = LatexOCR()



def to_tex(image):
    img=Image.fromarray(image, mode='RGB')
    return model(img)

with gr.Blocks(css=custom_css) as interface:
    gr.Markdown("## Image to Latex")
    
    
    with gr.Column():
        input_component = gr.Image(label="Input Image",placeholder="Drag image or click to upload")
        output_component = gr.Textbox(label="Generated Latex Code")
    
    
    btn = gr.Button("Convert")
    btn.click(fn=to_tex, inputs=input_component, outputs=output_component)
interface.launch(server_name="0.0.0.0")
