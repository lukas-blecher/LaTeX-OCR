#!/usr/bin/env python3
# Required parameters:
# @raycast.schemaVersion 1
# @raycast.title latex2ocr_remote
# @raycast.mode compact

# Optional parameters:
# @raycast.icon 🤖

import io

import pyperclip
import requests
from PIL import ImageGrab

# replace the address of your remote server
API_URL = 'http://127.0.0.1:8502/predict/'


def latex2ocr_remote():
    im = ImageGrab.grabclipboard()
    
    if im != None:
        print('Read image from clipboard')
        imbytes = io.BytesIO()

        im.save(imbytes, format='png')
        print('Convert to bytes')

        response = requests.post(API_URL, files={'file': imbytes.getvalue()})
        latex_code = response
        print(latex_code.json())

        pyperclip.copy(latex_code.json())
        print('Results copied to clipboard.')
    else:
        print('No Image in Clipboard')


if __name__ == "__main__":
    latex2ocr_remote()
