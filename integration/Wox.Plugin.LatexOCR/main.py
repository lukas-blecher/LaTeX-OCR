from util import WoxEx, WoxAPI, load_module

with load_module():
    import pyperclip
    import io
    import requests
    from PIL import ImageGrab

# replace the address of your remote server
API_URL = 'http://127.0.0.1:8502/predict/'

def latex_ocr_from_clipboard(im):
    
    imbytes = io.BytesIO()
    im.save(imbytes, format='png')

    response = requests.post(API_URL, files={'file': imbytes.getvalue()})
    latex_code = response

    return latex_code.json()


class Main(WoxEx):

    def query(self, keyword):
        results = list()
        im = ImageGrab.grabclipboard()
        results.append({
            "Title": "LatexOCR",
            "SubTitle": "No image founded in clipboard" if im is None else "Ready to OCR",
            "IcoPath": "Images/ico.ico",
            "JsonRPCAction": {
                "method": "test_func",
                "parameters": [keyword],
                "dontHideAfterAction": False      # 运行后是否隐藏Wox窗口
            }
        })
        return results

    def test_func(self, keyword):
        im = ImageGrab.grabclipboard()
        if im:
            out = latex_ocr_from_clipboard(im)
            pyperclip.copy(out)
        else:
            pyperclip.copy("No image founded in clipboard. Please copy an image to clipboard or take a screen shot first.")

if __name__ == "__main__":
    Main()