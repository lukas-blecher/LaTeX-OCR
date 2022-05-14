# Integration 

Integrations for [Raycast](https://www.raycast.com/) on macOS and [Wox](https://github.com/Wox-launcher/Wox) on Windows.

Currently, the integration use the api hosted on your local machine or a remote server.

## Usage

1. You need to run the LatexOCR api on your local machine on a remote server via:

```bash
    python -m pix2tex.api.run
```

2. Install the dependencies with:

```bash
    pip install pyperclip pillow requests
```

3. Edit the `API_URL` in [latex2ocr_remote.py](raycast/latex2ocr_remote.py) for macOS, and [main.py](Wox.Plugin.LatexOCR/main.py) for Windows.
4. Install the plugin according to the tutorial of Raycast or Wox.

## Preview

### Raycast on macOS

![raycast preview](raycast.gif)

### Wox on Windows

![wox preview](wox.gif)