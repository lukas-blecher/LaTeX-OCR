from shutil import which
import io
import subprocess
import sys
import os
import tempfile
from PyQt6 import QtCore, QtGui
from PyQt6.QtCore import Qt, pyqtSlot, pyqtSignal, QThread, QTimer
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import QMainWindow, QApplication, QMessageBox, QVBoxLayout, QWidget, \
    QPushButton, QTextEdit, QFormLayout, QHBoxLayout, QDoubleSpinBox
from pynput.mouse import Controller

from PIL import ImageGrab, Image
import numpy as np
from screeninfo import get_monitors
from pix2tex import cli
from pix2tex.utils import in_model_path

import pix2tex.resources.resources
class App(QMainWindow):
    isProcessing = False

    def __init__(self, args=None):
        super().__init__()
        self.args = args
        self.model = cli.LatexOCR(self.args)
        self.initUI()
        self.snipWidget = SnipWidget(self)
        self.show()

    def initUI(self):
        self.setWindowTitle("LaTeX OCR")
        QApplication.setWindowIcon(QtGui.QIcon(':/icons/icon.svg'))
        self.left = 300
        self.top = 300
        self.width = 500
        self.height = 300
        self.setGeometry(self.left, self.top, self.width, self.height)

        # Create LaTeX display
        self.webView = QWebEngineView()
        self.webView.setHtml("")
        self.webView.setMinimumHeight(80)

        # Create textbox
        self.textbox = QTextEdit(self)
        self.textbox.textChanged.connect(self.displayPrediction)
        self.textbox.setMinimumHeight(40)

        # Create temperature text input
        self.tempField = QDoubleSpinBox(self)
        self.tempField.setValue(self.args.temperature)
        self.tempField.setRange(0, 1)
        self.tempField.setSingleStep(0.1)

        # Create snip button
        if sys.platform == "darwin":
            self.snipButton = QPushButton('Snip [Option+S]', self)
            self.snipButton.clicked.connect(self.onClick)
        else:
            self.snipButton = QPushButton('Snip [Alt+S]', self)
            self.snipButton.clicked.connect(self.onClick)

        self.shortcut = QtGui.QShortcut(QtGui.QKeySequence('Alt+S'), self)
        self.shortcut.activated.connect(self.onClick)

        # Create retry button
        self.retryButton = QPushButton('Retry', self)
        self.retryButton.setEnabled(False)
        self.retryButton.clicked.connect(self.returnSnip)

        # Create layout
        centralWidget = QWidget()
        centralWidget.setMinimumWidth(200)
        self.setCentralWidget(centralWidget)

        lay = QVBoxLayout(centralWidget)
        lay.addWidget(self.webView, stretch=4)
        lay.addWidget(self.textbox, stretch=2)
        buttons = QHBoxLayout()
        buttons.addWidget(self.snipButton)
        buttons.addWidget(self.retryButton)
        lay.addLayout(buttons)
        settings = QFormLayout()
        settings.addRow('Temperature:', self.tempField)
        lay.addLayout(settings)

    def toggleProcessing(self, value=None):
        if value is None:
            self.isProcessing = not self.isProcessing
        else:
            self.isProcessing = value
        if self.isProcessing:
            text = 'Interrupt'
            func = self.interrupt
        else:
            if sys.platform == "darwin":
                text = 'Snip [Option+S]'
            else:
                text = 'Snip [Alt+S]'
            func = self.onClick
            self.retryButton.setEnabled(True)
        self.shortcut.setEnabled(not self.isProcessing)
        self.snipButton.setText(text)
        self.snipButton.clicked.disconnect()
        self.snipButton.clicked.connect(func)
        self.displayPrediction()

    @pyqtSlot()
    def onClick(self):
        self.close()
        if os.environ.get('SCREENSHOT_TOOL') == "gnome-screenshot":
            self.snip_using_gnome_screenshot()
        elif os.environ.get('SCREENSHOT_TOOL') == "grim":
            self.snip_using_grim()
        elif os.environ.get('SCREENSHOT_TOOL') == "pil":
            self.snipWidget.snip()
        elif which('gnome-screenshot'):
            self.snip_using_gnome_screenshot()
        elif which('grim') and which('slurp'):
            self.snip_using_grim()
        else:
            self.snipWidget.snip()

    @pyqtSlot()
    def interrupt(self):
        if hasattr(self, 'thread'):
            self.thread.terminate()
            self.thread.wait()
            self.toggleProcessing(False)

    def snip_using_gnome_screenshot(self):
        try:
            with tempfile.NamedTemporaryFile() as tmp:
                subprocess.run(["gnome-screenshot", "--area", f"--file={tmp.name}"])
                # Use `tmp.name` instead of `tmp.file` due to compatability issues between Pillow and tempfile
                self.returnSnip(Image.open(tmp.name))
        except:
            print(f"Failed to load saved screenshot! Did you cancel the screenshot?")
            print("If you don't have gnome-screenshot installed, please install it.")
            self.returnSnip()

    def snip_using_grim(self):
        try:
            p = subprocess.run('slurp',
                               check=True,
                               capture_output=True,
                               text=True)
            geometry = p.stdout.strip()

            p = subprocess.run(['grim', '-g', geometry, '-'],
                               check=True,
                               capture_output=True)
            self.returnSnip(Image.open(io.BytesIO(p.stdout)))
        except:
            print(f"Failed to load saved screenshot! Did you cancel the screenshot?")
            print("If you don't have slurp and grim installed, please install them.")
            self.returnSnip()

    def returnSnip(self, img=None):
        self.toggleProcessing(True)
        self.retryButton.setEnabled(False)

        self.show()
        try:
            self.model.args.temperature = self.tempField.value()
            if self.model.args.temperature == 0:
                self.model.args.temperature = 1e-8
        except:
            pass
        # Run the model in a separate thread
        self.thread = ModelThread(img=img, model=self.model)
        self.thread.finished.connect(self.returnPrediction)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def returnPrediction(self, result):
        self.toggleProcessing(False)
        success, prediction = result["success"], result["prediction"]

        if success:
            self.displayPrediction(prediction)
            self.retryButton.setEnabled(True)
        else:
            self.webView.setHtml("")
            msg = QMessageBox()
            msg.setWindowTitle(" ")
            msg.setText("Prediction failed.")
            msg.exec()

    def displayPrediction(self, prediction=None):
        if self.isProcessing:
            pageSource = """<center>
            <img src="qrc:/icons/processing-icon-anim.svg" width="50", height="50">
            </center>"""
        else:
            if prediction is not None:
                self.textbox.setText("${equation}$".format(equation=prediction))
            else:
                prediction = self.textbox.toPlainText().strip('$')
            pageSource = """
            <html>
            <head><script id="MathJax-script" src="qrc:MathJax.js"></script>
            <script>
            MathJax.Hub.Config({messageStyle: 'none',tex2jax: {preview: 'none'}});
            MathJax.Hub.Queue(
                function () {
                    document.getElementById("equation").style.visibility = "";
                }
                );
            </script>
            </head> """ + """
            <body>
            <div id="equation" style="font-size:1em; visibility:hidden">$${equation}$$</div>
            </body>
            </html>
                """.format(equation=prediction)
        self.webView.setHtml(pageSource)


class ModelThread(QThread):
    finished = pyqtSignal(dict)

    def __init__(self, img, model):
        super().__init__()
        self.img = img
        self.model = model

    def run(self):
        try:
            prediction = self.model(self.img)
            # replace <, > with \lt, \gt so it won't be interpreted as html code
            prediction = prediction.replace('<', '\\lt ').replace('>', '\\gt ')
            self.finished.emit({"success": True, "prediction": prediction})
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished.emit({"success": False, "prediction": None})


class SnipWidget(QMainWindow):
    isSnipping = False

    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        monitos = get_monitors()
        bboxes = np.array([[m.x, m.y, m.width, m.height] for m in monitos])
        x, y, _, _ = bboxes.min(0)
        w, h = bboxes[:, [0, 2]].sum(1).max(), bboxes[:, [1, 3]].sum(1).max()
        self.setGeometry(x, y, w-x, h-y)

        self.begin = QtCore.QPoint()
        self.end = QtCore.QPoint()

        self.mouse = Controller()

        # Create and start the timer
        self.factor = QGuiApplication.primaryScreen().devicePixelRatio()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_geometry_based_on_cursor_position)
        self.timer.start(500)

    def update_geometry_based_on_cursor_position(self):
        if not self.isSnipping:
            return

        # Update the geometry of the SnipWidget based on the current screen
        mouse_pos = QtGui.QCursor.pos()
        screen = QGuiApplication.screenAt(mouse_pos)
        if screen:
            self.factor = screen.devicePixelRatio()
            screen_geometry = screen.geometry()
            self.setGeometry(screen_geometry)


    def snip(self):
        self.isSnipping = True
        self.setWindowFlags(QtCore.Qt.WindowType.WindowStaysOnTopHint)
        QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.CursorShape.CrossCursor))

        self.show()

    def paintEvent(self, event):
        if self.isSnipping:
            brushColor = (0, 180, 255, 100)
            opacity = 0.3
        else:
            brushColor = (255, 255, 255, 0)
            opacity = 0

        self.setWindowOpacity(opacity)
        qp = QtGui.QPainter(self)
        qp.setPen(QtGui.QPen(QtGui.QColor('black'), 2))
        qp.setBrush(QtGui.QColor(*brushColor))
        qp.drawRect(QtCore.QRect(self.begin, self.end))

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key.Key_Escape.value:
            QApplication.restoreOverrideCursor()
            self.close()
            self.parent.show()
        event.accept()

    def mousePressEvent(self, event):
        self.startPos = self.mouse.position

        self.begin = event.pos()
        self.end = self.begin
        self.update()

    def mouseMoveEvent(self, event):
        self.end = event.pos()
        self.update()

    def mouseReleaseEvent(self, event):
        self.isSnipping = False
        QApplication.restoreOverrideCursor()

        startPos = self.startPos
        endPos = self.mouse.position

        x1 = int(min(startPos[0], endPos[0]))
        y1 = int(min(startPos[1], endPos[1]))
        x2 = int(max(startPos[0], endPos[0]))
        y2 = int(max(startPos[1], endPos[1]))

        self.repaint()
        QApplication.processEvents()
        try:
            img = ImageGrab.grab(bbox=(x1, y1, x2, y2), all_screens=True)
        except Exception as e:
            if sys.platform == "darwin":
                img = ImageGrab.grab(bbox=(x1//self.factor, y1//self.factor,
                                           x2//self.factor, y2//self.factor), all_screens=True)
            else:
                raise e
        QApplication.processEvents()

        self.close()
        self.begin = QtCore.QPoint()
        self.end = QtCore.QPoint()
        self.parent.returnSnip(img)

def main(arguments):
    with in_model_path():
        if os.name != 'nt':
            os.environ['QTWEBENGINE_DISABLE_SANDBOX'] = '1'
        app = QApplication(sys.argv)
        ex = App(arguments)
        sys.exit(app.exec())
