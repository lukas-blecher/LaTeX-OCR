import sys
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWidgets import QMainWindow, QApplication, QVBoxLayout, QWidget, QPushButton, QTextEdit
import resources
from pynput.mouse import Controller

import tkinter as tk
from PIL import ImageGrab

import pix2tex

QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initModel()
        self.initUI()
        self.snipWidget = SnipWidget(self)

        self.show()

    def initModel(self):
        args, *objs = pix2tex.initialize()
        self.args = args
        self.objs = objs


    def initUI(self):
        self.setWindowTitle("LaTeX OCR")
        self.setWindowIcon(QtGui.QIcon(':/icons/icon.svg'))
        self.left = 300
        self.top = 300
        self.width = 300
        self.height = 200
        self.setGeometry(self.left, self.top, self.width, self.height)


        # Create LaTeX display
        pageSource = ""
        self.webView = QWebEngineView()
        self.webView.setHtml(pageSource)
        self.webView.setMinimumHeight(70)


        # Create textbox
        self.textbox = QTextEdit(self)

        # Create snip button
        self.snipButton = QPushButton('Snip', self)
        self.snipButton.clicked.connect(self.onClick)

        # Create layout
        centralWidget = QWidget()
        centralWidget.setMinimumWidth(200)
        self.setCentralWidget(centralWidget)

        lay = QVBoxLayout(centralWidget)
        lay.addWidget(self.webView, stretch=2)
        lay.addWidget(self.textbox, stretch=3)
        lay.addWidget(self.snipButton)


    @pyqtSlot()
    def onClick(self):
        self.close()
        self.snipWidget.snip()


    def returnSnip(self, img):
        self.show()

        success = False
        try:
            prediction = pix2tex.call_model(img, self.args, *self.objs)
            success = True
        except:
            print("Prediction failed!")

        if success:
            self.textbox.setText("${equation}$".format(equation=prediction))

            pageSource = """
            <html>
            <head><script id="MathJax-script" src="qrc:MathJax.js"></script></head>
            <body>
            <p><mathjax style="font-size:1em">$${equation}$$</mathjax></p>
            </body>
            </html>
             """.format(equation=prediction)
            self.webView.setHtml(pageSource)


class SnipWidget(QMainWindow):
    is_snipping = False

    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        self.setGeometry(0, 0, screen_width, screen_height)

        self.begin = QtCore.QPoint()
        self.end = QtCore.QPoint()

        self.mouse = Controller()

    def snip(self):
        self.is_snipping = True
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))

        self.show()


    def paintEvent(self, event):
        if self.is_snipping:
            brush_color = (0, 180, 255, 100)
            lw = 3
            opacity = 0.3
        else:
            brush_color = (0, 200, 0, 128)
            lw = 3
            opacity = 0.3

        self.setWindowOpacity(opacity)
        qp = QtGui.QPainter(self)
        qp.setPen(QtGui.QPen(QtGui.QColor('black'), lw))
        qp.setBrush(QtGui.QColor(*brush_color))
        qp.drawRect(QtCore.QRect(self.begin, self.end))


    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
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
        self.is_snipping = False
        QApplication.restoreOverrideCursor()

        startPos = self.startPos
        endPos = self.mouse.position

        x1 = min(startPos[0], endPos[0])
        y1 = min(startPos[1], endPos[1])
        x2 = max(startPos[0], endPos[0])
        y2 = max(startPos[1], endPos[1])

        self.repaint()
        QApplication.processEvents()
        img = ImageGrab.grab(bbox=(x1, y1, x2, y2))
        QApplication.processEvents()

        self.close()
        self.begin = QtCore.QPoint()
        self.end = QtCore.QPoint()
        self.parent.returnSnip(img)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())