import sys
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import QObject, Qt, pyqtSlot, pyqtSignal, QThread
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QVBoxLayout, QWidget,\
    QPushButton, QTextEdit, QLabel, QLineEdit, QFormLayout
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
        QApplication.setWindowIcon(QtGui.QIcon(':/icons/icon.svg'))
        self.left = 300
        self.top = 300
        self.width = 500
        self.height = 300
        self.setGeometry(self.left, self.top, self.width, self.height)

        # Create LaTeX display
        self.webView = QWebEngineView()
        self.webView.setHtml("")
        self.webView.setMinimumHeight(70)

        # Create textbox
        self.textbox = QTextEdit(self)
        self.textbox.textChanged.connect(self.displayPrediction)

        # Create temperature text input
        self.tempField = QLineEdit(self)
        self.tempField.setText(str(self.args.get('temperature', 0.25)))

        # Create snip button
        self.snipButton = QPushButton('Snip', self)
        self.snipButton.clicked.connect(self.onClick)

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
        lay.addWidget(self.snipButton)
        lay.addWidget(self.retryButton)
        settings = QFormLayout()
        settings.addRow('Temperature:', self.tempField)
        lay.addLayout(settings)

    @pyqtSlot()
    def onClick(self):
        self.close()
        self.snipWidget.snip()

    def returnSnip(self, img=None):
        # Show processing icon
        pageSource = """<center>
        <img src="qrc:/icons/processing-icon-anim.svg" width="50", height="50">
        </center>"""
        self.webView.setHtml(pageSource)
        self.textbox.setText("")

        self.snipButton.setEnabled(False)

        self.show()
        try:
            self.args.temperature = float(self.tempField.text())
            if self.args.temperature == 0:
                self.args.temperature = 1e-8
        except:
            pass
        # Run the model in a separate thread
        self.thread = ModelThread(img=img, args=self.args, objs=self.objs)
        self.thread.finished.connect(self.returnPrediction)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def returnPrediction(self, result):
        self.snipButton.setEnabled(True)

        success, prediction = result["success"], result["prediction"]

        if success:
            self.displayPrediction(prediction)
            self.retryButton.setEnabled(True)
        else:
            self.webView.setHtml("")
            msg = QMessageBox()
            msg.setWindowTitle(" ")
            msg.setText("Prediction failed.")
            msg.exec_()

    def displayPrediction(self, prediction=None):
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
        <div id="equation" style="font-size:17pt; visibility:hidden">$${equation}$$</div>
        </body>
        </html>
            """.format(equation=prediction)
        self.webView.setHtml(pageSource)


class ModelThread(QThread):
    finished = pyqtSignal(dict)

    def __init__(self, img, args, objs):
        super().__init__()
        self.img = img
        self.args = args
        self.objs = objs

    def run(self):
        try:
            prediction = pix2tex.call_model(self.args, *self.objs, img=self.img)
            self.finished.emit({"success": True, "prediction": prediction})
        except Exception as e:
            print(e)
            self.finished.emit({"success": False, "prediction": None})


class SnipWidget(QMainWindow):
    isSnipping = False

    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        root = tk.Tk()
        screenWidth = root.winfo_screenwidth()
        screenHeight = root.winfo_screenheight()
        self.setGeometry(0, 0, screenWidth, screenHeight)

        self.begin = QtCore.QPoint()
        self.end = QtCore.QPoint()

        self.mouse = Controller()

    def snip(self):
        self.isSnipping = True
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))

        self.show()

    def paintEvent(self, event):
        if self.isSnipping:
            brushColor = (0, 180, 255, 100)
            lw = 3
            opacity = 0.3
        else:
            brushColor = (255, 255, 255, 0)
            lw = 3
            opacity = 0

        self.setWindowOpacity(opacity)
        qp = QtGui.QPainter(self)
        qp.setPen(QtGui.QPen(QtGui.QColor('black'), lw))
        qp.setBrush(QtGui.QColor(*brushColor))
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
        self.isSnipping = False
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
