import sys
import os
from PyQt6 import QtCore
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import (QApplication, QWidget, QPushButton, QTextEdit, QComboBox, QFileDialog,
                            QHBoxLayout, QVBoxLayout, QLabel)

from ImageDifference import *

class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        # self.window_width, self.window_height = 800, 100
        self.window_width, self.window_height = 800, 300
        self.setMinimumSize(self.window_width, self.window_height)

        layout = QVBoxLayout()
        self.setLayout(layout)

        # Creates the drop down options
        self.options = ('Get comparison images', 'getOpenFileNames()', 'getExistingDirectory()', 'getSaveFileName()')

        # Combo box
        self.combo = QComboBox()
        self.combo.addItems(self.options)
        layout.addWidget(self.combo)

        # Push button
        btn = QPushButton('Launch')
        btn.clicked.connect(self.launchDialog)
        layout.addWidget(btn)

        # Text box
        # self.textbox = QTextEdit()
        # layout.addWidget(self.textbox)

        # Label for images
        self.imagebox = QLabel(self)
        self.imagebox.setStyleSheet("border: 1px solid black;")
        layout.addWidget(self.imagebox)

        # Important variables
        self.ratio = ()
        self.original_path = ""


    def launchDialog(self):
        option = self.options.index(self.combo.currentText())

        if option == 0:
            response = self.getImage()
        elif option == 1:
            response = self.getFileNames()
        elif option == 2:
            response = self.getDirectory()
        elif option == 3:
            response = self.getSaveFileName()
        else:
            print('Got Nothing')


    def getImage(self):
        file_filter = 'Image File (*.png *.jpg);; Document File (*.pdf)'
        response = QFileDialog.getOpenFileName(
            parent=self,
            caption='Select a file',
            directory=os.getcwd(),
            filter=file_filter,
            initialFilter='Image File (*.png *.jpg)'
        )
        # self.textbox.setText(str(response))
        
        self.original_path = response[0]

        if os.path.exists(self.original_path):
            h, w = self.size().height(), self.size().width()
            self.ratio = (h, w)

            image = diff_pipeline(self.original_path, self.ratio)

            height, width, channel = image.shape
            qImg = QImage(image.data, width, height, 3*width, QImage.Format.Format_RGB888)
            self.imagebox.setPixmap(QPixmap(qImg))
            # self.resize(pixmap.width(), pixmap.height())
        else:
            print("Invalid file path")


    def getFileNames(self):
        file_filter = 'Data File (*.xlsx *.csv *.dat);; Excel File (*.xlsx *.xls);; Image File (*.png *.jpg)'
        response = QFileDialog.getOpenFileNames(
            parent=self,
            caption='Select file(s)',
            directory=os.getcwd(),
            filter=file_filter,
            initialFilter='Excel File (*.xlsx *.xls)'
        )
        # self.textbox.setText(str(response))


    def getDirectory(self):
         response = QFileDialog.getExistingDirectory(
             self,
             # caption='Select a folder'
         )
        #  self.textbox.setText(str(response))


    def getSaveFileName(self):
         file_filter = 'Data File (*.xlsx *.csv *.dat);; Excel File (*.xlsx *.xls)'
         response = QFileDialog.getSaveFileName(
             parent=self,
             caption='Select a data file',
             directory= 'Data File.dat',
             filter=file_filter,
             initialFilter='Excel File (*.xlsx *.xls)'
         )
        #  self.textbox.setText(str(response))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet('''
        QWidget {
            font-size: 20px;
        }
    ''')
    
    myApp = MyApp()
    myApp.show()

    try:
        sys.exit(app.exec())
    except SystemExit:
        print('Closing Window...')