import sys
import os
from PyQt6 import QtCore
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage, QWheelEvent, QAction, QImageWriter
from PyQt6.QtWidgets import (QMainWindow, QApplication, QWidget, QPushButton, QTextEdit, QComboBox, QFileDialog,
                            QHBoxLayout, QVBoxLayout, QLabel, QScrollArea, QGraphicsScene, QGraphicsView)

from ImageDifference import *

class MyApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # self.window_width, self.window_height = 800, 100
        self.window_width, self.window_height = 800, 300
        self.setMinimumSize(self.window_width, self.window_height)

        widget = QWidget()
        layout = QVBoxLayout()
        # self.setLayout(layout)
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        # Creates the drop down options
        self.options = ('Get comparison image', 'Get multiple images', 'Get two, separate comparison images', 'getSaveFileName()')

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

        # Create a QGraphicsScene object and add the QPixmap object to it
        self.scene = QGraphicsScene()
        
        # Create a QGraphicsView object and set its scene property to the QGraphicsScene object
        self.view = QGraphicsView()
        self.view.setScene(self.scene)
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

        # Label for images
        self.imagebox = QLabel(self)
        self.imagebox.setStyleSheet("border: 1px solid black;") # *
        self.imagebox.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Scroll area
        self.scroll = QScrollArea()
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.imagebox)

        layout.addWidget(self.view)

        # Push button
        save_btn = QAction('Save file', self)
        save_btn.setShortcut("Ctrl+S")
        save_btn.triggered.connect(self.launchSaveDialog)
        # save_btn.clicked.connect(self.launchSaveDialog)
        # layout.addWidget(save_btn)

        # Add the "Save file" action to the menu bar or toolbar
        toolbar = self.addToolBar("Save file")
        toolbar.addAction(save_btn)

        # Important variables
        self.ratio = ()
        self.original_path = ""
        self.current_image = QPixmap()
        # print(QImageWriter.supportedMimeTypes())

    
    def wheelEvent(self, event: QWheelEvent):
        # Update the zoom level of the QGraphicsView object based on the direction and amount of the wheel scroll
        delta = event.angleDelta().y()
        factor = 1.1 if delta > 0 else 0.9
        self.view.setTransform(self.view.transform().scale(factor, factor))


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
            print('No response')

    
    def launchSaveDialog(self):
        if not self.current_image.isNull():
            # Get the file name and path from the user using a file dialog
            file_filter = 'BMP File (*.bmp);; GIF File (*.gif);; JPEG File (*.jpeg);; PNG File (*.png);; PBM File (*.pbm);; PGM File (*.pgm);; PPM File (*.ppm);; XBM File (*.xbm);; XPM File (*.xpm)'
            file_dialog = QFileDialog(self)
            file_dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
            file_dialog.setNameFilters(["BMP File (*.bmp)", "GIF File (*.gif)", "JPEG File (*.jpeg)", "PNG File (*.png)", "PBM File (*.pbm)", "PGM File (*.pgm)", "PPM File (*.ppm)", "XBM File (*.xbm)", "XPM File (*.xpm)"])
            file_dialog.setMimeTypeFilters(["image/bmp", "image/gif", "image/jpeg", "image/png", "image/x-portable-bitmap", "image/x-portable-graymap", "image/x-portable-pixmap", "image/x-xbitmap", "image/x-xpixmap"])
            # file_name, _ = file_dialog.getSaveFileName(self, "Save image file", "", filter=file_filter)
            file_dialog.selectFile("new")

            # If the user didn't cancel the file dialog, save the image
            # if file_name:
            if file_dialog.exec():
                # Save the pixmap to a file with the chosen image format
                selected_file = file_dialog.selectedFiles()[0]
                image_writer = QImageWriter(selected_file)
                # image_writer.setQuality(100)

                for i in range(0, len(QImageWriter.supportedMimeTypes())):
                    current_mime_type = QImageWriter.supportedMimeTypes()[i]
                    current_mime_type = bytes(current_mime_type).decode("utf-8")

                    if current_mime_type == file_dialog.selectedMimeTypeFilter():
                        try:
                            image_writer.write(self.current_image.toImage())
                        except:
                            print("ImageWriterError: Cannot save image file")

                        break

        # Test to check if there
        # if 


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

        if self.original_path:
            self.imagebox.clear()
            self.scene.clear()
            h, w = self.size().height(), self.size().width()
            self.ratio = (h, w)

            image = diff_pipeline(self.original_path, self.ratio)

            height, width, _ = image.shape
            qImg = QImage(image.data, width, height, 3*width, QImage.Format.Format_RGB888)
            self.current_image = QPixmap(qImg)
            self.scene.addPixmap(self.current_image)
            self.scene.update()
            # self.imagebox.setPixmap(QPixmap(qImg))


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