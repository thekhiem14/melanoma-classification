import os
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QLabel, QVBoxLayout, QHBoxLayout,
                             QWidget, QFileDialog, QFrame, QSplashScreen,
                             QGraphicsDropShadowEffect, QApplication)
from PyQt5.QtGui import QPixmap, QIcon, QColor
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QSize
import tensorflow as tf
from model_loader import LoadModelThread
from utils import ResultsChart, StyledButton, ImagePanel
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

label_map = {0: 'akiec', 1: 'bcc', 2: 'bkl', 3: 'df', 4: 'nv', 5: 'vasc', 6: 'mel'}


class SplashScreen(QSplashScreen):
    def __init__(self):
        super().__init__()
        from PyQt5.QtGui import QPainter, QBrush, QLinearGradient, QFont

        pixmap = QPixmap(400, 300)
        gradient = QLinearGradient(0, 0, 0, 300)
        gradient.setColorAt(0.0, QColor(48, 194, 124))
        gradient.setColorAt(1.0, QColor(73, 126, 223))

        painter = QPainter(pixmap)
        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.NoPen)
        painter.drawRect(0, 0, 400, 300)

        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("Arial", 22, QFont.Bold))
        painter.drawText(pixmap.rect(), Qt.AlignCenter, "Ứng dụng Phân Loại\nMelanoma")

        painter.setFont(QFont("Arial", 14))
        painter.drawText(pixmap.rect().adjusted(0, 100, 0, 0), Qt.AlignCenter, "Melanoma Classifier v1.0")
        painter.end()

        self.setPixmap(pixmap)

        from PyQt5.QtWidgets import QProgressBar
        self.progressBar = QProgressBar(self)
        self.progressBar.setGeometry(20, 250, 360, 20)
        self.progressBar.setStyleSheet("""
            QProgressBar {
                border: 2px solid white;
                border-radius: 8px;
                text-align: center;
                background-color: rgba(255, 255, 255, 50);
            }
            QProgressBar::chunk {
                background-color: white;
                border-radius: 6px;
            }
        """)


class MelanomaClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Phân Loại Melanoma | Version 1.0')
        self.setMinimumSize(900, 700)
        self.setGeometry(100, 100, 900, 700)

        # Splash screen
        self.splash = SplashScreen()
        self.splash.show()

        model_path = os.path.join(os.path.dirname(__file__), "assets", "alternative_model.keras")
        self.model_thread = LoadModelThread(model_path)
        self.model_thread.progress.connect(self.splash.progressBar.setValue)
        self.model_thread.finished.connect(self.onModelLoaded)
        self.model_thread.start()

        self.setupUI()

        self.class_names = [label_map[i] for i in range(len(label_map))]

        QTimer.singleShot(3000, self.showApp)

        self.setAcceptDrops(True)
        self.model = None

    def showApp(self):
        self.splash.close()
        self.show()

        # Animation khi hiển thị
        self.mainWidget.setGeometry(0, self.height(), self.width(), self.height())
        self.anim = QPropertyAnimation(self.mainWidget, b"geometry")
        self.anim.setDuration(1000)
        self.anim.setStartValue(self.mainWidget.geometry())
        self.anim.setEndValue(self.mainWidget.geometry().adjusted(0, -self.height(), 0, 0))
        self.anim.setEasingCurve(QEasingCurve.OutExpo)
        self.anim.start()

    def onModelLoaded(self, result):
        print(result)
        if isinstance(result, str):
            self.statusBar().showMessage(f"Lỗi khi tải mô hình: {result}")
            self.model = None
        else:
            self.model = result
            self.statusBar().showMessage("Mô hình đã được tải thành công!")

    def setupUI(self):
        # Widget chính
        self.mainWidget = QWidget()
        self.setCentralWidget(self.mainWidget)

        main_layout = QVBoxLayout()
        self.mainWidget.setLayout(main_layout)

        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f7fa;
            }
            QLabel {
                color: #2c3e50;
                font-family: 'Roboto', sans-serif;
            }
            QPushButton {
                font-family: 'Roboto', sans-serif;
            }
            QStatusBar {
                background-color: #34495e;
                color: white;
                font-weight: bold;
            }
        """)

        header = QWidget()
        header_layout = QVBoxLayout()
        header.setLayout(header_layout)
        header.setStyleSheet("""
            background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, 
                          stop:0 #5D9CEC, stop:1 #55EFC4);
            border-radius: 15px;
            margin: 10px;
            padding: 20px;
        """)

        title_label = QLabel("PHÂN LOẠI MELANOMA BẰNG AI")
        title_label.setStyleSheet("""
            color: white; 
            font-size: 24pt; 
            font-weight: bold;
            text-align: center;
        """)
        title_label.setAlignment(Qt.AlignCenter)

        subtitle_label = QLabel("Upload ảnh da để phân loại các loại tổn thương da")
        subtitle_label.setStyleSheet("color: white; font-size: 12pt;")
        subtitle_label.setAlignment(Qt.AlignCenter)

        header_layout.addWidget(title_label)
        header_layout.addWidget(subtitle_label)
        main_layout.addWidget(header)

        # Content area
        content = QWidget()
        content_layout = QHBoxLayout()
        content.setLayout(content_layout)

        # Panel bên trái
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)

        self.image_panel = ImagePanel()

        # Buttons
        buttons_widget = QWidget()
        buttons_layout = QHBoxLayout()
        buttons_widget.setLayout(buttons_layout)

        self.btn_load = StyledButton("Tải Ảnh Lên", color="#5D9CEC")
        self.btn_clear = StyledButton("Xóa", color="#95A5A6")

        buttons_layout.addWidget(self.btn_load)
        buttons_layout.addWidget(self.btn_clear)

        left_layout.addWidget(QLabel("Ảnh da:"))
        left_layout.addWidget(self.image_panel)
        left_layout.addWidget(buttons_widget)

        # Panel bên phải
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)

        # Khu vực kết quả
        result_panel = QFrame()
        result_panel.setObjectName("resultPanel")
        result_panel.setStyleSheet("""
            #resultPanel {
                background-color: white;
                border-radius: 15px;
                padding: 10px;
            }
        """)

        result_layout = QVBoxLayout()
        result_panel.setLayout(result_layout)

        result_title = QLabel("KẾT QUẢ PHÂN LOẠI:")
        result_title.setStyleSheet("font-size: 16pt; font-weight: bold; color: #2c3e50;")

        self.result_label = QLabel("Chưa có kết quả phân loại")
        self.result_label.setStyleSheet("""
            font-size: 18pt;
            font-weight: bold;
            color: #3498db;
            qproperty-alignment: AlignCenter;
            margin: 15px;
        """)

        self.confidence_label = QLabel("Độ tin cậy: -")
        self.confidence_label.setStyleSheet("""
            font-size: 14pt;
            color: #7f8c8d;
            qproperty-alignment: AlignCenter;
        """)

        # Biểu đồ kết quả
        self.results_chart = ResultsChart(self, width=4, height=3)

        # Ảnh minh họa
        self.result_image = QLabel()
        self.result_image.setAlignment(Qt.AlignCenter)
        self.result_image.setMinimumHeight(100)

        result_layout.addWidget(result_title)
        result_layout.addWidget(self.result_label)
        result_layout.addWidget(self.confidence_label)
        result_layout.addWidget(self.results_chart)
        result_layout.addWidget(self.result_image)

        right_layout.addWidget(QLabel("Kết quả phân tích:"))
        right_layout.addWidget(result_panel)

        content_layout.addWidget(left_panel, 1)
        content_layout.addWidget(right_panel, 1)
        main_layout.addWidget(content, 1)

        self.btn_load.clicked.connect(self.load_image)
        self.btn_clear.clicked.connect(self.clear_image)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 30))
        shadow.setOffset(0, 5)
        result_panel.setGraphicsEffect(shadow)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 'Chọn ảnh da', '', 'Image files (*.jpg *.jpeg *.png)'
        )
        if file_path:
            pixmap = QPixmap(file_path)
            self.image_panel.setPixmap(pixmap)
            self.classify_image(file_path)

    def clear_image(self):
        self.image_panel.image_label.setText("Thả ảnh da vào đây")
        self.image_panel.setStyleSheet("""
            #imagePanel {
                background-color: #f7f7f7;
                border-radius: 15px;
                border: 2px dashed #bbbbbb;
            }
        """)
        self.result_label.setText("Chưa có kết quả phân loại")
        self.confidence_label.setText("Độ tin cậy: -")
        self.result_image.clear()

    def classify_image(self, img_path):
        if not self.model:
            self.result_label.setText('Mô hình chưa được tải, không thể phân loại')

            return

        try:

            self.result_label.setText("Đang xử lý...")
            self.confidence_label.setText("Vui lòng đợi...")
            QApplication.processEvents()

            # Tiền xử lý ảnh
            img = image.load_img(img_path, target_size=(256, 256))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            # Dự đoán
            predictions = self.model.predict(img_array)
            predicted_class = np.argmax(predictions[0])
            confidence = round(float(predictions[0][predicted_class]) * 100, 2)
            # Hiển thị kết quả
            self.result_label.setText(f"{self.class_names[predicted_class]}")
            self.confidence_label.setText(f"Độ tin cậy: {confidence}%")

            if confidence > 90:
                self.result_label.setStyleSheet("font-size: 18pt; font-weight: bold; color: #27ae60;")
            elif confidence > 70:
                self.result_label.setStyleSheet("font-size: 18pt; font-weight: bold; color: #2980b9;")
            else:
                self.result_label.setStyleSheet("font-size: 18pt; font-weight: bold; color: #e67e22;")

            self.results_chart.update_chart(predictions[0], self.class_names)

            result_image_path = os.path.join(os.path.dirname(__file__), "assets", "melanoma",
                                             f"{self.class_names[predicted_class].lower()}.jpg")
            if os.path.exists(result_image_path):
                result_pixmap = QPixmap(result_image_path)
                self.result_image.setPixmap(result_pixmap.scaled(100, 100, Qt.KeepAspectRatio))

        except Exception as e:
            self.result_label.setText(f'Lỗi khi phân loại!')
            self.confidence_label.setText(str(e))

    # Hỗ trợ kéo thả file
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.CopyAction)
            file_path = event.mimeData().urls()[0].toLocalFile()
            pixmap = QPixmap(file_path)
            self.image_panel.setPixmap(pixmap)
            self.classify_image(file_path)
            event.accept()
        else:
            event.ignore()
