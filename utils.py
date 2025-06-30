from PyQt5.QtWidgets import QPushButton, QFrame, QLabel, QVBoxLayout, QGraphicsDropShadowEffect
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np


class ResultsChart(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        plt.close()

    def update_chart(self, predictions, class_names):
        self.ax.clear()

        # Top 5 kết quả
        values = predictions
        indices = np.argsort(values)[::-1][:5]

        bar_colors = ['#36a2eb', '#ff6384', '#4bc0c0', '#ff9f40', '#9966ff']
        bars = self.ax.barh([class_names[i] for i in indices],
                            [values[i] * 100 for i in indices],
                            color=bar_colors)

        # Thêm phần trăm
        for i, bar in enumerate(bars):
            width = bar.get_width()
            self.ax.text(width + 1, bar.get_y() + bar.get_height() / 2,
                         f'{width:.1f}%', ha='left', va='center',
                         color='#333333', fontsize=9, fontweight='bold')

        self.ax.set_title('Top 5 Kết Quả Dự Đoán', fontsize=12, fontweight='bold')
        self.ax.set_xlim(0, 100)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)

        self.fig.tight_layout()
        self.draw()


class StyledButton(QPushButton):
    def __init__(self, text, parent=None, color="#5D9CEC"):
        super().__init__(text, parent)
        self.setMinimumHeight(50)
        self.setMinimumWidth(150)
        self.setCursor(Qt.PointingHandCursor)

        self.color = color
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: none;
                border-radius: 10px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {self._lighten_color(color)};
            }}
            QPushButton:pressed {{
                background-color: {self._darken_color(color)};
            }}
        """)

        # Shadow effect
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 80))
        shadow.setOffset(0, 5)
        self.setGraphicsEffect(shadow)

    def _lighten_color(self, color, factor=1.2):
        color = color.lstrip('#')
        r, g, b = [int(color[i:i + 2], 16) for i in (0, 2, 4)]
        r = min(int(r * factor), 255)
        g = min(int(g * factor), 255)
        b = min(int(b * factor), 255)
        return f'#{r:02x}{g:02x}{b:02x}'

    def _darken_color(self, color, factor=0.8):
        color = color.lstrip('#')
        r, g, b = [int(color[i:i + 2], 16) for i in (0, 2, 4)]
        r = max(int(r * factor), 0)
        g = max(int(g * factor), 0)
        b = max(int(b * factor), 0)
        return f'#{r:02x}{g:02x}{b:02x}'


class ImagePanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("imagePanel")
        self.setMinimumSize(350, 350)
        self.setStyleSheet("""
            #imagePanel {
                background-color: #f7f7f7;
                border-radius: 15px;
                border: 2px dashed #bbbbbb;
            }
        """)

        self.image_label = QLabel("Thả ảnh da vào đây")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("color: #888888; font-size: 16px;")

        layout = QVBoxLayout(self)
        layout.addWidget(self.image_label)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 30))
        shadow.setOffset(0, 5)
        self.setGraphicsEffect(shadow)

    def setPixmap(self, pixmap):
        self.image_label.setPixmap(
            pixmap.scaled(
                self.image_label.width() - 20,
                self.image_label.height() - 20,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )
        self.setStyleSheet("""
            #imagePanel {
                background-color: white;
                border-radius: 15px;
                border: none;
            }
        """)
