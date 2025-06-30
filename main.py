import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from ui_components import MelanomaClassifierApp

if __name__ == '__main__':
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)

    from PyQt5.QtGui import QFont

    font_db = QFont("Roboto")
    app.setFont(font_db)

    window = MelanomaClassifierApp()

    sys.exit(app.exec_())
