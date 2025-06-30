from PyQt5.QtCore import QThread, pyqtSignal
import tensorflow as tf


class LoadModelThread(QThread):
    finished = pyqtSignal(object)
    progress = pyqtSignal(int)

    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path

    def run(self):
        try:
            for i in range(0, 101, 10):
                self.progress.emit(i)
                self.msleep(200)

            model = tf.keras.models.load_model(self.model_path)
            self.finished.emit(model)
        except Exception as e:
            self.finished.emit(str(e))
