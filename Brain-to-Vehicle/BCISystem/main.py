# main.py
import sys
from PyQt6.QtWidgets import QApplication
from ui.main_window import MainWindow
import config


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = MainWindow()
    window.setWindowTitle("脑机接口小车控制系统")
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()