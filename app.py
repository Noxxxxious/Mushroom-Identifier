import sys

from PyQt5.QtWidgets import QApplication
import MyWindow


def window():
    app = QApplication(sys.argv)
    win = MyWindow.MyWindow()
    sys.exit(app.exec_())


if __name__ == '__main__':
    window()

