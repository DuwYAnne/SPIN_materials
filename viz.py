import matplotlib
import json
# Make sure that we are using QT5
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


class ScrollableWindow(QtWidgets.QMainWindow):
    def __init__(self, fig):
        self.qapp = QtWidgets.QApplication([])

        QtWidgets.QMainWindow.__init__(self)
        self.widget = QtWidgets.QWidget()
        self.setCentralWidget(self.widget)
        self.widget.setLayout(QtWidgets.QVBoxLayout())
        self.widget.layout().setContentsMargins(0,0,0,0)
        self.widget.layout().setSpacing(0)

        self.fig = fig
        self.canvas = FigureCanvas(self.fig)
        self.canvas.draw()
        self.scroll = QtWidgets.QScrollArea(self.widget)
        self.scroll.setWidget(self.canvas)

        self.nav = NavigationToolbar(self.canvas, self.widget)
        self.widget.layout().addWidget(self.nav)
        self.widget.layout().addWidget(self.scroll)

        self.show()
        exit(self.qapp.exec_()) 

with open('hyperparam_search.json', 'r') as f:
    tests = json.load(f)
fig, axs = plt.subplots(ncols=6, nrows=9, figsize=(16, 16))
i = 0
for ax in axs.flatten():
    ax.plot(tests[i]['mae'])
    lr = tests[i]['lr']
    ratio = tests[i]['test-ratio']
    hidden = tests[i]['n_h']
    optim = tests[i]['optim']
    ax.set_title(f'lr={lr}, r={ratio}, h={hidden}, o={optim}')
    i += 1
fig.tight_layout()
a = ScrollableWindow(fig)