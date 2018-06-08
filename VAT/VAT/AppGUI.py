#http://python-gtk-3-tutorial.readthedocs.io/en/latest/index.html

from tkinter import *
import threading

class Window(Frame):
    root = Tk()
    def refresh(self):
        self.root.update()
        self.root.after(1000, self.refresh)

    def threadFun():
        pass

    def start(self):
        self.refresh()
        threading.Thread(target=threadFun).start()

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master

def f():
    root = Tk()
    app = Window(root)
    app.start()
    app.refresh()