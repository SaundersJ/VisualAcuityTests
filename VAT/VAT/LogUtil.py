from pathlib import Path
from datetime import datetime

def init(name):
    split = name.split(".")
    global fileName
    fileName = split[0] + "_log.txt"
    write("\n\n")
    write("==== Init Program {}".format(datetime.now()))

def write(msg):
    file = open(fileName, "a")
    file.write(msg)
    file.write("\n")
    file.close()