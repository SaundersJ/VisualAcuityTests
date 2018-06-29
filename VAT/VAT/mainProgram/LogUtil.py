from pathlib import Path
from datetime import datetime
import csv
import xlwt
import xlrd
from xlutils.copy import copy

#global fileName
#global csvName
#global resultsName


def init(name):
    split = name.split(".")
    global fileName
    global csvName
    global resultsName
    fileName = split[0] + "_log.txt"
    csvName = split[0] + "_Results.csv"
    resultsName = split[0] + "_Results.xls"
    write("\n\n")
    write("==== Init Program {}".format(datetime.now()))
    #writeToCSV(["Init Program {}".format(datetime.now())])

def write(msg):
    file = open(fileName, "a")
    file.write(msg)
    file.write("\n")
    file.close()

def writeToCSV(msg):
    with open(csvName, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(msg)



#https://stackoverflow.com/questions/13437727/python-write-to-excel-spreadsheet
def writeToResults(col, row, msg):
    print(msg)
    global ws
    try:
        rwb = xlrd.open_workbook(resultsName)
        wb = copy(rwb)
        ws = wb.get_sheet('Results')
    except FileNotFoundError:
        wb = xlwt.Workbook()
        ws = wb.add_sheet('Results')
    
    #wb = xlwt.Workbook()

    
    #write(col, row)
    ws.write(col, row, msg)

    wb.save(resultsName)