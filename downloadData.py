from urllib import request
import pandas as pd

file_url = 'http://127.0.0.1:5000/download'

# dataset = pd.read_csv("test1.csv", sep =',')

def downloadFile(url):
    fileOpen = request.urlopen(url)
    file_info = fileOpen.read()
    # file_info_str = str(file_info)
    file_info_str = file_info.decode('utf-8')
    file_lines = file_info_str.split('\\n')

    newFile = open('dataFile.data', "w")
    for info in file_lines:
        info = info.replace(r'\r', '')
        newFile.write(info + "\n")
    newFile.close()

    return 0