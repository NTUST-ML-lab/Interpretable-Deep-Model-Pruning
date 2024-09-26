import os
import time
import pandas as pd
import numpy as np
import json
import shutil

def checkkk(strr):
    '''
    check path available, if not, it will create.
    return absolute path of input (end withou / )
    '''
    strr = strr.replace("\n", "")
    path = os.path.split(os.path.realpath(__file__))[0] + "/" + strr
    if not os.path.isdir(path):
        os.makedirs(path)
    return path    

def existFile(path):
    return os.path.exists(path)

def checkFile(path : str, copy=True):
    '''
    If copy: copy the old file and return original path
    else: return new path
    '''

    name = path.split(".")[:-1]
    name = "".join(name)
    subFile = path.split(".")[-1]
    if existFile(path):
        if copy:
            copyPath = name + "-"
            cnt = 1
            while(existFile(copyPath + str(cnt) + "." + subFile)):
                cnt += 1
            shutil.copyfile(path, copyPath + str(cnt) + "." + subFile)
            return path
        else:
            cnt = 1
            name += "-"
            while(existFile(name + str(cnt) + "." + subFile)):
                cnt += 1
            return name + str(cnt) + "." + subFile
    return path
    
def listdirs(folder):
    '''
    return dirs in the path
    '''
    return [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]

def sortDict(dictt, reverse = True):
    return dict(sorted(dictt.items(), key=lambda item: item[1], reverse=reverse))

def DateCompare(date1, date2): # YYYY-MM-DD
    '''
    if  date1 >  date2 return 1
        date1 == date2 return 0
        date1 <  date2 return -1
    '''
    date1 = date1[:10]
    date2 = date2[:10]

    formatted_date1 = time.strptime(date1, "%Y-%m-%d")
    formatted_date2 = time.strptime(date2, "%Y-%m-%d")
    
    if formatted_date1 < formatted_date2:
        return -1
    else:
        return int( formatted_date1 > formatted_date2 )

def readJson(path):
    f = open( path )
    return json.load(f)

def checkFileName(fileName:str):
    '''
    check the fileName contains non-available character
    return abailable fileName
    '''
    fileName = fileName.replace("\\", "＼")
    fileName = fileName.replace("/", "／")
    fileName = fileName.replace(":", "：")
    fileName = fileName.replace("*", "＊")
    fileName = fileName.replace("?", "？")
    fileName = fileName.replace('"', "〞")
    fileName = fileName.replace("<", "＜")
    fileName = fileName.replace(">", "＞")
    fileName = fileName.replace("|", "▕")
    return fileName[:30]


if __name__ == "__main__":
    tmp = checkFile("test0/ZZZZ_test2.cmd")
    pass