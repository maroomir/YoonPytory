import os
import csv
from datetime import datetime


def parse_nlm(strLogPath: str,
              strMode="Train"  # Train, Eval, Test
              ):
    def to_partition(strLine: str):
        pList = strLine.replace(",", "").split(" ")  # Clear the tags
        pDic = {}
        for strItem in pList:
            if strItem.find("=") > 0:
                pDic[strItem.split("=")[0]] = strItem.split("=")[1]
        return pDic

    with open(strLogPath, 'r') as pFile:
        pListTag = pFile.read().split("\n")
    pListLog = []
    for strTag in pListTag:
        if strTag.find(strMode) >= 0:
            pListLog.append(to_partition(strTag))
        else:
            pass
    return pListLog


class YoonNLM:  # Network Log Manager
    def __init__(self,
                 nStartEpoch=0,
                 strRoot="./NLM",
                 strMode="Train"  # Train, Eval, Test
                 ):
        self.epoch = nStartEpoch
        self.root = strRoot
        self.txt_path = ""
        self.csv_path = ""
        self.mode = strMode

    def write(self,
              iItem: int,
              nLength: int,
              **kwargs):
        strMessage = self.mode + " epoch={:3d} [{}/{}]".format(self.epoch, iItem + 1, nLength)
        for pItem in kwargs.items():
            strMessage += " {}={}".format(pItem[0], pItem[1])
        if iItem >= nLength - 1:
            self.epoch += 1
            self.__trace__(strMessage)
        return strMessage

    def __trace__(self, strMessage: str):
        pNow = datetime.now()
        strDirPath = os.path.join(self.root, str(pNow.year), str(pNow.month)).__str__()
        if not os.path.exists(strDirPath):
            os.makedirs(strDirPath)
        strCurrentFilePath = strDirPath + "/{}.txt".format(pNow.day)
        if strCurrentFilePath != self.txt_path:
            pListLog = parse_nlm(self.txt_path, self.mode)
            self.__record__(pListLog)
            self.txt_path = strCurrentFilePath
        with open(self.txt_path, mode='a') as pFile:
            pFile.write("[" + pNow.strftime("%H:%M:%S") + "] " + strMessage + "\n")

    def __record__(self, pListLog: list):
        self.csv_path = self.txt_path.replace('.txt', '.csv')
        with open(self.csv_path, 'a', newline='') as pFile:
            pWriter = csv.writer(pFile)
            for i in range(len(pListLog)):
                if i == 0:
                    pWriter.writerow([strItem for strItem in pListLog[i].keys()])
                pWriter.writerow([strContents for strContents in pListLog[i].values()])
