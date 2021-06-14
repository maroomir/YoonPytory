import csv
from datetime import datetime


def train_log(strFilePath: str,
              nEpoch: int,
              iItem: int,
              nLength: int,
              **kwargs):
    strMessage = "Train epoch={:3d} [{}/{} ]".format(nEpoch, iItem + 1, nLength)
    for pItem in kwargs.items():
        strMessage += " {}={}".format(pItem[0], pItem[1])
    if iItem == nLength - 1:
        __trace__(strFilePath, strMessage)
    return strMessage


def eval_log(strFilePath: str,
             nEpoch: int,
             iItem: int,
             nLength: int,
             **kwargs):
    strMessage = "Eval epoch={} [{}/{}]".format(nEpoch, iItem + 1, nLength)
    for pItem in kwargs.items():
        strMessage += " {}={}".format(pItem[0], pItem[1])
    if iItem == nLength - 1:
        __trace__(strFilePath, strMessage)
    return strMessage


def __trace__(strFilePath: str, strMessage: str):
    pNow = datetime.now()
    with open(strFilePath, mode='a') as pFile:
        pFile.write("[" + pNow.strftime("%H:%M:%S") + "] " + strMessage + "\n")


def parse_log(strLogPath: str):
    def to_partition(strLine: str):
        pList = strLine.replace(",", "").split(" ")  # Clear the tags
        pDic = {}
        for strItem in pList:
            if strItem.find("=") > 0:
                pDic[strItem.split("=")[0]] = strItem.split("=")[1]
        return pDic

    with open(strLogPath, 'r') as pFile:
        pList = pFile.read().split("\n")
    pListTrainLog = []
    pListEvalLog = []
    for strTag in pList:
        if strTag.find("Train") >= 0:
            pListTrainLog.append(to_partition(strTag))
        elif strTag.find("Eval") >= 0:
            pListEvalLog.append(to_partition(strTag))
        else:
            pass
    return pListTrainLog, pListEvalLog


def save_csv(strCsvPath: str, pListLog: list):
    with open(strCsvPath, 'w', newline='') as pFile:
        pWriter = csv.writer(pFile)
        for i in range(len(pListLog)):
            if i == 0:
                pWriter.writerow([strItem for strItem in pListLog[i].keys()])
            pWriter.writerow([strContents for strContents in pListLog[i].values()])


if __name__ == '__main__':
    trains, evals = parse_log("LOG_TEST.txt")
    save_csv("LOG_TEST_TRAIN.csv", trains)
    save_csv("LOG_TEST_EVAL.csv", evals)

