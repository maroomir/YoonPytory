import os
import csv
from datetime import datetime


def parse_nlm(log_path: str,
              mode="Train"  # Train, Eval, Test
              ):
    def to_partition(line: str):
        part_list = line.replace(",", "").split(" ")  # Clear the tags
        dic = {}
        for item in part_list:
            if item.find("=") > 0:
                dic[item.split("=")[0]] = item.split("=")[1]
        return dic

    with open(log_path, 'r') as file:
        tag_list = file.read().split("\n")
    log_list = []
    for tag in tag_list:
        if tag.find(mode) >= 0:
            log_list.append(to_partition(tag))
        else:
            pass
    return log_list


class YoonNLM:  # Network Log Manager
    """
    The shared area of YoonDataset class
    All of instances are using this shared area
    """
    def __init__(self,
                 start_epoch=0,
                 root="./NLM",
                 mode="Train"  # Train, Eval, Test
                 ):
        self.epoch = start_epoch
        self.root = root
        self.txt_path = ""
        self.csv_path = ""
        self.mode = mode

    def write(self,
              count: int,
              length: int,
              **kwargs):
        message = self.mode + " epoch={:3d} [{}/{}]".format(self.epoch, count + 1, length)
        for item in kwargs.items():
            message += " {}={:.4f}".format(item[0], item[1])
        if count >= length - 1:
            self.epoch += 1
            self.__trace__(message)
        return message

    def __trace__(self, message: str):
        now = datetime.now()
        dir_path = os.path.join(self.root, str(now.year), str(now.month)).__str__()
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        current_file_path = dir_path + "/{}.txt".format(now.day)
        if self.txt_path == "":
            self.txt_path = current_file_path
        elif current_file_path != self.txt_path:
            log_list = parse_nlm(self.txt_path, self.mode)
            self.__record__(log_list)
            self.txt_path = current_file_path
        with open(self.txt_path, mode='a') as file:
            file.write("[" + now.strftime("%H:%M:%S") + "] " + message + "\n")

    def __record__(self, logs: list):
        self.csv_path = self.txt_path.replace('.txt', '.csv')
        with open(self.csv_path, 'a', newline='') as file:
            writer = csv.writer(file)
            for i in range(len(logs)):
                if i == 0:
                    writer.writerow([item for item in logs[i].keys()])
                writer.writerow([contents for contents in logs[i].values()])
