import numpy
import torch
from torch.utils.data import Dataset

import yoonspeech.data
from yoonspeech.data import YoonDataset


# Define a SpeakerDataset class
class DvectorDataset(Dataset):
    def __init__(self,
                 pDataset: YoonDataset):
        self.data = pDataset

    def __len__(self):  # Return Dataset length to decision data-loader size
        return self.data.__len__()

    def __getitem__(self, item: int):  # obtain label and file name
        nTarget = self.data[item].label
        pArrayInput = self.data[item].buffer
        return pArrayInput, nTarget


class ASRDataset(Dataset):
    def __init__(self,
                 pDataset: YoonDataset):
        self.data = pDataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        pArrayInput = self.data[item].buffer
        pArrayTarget = yoonspeech.data.get_phonemes()
        return pArrayInput, pArrayTarget


# Define a collate function for the data loader (Assort for Batch)
def collate_dvector(pListTensor):
    pListInput = []
    pListTarget = []
    nLengthMin = min([len(pData) for pData, nLabel in pListTensor]) - 1
    for pInputData, nTargetLabel in pListTensor:
        nStart = numpy.random.randint(len(pInputData) - nLengthMin)
        # Change the tensor shape (Frame, Deltas) to (CH, Frame, Deltas)
        pListInput.append(torch.tensor(pInputData[nStart:nStart + nLengthMin]).unsqueeze(0))
        pListTarget.append(torch.tensor(nTargetLabel))
    # Grouping batch
    pListInput = torch.cat(pListInput, 0)  # (CH, Freme, Deltas) -> (CH * Batch, Frame, Deltas)
    pListTarget = torch.LongTensor(pListTarget)
    return pListInput, pListTarget


def collate_asrdata(pListTensor):
    pListInput = [torch.tensor(pData) for pData, nLabel in pListTensor]
    pListTarget = [torch.tensor(nLabel) for pData, nLabel in pListTensor]
    pListInputLength = [len(pData) for pData in pListInput]
    pListTargetLength = [len(pData) for pData in pListTarget]
    pTensorInput = torch.nn.utils.rnn.pad_sequence(pListInput, batch_first=True)
    pTensorTarget = torch.nn.utils.rnn.pad_sequence(pListTarget, batch_first=True)
    return pTensorInput, pTensorTarget, pListInputLength, pListTargetLength


# Define the Levenshtein Distance Algorithm
def levenshetein_distance(pTensorA, pTensorB):
    nLengthA, nLengthB = len(pTensorA), len(pTensorB)
    if nLengthB > nLengthA:
        return levenshetein_distance(nLengthB, nLengthA)
    if nLengthB is 0:
        return nLengthA
    pListRowPrev = list(range(nLengthB + 1))
    pListRowCurrent = [0] * (nLengthB + 1)
    for i, iA in enumerate(pTensorA):
        pListRowCurrent[0] = i + 1
        for j, jB in enumerate(pTensorB):
            dInsert = pListRowCurrent[j] + 1
            dDelete = pListRowPrev[j + 1] + 1
            dReplace = pListRowPrev[j] + (1 if iA != jB else 0)
            pListRowCurrent[j + 1] = min(dInsert, dDelete, dReplace)
        pListRowPrev[:] = pListRowCurrent[:]
    return pListRowPrev[-1]