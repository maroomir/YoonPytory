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

    # Define the Levenshtein Distance Algorithm
    @staticmethod
    def levenshetein(pSource, pTarget):
        # For all i and j, distance[i, j] will hold the Levenshetein distance
        # between the first i characters of source and the first j characters of target
        nLengthSource, nLengthTarget = len(pSource), len(pTarget)
        pDistance = numpy.zeros((nLengthSource + 1, nLengthTarget + 1))
        # source prefixes can be transformed into empty string
        # by dropping all characters
        pDistance[:, 0] = numpy.array([i for i in range(1, nLengthSource)])
        # target prefixes can be reached from empty source prefix
        # by inserting every character
        pDistance[0, :] = numpy.array([j for j in range(1, nLengthTarget)])
        for j in range(1, nLengthTarget):
            for i in range(1, nLengthSource):
                nCost = 0 if pSource[i] == pTarget[j] else 1
                pDistance[i, j] = min(pDistance[i - 1, j] + 1,  # deletion
                                      pDistance[i, j - 1] + 1,  # insertion
                                      pDistance[i - 1, j - 1] + nCost  # substitution
                                      )
        return pDistance[nLengthSource, nLengthTarget]


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
