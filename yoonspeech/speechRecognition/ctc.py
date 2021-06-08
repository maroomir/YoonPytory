import os

import Levenshtein
import torch
import torch.nn
import torch.nn.functional
from torch import tensor
from torch.nn import Module
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

from yoonspeech.data import YoonDataset


class ASRDataset(Dataset):
    def __init__(self,
                 pDataset: YoonDataset):
        self.data = pDataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        pArrayInput = self.data[item].buffer
        pArrayTarget = self.data[item].get_phonemes_array()
        return pArrayInput, pArrayTarget


def collate_tensor(pListTensor):
    pListInput = [torch.tensor(pData) for pData, nLabel in pListTensor]
    pListTarget = [torch.tensor(nLabel) for pData, nLabel in pListTensor]
    pListInputLength = [len(pData) for pData in pListInput]
    pListTargetLength = [len(pData) for pData in pListTarget]
    pTensorInput = torch.nn.utils.rnn.pad_sequence(pListInput, batch_first=True)
    pTensorTarget = torch.nn.utils.rnn.pad_sequence(pListTarget, batch_first=True)
    return pTensorInput, pTensorTarget, pListInputLength, pListTargetLength


# Define the CTC Model
class CTC(Module):
    def __init__(self,
                 nDimInput=39,
                 nDimOutput=256,
                 nCountClass=40,
                 nCountLayer=2):
        super(CTC, self).__init__()
        self.lstm = torch.nn.LSTM(nDimInput, nDimOutput, num_layers=nCountLayer, batch_first=True)
        self.fc_layer = torch.nn.Linear(nDimOutput, nDimOutput)
        self.classification_layer = torch.nn.Linear(nDimOutput, nCountClass)

    def forward(self, pTensorX: tensor):
        pTensorX, pHidden = self.lstm(pTensorX)
        pTensorX = self.fc_layer(pTensorX)
        pTensorX = self.classification_layer(pTensorX)
        return torch.nn.functional.log_softmax(pTensorX, dim=1).transpose(0, 1)


def __process_decode(pTensorData, pTensorLabel, pListLabelLength):
    dLER = 0
    for i in range(pTensorData.size(1)):
        # Find the maximum likelihood output
        pTensorClassification = torch.argmax(pTensorData[:, i, :], dim=-1)
        # Collapse the repeated outputs  (Note : simplified implementation)
        pCollapseLabel = [pTensorClassification[j] for j in range(len(pTensorClassification))
                          if (j == 0) or pTensorClassification[j] != pTensorClassification[j - 1]]
        # Compute the edit distance between the reference and estimated ones
        pCollapseLabel = torch.tensor([nLabel for nLabel in pCollapseLabel if nLabel != 0])
        dDistance = Levenshtein.distance(pCollapseLabel, pTensorLabel[i][:pListLabelLength[i]]) / pListLabelLength[i]
        dLER += dDistance
    dLER /= pTensorData.size(1)
    return dLER


def __process_train(nEpoch: int, pModel: CTC, pDataLoader: DataLoader, pCriterion, pOptimizer):
    if torch.cuda.is_available():
        pDevice = torch.device('cuda')
    else:
        pDevice = torch.device('cpu')
    pModel.train()
    dTotalLoss = 0
    dTotalLER = 0
    pBar = tqdm(enumerate(pDataLoader))
    for i, (pTensorData, pTensorTarget, pListDataLength, pListTargetLength) in pBar:
        # Move data and label to the device
        pTensorData = pTensorData.type(torch.FloatTensor).to(pDevice)
        pTensorTarget = pTensorTarget.type(torch.LongTensor).to(pDevice)
        # Predict the output given the input data
        pTensorOutput = pModel(pTensorData)
        # Compute the loss by comparing the predicted output to the reference labels
        pTensorLoss = pCriterion(pTensorOutput, pTensorTarget, pListDataLength, pListTargetLength)
        dTotalLoss += pTensorLoss.item()
        # Compute the loss by comparing the predicted output to the reference labels
        dLER = __process_decode(pTensorOutput, pTensorTarget, pListTargetLength)
        dTotalLER += dLER
        # Perform backpropagation if it is the training mode
        pOptimizer.zero_grad()
        pTensorLoss.backward()
        pOptimizer.step()
        # Display the running progress
        pBar.set_description("Train Epoch: {} [{}/{}] CTC_Loss: {:.4f} LER: {:.4f}".
                             format(nEpoch, i, len(pDataLoader), dTotalLoss / (i + 1), dTotalLER / (i + 1)))
    return dTotalLoss, dTotalLER


def __process_test(nEpoch: int, pModel: CTC, pDataLoader: DataLoader, pCriterion):
    if torch.cuda.is_available():
        pDevice = torch.device('cuda')
    else:
        pDevice = torch.device('cpu')
    pModel.eval()
    dTotalLoss = 0
    dTotalLER = 0
    pBar = tqdm(enumerate(pDataLoader))
    for i, (pTensorData, pTensorTarget, pListDataLength, pListTargetLength) in pBar:
        # Move data and label to the device
        pTensorData = pTensorData.type(torch.FloatTensor).to(pDevice)
        pTensorTarget = pTensorTarget.type(torch.LongTensor).to(pDevice)
        with torch.no_grad():
            # Predict the output given the input data
            pTensorOutput = pModel(pTensorData)
            # Compute the loss by comparing the predicted output to the reference labels
            pTensorLoss = pCriterion(pTensorOutput, pTensorTarget, pListDataLength, pListTargetLength)
            dTotalLoss += pTensorLoss.item()
            # Compute the loss by comparing the predicted output to the reference labels
            dLER = __process_decode(pTensorOutput, pTensorTarget, pListTargetLength)
            dTotalLER += dLER
            # Display the running progress
            pBar.set_description("Evaluate Epoch: {} [{}/{}] CTC_Loss: {:.4f} LER: {:.4f}".
                                 format(nEpoch, i, len(pDataLoader), dTotalLoss / (i + 1), dTotalLER / (i + 1)))
    return dTotalLoss, dTotalLER


def train(nEpoch: int,
          pTrainData: YoonDataset,
          pValidationData: YoonDataset,
          strModelPath="model_ctc.pth",
          nSizeBatch=8,
          bInitTrain=False,
          dLearningRate=0.01,
          nWorker=0,  # 0 = CPU, 4 = CUDA
          ):
    # Set the device for running the CTC model
    # Check if we can use a GPU device
    if torch.cuda.is_available():
        pDevice = torch.device('cuda')
    else:
        pDevice = torch.device('cpu')
    # Define a network architecture
    pModel = CTC(nDimInput=pTrainData.get_dimension(), nCountClass=pTrainData.phoneme_count)
    pModel = pModel.to('cpu')  #pModel.to(pDevice)
    # Define an optimizer
    pOptimizer = Adam(pModel.parameters(), lr=dLearningRate)
    # Define a training criterion
    pCriterion = torch.nn.CTCLoss(blank=0)
    # Load the pre-trained model if you resume the training from the model
    nStart = 0
    if os.path.isfile(strModelPath) and bInitTrain is False:
        pModelData = torch.load(strModelPath, map_location=pDevice)
        pModel.load_state_dict(pModelData['encoder'])
        pOptimizer.load_state_dict(pModelData['optimizer'])
        nStart = pModelData['epoch']
        print("## Success to load the CTC model : epoch {}".format(nStart))
    # Define training and test dataset
    pTrainDataset = ASRDataset(pTrainData)
    pTrainLoader = DataLoader(pTrainDataset, batch_size=nSizeBatch, collate_fn=collate_tensor, shuffle=True,
                              num_workers=nWorker, pin_memory=True)
    pValidDataset = ASRDataset(pValidationData)
    pValidLoader = DataLoader(pValidDataset, batch_size=nSizeBatch, collate_fn=collate_tensor, shuffle=False,
                              num_workers=nWorker, pin_memory=True)
    # Perform training / validation processing
    dMinLoss = 10000.0
    nCountDecrease = 0
    for iEpoch in range(nStart, nEpoch + 1):
        __process_train(iEpoch, pModel, pTrainLoader, pCriterion, pOptimizer)
        dValidLoss, dValidLER = __process_test(iEpoch, pModel, pTrainLoader, pCriterion)
        # Save the trained model at every 10 epochs
        if dMinLoss > dValidLoss:
            dMinLoss = dValidLoss
            torch.save({'epoch': iEpoch, 'encoder': pModel.state_dict(),
                        'optimizer': pOptimizer.state_dict()}, strModelPath)
            nCountDecrease = 0
        else:
            nCountDecrease += 1
            # Decrease the learning rate by 2 when the test loss decrease 5 times in a row
            if nCountDecrease == 5:
                pDicOptimizerState = pOptimizer.state_dict()
                pDicOptimizerState['param_groups'][0]['lr'] /= 2
                pOptimizer.load_state_dict(pDicOptimizerState)
                print('learning rate is divided by 2')
                nCountDecrease = 0

