import os

import torch
import torch.nn
import torch.nn.functional
from torch import tensor
from torch.nn import Module
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from yoonspeech.data import YoonDataset
from yoonspeech.recognition.dataset import ASRDataset, collate_asrdata


# Define the pyramidal LSTM class
class LSTM(Module):
    def __init__(self,
                 nDimInput: int,
                 nDimOutput: int,
                 nCountLayer: int,
                 dRateDropout=0.0):
        super(LSTM, self).__init__()
        self.lstm = torch.nn.LSTM(nDimInput, nDimOutput, batch_first=True, dropout=dRateDropout)
        self.layer_num = nCountLayer

    def forward(self, pTensorX: tensor):
        nBatch, nTime, nFeature = pTensorX.size()
        if self.layer_num != 0:
            pListIndexOdd = []
            pListIndexEven = []
            for i in range(nTime // 2):
                # Reduce the time resolution by half
                pListIndexOdd.append(2 * i)
                pListIndexEven.append(2 * i + 1)
            pTensorX = torch.cat((pTensorX[:, pListIndexOdd, :], pTensorX[:, pListIndexEven, :]), dim=-1)
        pTensorX, pHidden = self.lstm(pTensorX)
        return pTensorX


# Define the Attention network class
class Attention(Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self,
                pTensorDecoderState: tensor,
                pTensorTarget: tensor):
        """
        Args:
          pTensorDecoderState : N * 1 * D
          pTensorTarget  : N * T_i * D
        """
        pTensorScore = torch.bmm(pTensorDecoderState, pTensorTarget.transpose(1, 2))
        pTensorScore = torch.nn.functional.softmax(pTensorScore, dim=-1)
        pTensorOutput = torch.bmm(pTensorScore, pTensorTarget)
        return pTensorScore, pTensorOutput


# Define the listener network class
class Listener(Module):
    def __init__(self,
                 nDimInput: int,
                 nDimOutput: int,
                 nCountLayer: int,
                 dRateDropout=0.0):
        super(Listener, self).__init__()
        pListModule = []
        # Define the nCountLayer of prymidal LSTM
        for iLayer in range(nCountLayer):
            if iLayer == 0:
                pListModule.append(LSTM(nDimInput, nDimOutput, iLayer, dRateDropout))
            else:
                pListModule.append(LSTM(nDimOutput * 2, nDimOutput, iLayer, dRateDropout))
        self.network = torch.nn.Sequential(*pListModule)

    def forward(self, pTensorX: tensor):
        pTensorX = self.network(pTensorX)
        return pTensorX


# Define the listener network class
class Speller(Module):
    def __init__(self,
                 nDimInput: int,
                 nDimOutput: int,
                 nCountClass: int,
                 nCountLayer: int
                 ):
        super(Speller, self).__init__()
        self.class_count = nCountClass
        # Define the speller lstm architecture
        self.lstm = torch.nn.LSTM(nDimInput, nDimOutput, num_layers=nCountLayer, batch_first=True)
        # Define the speller classification for character distribution
        self.fc_layer = torch.nn.Linear(nDimOutput * 2, nCountClass)
        # Define the attention network architecture
        self.attention = Attention()
        self.max_step = 100
        self.output_dim = nDimOutput
        self.class_num = nCountClass

    def forward(self,
                pTensorX: tensor,
                pTensorLabel: tensor = None,
                dLearningRate=1.0,
                bTrain=True):
        pListPrediction = []
        pListScore = []
        # Process the labeling : concat <sos> and <eos> to the label
        pTensorStartContext, pTensorEndContext = self._process_labeling(pTensorLabel)
        # Get the one-hot encoder labels
        pTensorOneHotStartContext = self._process_one_hot(pTensorStartContext)
        pTensorOneHotEndContext = self._process_one_hot(pTensorEndContext)
        # Get the input of the first character : <sos>
        pTensorInputWord, pTensorHidden = self.initialize(pTensorOneHotStartContext)
        if dLearningRate > 0:
            nMaxStep = pTensorEndContext.size(1)
        else:
            nMaxStep = self.max_step
        # Decode the sequence to sequence network
        for iStep in range(nMaxStep):
            # Forward each time step
            pTensorPredict, pTensorHidden, pTensorContext, pTensorScore = self._process_network(pTensorX, pTensorHidden,
                                                                                                pTensorInputWord)
            pListPrediction.append(pTensorPredict)
            pListScore.append(pTensorScore)
            if dLearningRate > 0:
                # Need to implement the tf-rate version
                # Make the next step input with the ground truth for teacher forcing
                pTensorInputWord = torch.cat((pTensorOneHotStartContext[:, iStep, :].unsqueeze(1), pTensorContext), dim=-1)
            else:
                # Make the next step input with the predicted output of current step
                pTensorInputWord = torch.cat((pTensorPredict, pTensorContext), dim=-1)
        return torch.cat(pListPrediction, dim=1), torch.cat(pListScore, dim=1), pTensorEndContext

    # Define a function for making the first input
    def initialize(self, pTensorTarget):
        nBatchSize = pTensorTarget.size(0)
        pTensorWord = pTensorTarget[:, 0, :].view(nBatchSize, 1, -1)
        pTensorContext = torch.zeros((nBatchSize, 1, self.output_dim)).to(pTensorTarget.device)
        pTensorFirstWord = torch.cat((pTensorWord, pTensorContext), dim=-1)
        pHidden = None
        return pTensorFirstWord, pHidden

    # Define a step forward for a sequence-to-sequence style network
    def _process_network(self,
                         pTensorFeature: tensor,
                         pTensorHidden: tensor,
                         pTensorWord: tensor):
        # Set inputs with the output word of previous time-step and last hidden output
        pTensorOutputRNN, pTensorHidden = self.lstm(pTensorWord, pTensorHidden)
        # Compute attention and context c_i from the RNN Output s_i and listener features H
        pTensorScore, pTensorOutputContext = self.attention(pTensorOutputRNN, pTensorFeature)
        # Predict the word character from the RNN output s_i and context c_i
        pTensorOutput = torch.cat((pTensorOutputRNN, pTensorOutputContext), dim=-1)
        pTensorPredict = torch.nn.functional.softmax(self.fc_layer(pTensorOutput), dim=-1)
        return pTensorPredict, pTensorHidden, pTensorOutputContext, pTensorScore

    # Define a one-hot encoded labels
    def _process_one_hot(self, pTensorLabel: tensor):
        nBatch, nLength = pTensorLabel.size()
        pDevice = pTensorLabel.device
        pTensorOneHot = torch.zeros((nBatch, nLength, self.class_count)).to(pDevice)
        pTensorOneHotIdentity = torch.sparse.torch.eye(self.class_count).to(pDevice)
        for i in range(nBatch):
            pTensorOneHot[i] = pTensorOneHotIdentity.index_select(dim=0, index=pTensorLabel[i])
        return pTensorOneHot

    # Define a function for label precessing
    def _process_labeling(self, pTensorLabel: tensor):
        pTensor = torch.tensor((), dtype=pTensorLabel.dtype).to(pTensorLabel.device)
        pTensor = pTensor.new_full((len(pTensorLabel), 1), 41)  # <sos> : Start of Sentence
        pTensorLabelAttachedSOS = torch.cat((pTensor, pTensorLabel), dim=-1)
        pTensor = torch.tensor((), dtype=pTensorLabel.dtype).to(pTensorLabel.device)
        pTensor = pTensor.new_full((len(pTensorLabel), 1), 42)  # <eos> : End of Sentence
        pTensorLabelAttachedEOS = torch.cat((pTensorLabel, pTensor), dim=-1)
        return pTensorLabelAttachedSOS, pTensorLabelAttachedEOS


# Define the LAS Model
class LAS(Module):
    def __init__(self,
                 nDimInputListener: int,
                 nDimOutputListener: int,
                 nCountLayerListener: int,
                 nDimInputSpeller: int,
                 nDimOutputSpeller: int,
                 nCountClass: int,
                 nCountLayerSpeller: int):
        super(LAS, self).__init__()
        # Set sub-networks: listener, speller
        self.listener = Listener(nDimInputListener, nDimOutputListener, nCountLayerListener)
        self.speller = Speller(nDimInputSpeller, nDimOutputSpeller, nCountClass, nCountLayerSpeller)

    def forward(self,
                pTensorX: tensor,
                pTensorTarget: tensor = None,
                dLearningRate=0.9,
                bTrain=True):
        # Get high-level embeddings, H, given acoustic features, X.
        pTensorEmbedding = self.listener(pTensorX)
        # Predict output characters/phonemes using the high-level embeddings, H
        return self.speller(pTensorEmbedding, pTensorTarget, dLearningRate, bTrain)


def __process_decode(pTensorData, pTensorLabel, pListLabelLength):
    dLER = 0
    for i in range(pTensorData.size(0)):
        # Find the maximum likelihood output
        pTensorClassification = torch.argmax(pTensorData[i, :, :], dim=-1)
        # Collapse the repeated outputs  (Note : simplified implementation)
        pCollapseLabel = [pTensorClassification[j] for j in range(len(pTensorClassification))
                          if (j == 0) or pTensorClassification[j] != pTensorClassification[j - 1]]
        # Compute the edit distance between the reference and estimated ones
        pCollapseLabel = torch.tensor([nLabel for nLabel in pCollapseLabel if nLabel != 0])
        dDistance = ASRDataset.levenshetein(pCollapseLabel, pTensorLabel[i][:pListLabelLength[i]]) / pListLabelLength[i]
        dLER += dDistance
    dLER /= pTensorData.size(0)
    return dLER


def __process_train(nEpoch: int, pModel: LAS, pDataLoader: DataLoader, pCriterion, pOptimizer):
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
        pTensorOutput, pTensorScore, pTensorOutputLabel = pModel(pTensorData, pTensorTarget,
                                                                 dLearningRate=1.0, bTrain=True)
        # Compute the loss by comparing the predicted output to the reference labels
        dLoss = pCriterion(torch.log(pTensorOutput).transpose(1, 2),
                           pTensorOutputLabel[:, :pTensorOutput.size(1)])
        dTotalLoss += dLoss
        # Compute the loss by comparing the predicted output to the reference labels
        dLER = __process_decode(pTensorOutput, pTensorTarget, pListTargetLength)
        dTotalLER += dLER
        # Perform backpropagation if it is the training mode
        pOptimizer.zero_grad()
        dLoss.backward()
        pOptimizer.step()
        # Display the running progress
        pBar.set_description("Train Epoch: {} [{}/{}] LAS_Loss: {:.4f} LER: {:.4f}".
                             format(nEpoch, i, len(pDataLoader), dTotalLoss / (i + 1), dTotalLER / (i + 1)))
    return dTotalLoss, dTotalLER


def __process_test(nEpoch: int, pModel: LAS, pDataLoader: DataLoader, pCriterion):
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
            dLoss = pCriterion(pTensorOutput, pTensorTarget, pListDataLength, pListTargetLength)
            dTotalLoss += dLoss
            # Compute the loss by comparing the predicted output to the reference labels
            dLER = __process_decode(pTensorOutput, pTensorTarget, pListTargetLength)
            dTotalLER += dLER
            # Display the running progress
            pBar.set_description("Evaluate Epoch: {} [{}/{}] LAS_Loss: {:.4f} LER: {:.4f}".
                                 format(nEpoch, i, len(pDataLoader), dTotalLoss / (i + 1), dTotalLER / (i + 1)))
    return dTotalLoss, dTotalLER


def train(nEpoch: int,
          pTrainData: YoonDataset,
          pValidationData: YoonDataset,
          strModelPath='model_las.pth',
          nSizeBatch=4,
          bInitTrain=False,
          dLearningRate=0.001,
          nWorker=0,  # 0 = CPU, 4 = CUDA
          ):
    # Set the device for running the LAS model
    # Check if we can use a GPU device
    if torch.cuda.is_available():
        pDevice = torch.device('cuda')
    else:
        pDevice = torch.device('cpu')
    # Define a network architecture
    nDimHidden = 256
    nCountLayerListener = 3
    nCountLayerSpeller = 1
    nCountClass = 40 + 1 + 1 + 1  # 40 phoneme + <sos, eos, pad>
    pModel = LAS(nDimInputListener=pTrainData.get_dimension(), nDimOutputListener=nDimHidden,
                 nCountLayerListener=nCountLayerListener, nCountClass=nCountClass,
                 nDimInputSpeller=nDimHidden + nCountClass, nDimOutputSpeller=nDimHidden,
                 nCountLayerSpeller=nCountLayerSpeller)
    pModel = pModel.to(pDevice)
    # Define an optimizer
    pOptimizer = Adam(pModel.parameters(), lr=dLearningRate)
    # Define a training criterion
    pCriterion = torch.nn.NLLLoss(ignore_index=0)  # ignore <pad> labels
    # Load the pre-trained model if you resume the training from the model
    nStart = 0
    if os.path.isfile(strModelPath) and bInitTrain is False:
        pModelData = torch.load(strModelPath, map_location=pDevice)
        pModel.load_state_dict(pModelData['encoder'])
        pOptimizer.load_state_dict(pModelData['optimizer'])
        nStart = pModelData['epoch']
        print("## Success to load the LAS model : epoch {}".format(nStart))
    # Define training and test dataset
    pTrainDataset = ASRDataset(pTrainData)
    pTrainLoader = DataLoader(pTrainDataset, batch_size=nSizeBatch, collate_fn=collate_asrdata, shuffle=True,
                              num_workers=nWorker, pin_memory=True)
    pValidDataset = ASRDataset(pValidationData)
    pValidLoader = DataLoader(pValidDataset, batch_size=nSizeBatch, collate_fn=collate_asrdata, shuffle=False,
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

