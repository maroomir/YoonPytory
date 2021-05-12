import os
import torch
import torch.nn
import numpy
import matplotlib.pyplot
from tqdm import tqdm
from torch import tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import Module
from torch.nn import CrossEntropyLoss
from sklearn.manifold import TSNE
from yoonspeech.data import YoonDataset
from yoonspeech.speech import YoonSpeech


# Define a SpeakerDataset class
class SpeakerDataset(Dataset):
    def __init__(self,
                 pDataset: YoonDataset):
        self.data = pDataset

    def __len__(self):  # Return Dataset length to decision data-loader size
        return self.data.__len__()

    def __getitem__(self, item: int):  # obtain label and file name
        nTarget = self.data[item].label
        pArrayInput = self.data[item].buffer
        return pArrayInput, nTarget


class DVector(Module):
    def __init__(self,
                 pDataset: YoonDataset):
        super(DVector, self).__init__()
        # Set the number of speakers to classify
        nDimInput = pDataset[0].get_dimension()
        nDimOutput = pDataset.class_count
        # Set 4 layers of feed-forward network (FFN) (+Activation)
        self.network = torch.nn.Sequential(torch.nn.Linear(nDimInput, 512),
                                           torch.nn.LeakyReLU(negative_slope=0.2),
                                           torch.nn.Linear(512, 512),
                                           torch.nn.LeakyReLU(negative_slope=0.2),
                                           torch.nn.Linear(512, 512),
                                           torch.nn.LeakyReLU(negative_slope=0.2),
                                           torch.nn.Linear(512, 512),
                                           torch.nn.LeakyReLU(negative_slope=0.2))
        # Set a classification layer
        self.classification_layer = torch.nn.Linear(512, nDimOutput)

    def forward(self, pTensorX: tensor, bExtract=False):  # override
        # Normalize input features (zero mean and unit variance).
        pXMean = torch.mean(pTensorX, -1)  # Mean of One frame (sum / deltas)
        pXStd = torch.std(pTensorX, -1)  # Std of One frame (sum / deltas)
        pXStd[pXStd < 0.01] = 0.01
        pTensorX = (pTensorX - pXMean[:, :, None]) / pXStd[:, :, None]
        # Pass FFN Layers
        pTensorResult = self.network(pTensorX)
        # Use a mean pooling approach to obtain a D-Vector
        pTensorResult = pTensorResult.mean(dim=1)  # Mean of frames (Batch * CH, Frame, deltas)
        # Perform a classification task in the training process
        if bExtract:
            pTensorResult = self.classification_layer(pTensorResult)
        return pTensorResult


# Define a collate function for the data loader (Assort for Batch)
def collate_tensor(pListTensor):
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


# Define a train function
def __process_train(nEpoch: int, pModel: DVector, pDataLoader: DataLoader, pCriterion: CrossEntropyLoss,
                    pOptimizer: Adam):
    # Check if we can use a GPU Device
    if torch.cuda.is_available():
        pDevice = torch.device('cuda')
    else:
        pDevice = torch.device('cpu')
    # Perform a training using the defined network
    pModel.train()
    # Warp the iterable Data Loader with TQDM
    pBar = tqdm(enumerate(pDataLoader))
    nLengthSample = 0
    dTotalLoss = 0
    nTotalAcc = 0
    for i, (pTensorInput, pTensorTarget) in pBar:
        # Move data and label to device
        pTensorInput = pTensorInput.type(torch.FloatTensor).to(pDevice)
        pTensorTarget = pTensorTarget.to(pDevice)
        # Pass the input data through the defined network architecture
        pPredictedLabel = pModel(pTensorInput, bExtract=True)  # Module
        # Compute a loss function
        pLoss = pCriterion(pPredictedLabel, pTensorTarget)
        dTotalLoss += pLoss.item() * len(pTensorTarget)  # Channel
        # Compute speaker recognition accuracy
        nAcc = torch.sum(torch.eq(torch.argmax(pPredictedLabel, -1), pTensorTarget)).item()
        nLengthSample += len(pTensorTarget)
        nTotalAcc += nAcc
        # Perform backpropagation to update network parameters
        pOptimizer.zero_grad()
        pLoss.backward()
        pOptimizer.step()
        pBar.set_description('Epoch:{:3d} [{}/{} {:.2f}%] CE Loss: {:.3f} ACC: {:.2f}%'
                             .format(nEpoch, i, len(pDataLoader), 100.0 * (i / len(pDataLoader)),
                                     dTotalLoss / nLengthSample, (nTotalAcc / nLengthSample) * 100.0))


# Define a test function
def __process_test(pModel: DVector, pDataLoader: DataLoader, pCriterion: CrossEntropyLoss):
    # Check if we can use a GPU Device
    if torch.cuda.is_available():
        pDevice = torch.device('cuda')
    else:
        pDevice = torch.device('cpu')
    # Perform an evaluation using the defined network
    pModel.eval()
    # Warp the iterable Data Loader with TQDM
    pBar = tqdm(enumerate(pDataLoader))
    nLengthSample = 0
    nTotalLoss = 0
    nTotalAcc = 0
    for i, (pTensorInput, pTensorTarget) in pBar:
        # Move data and label to device
        pTensorInput = pTensorInput.type(torch.FloatTensor).to(pDevice)
        pTensorTarget = pTensorTarget.to(pDevice)
        # Pass the input data through the defined network architecture
        pPredictedLabel = pModel(pTensorInput, bExtract=True)  # Module
        # Compute a loss function
        pLoss = pCriterion(pPredictedLabel, pTensorTarget)
        nTotalLoss += pLoss.item() * len(pTensorTarget)
        # Compute speaker recognition accuracy
        nAcc = torch.sum(torch.eq(torch.argmax(pPredictedLabel, -1), pTensorTarget)).item()
        nLengthSample += len(pTensorTarget)
        nTotalAcc += nAcc
    return nTotalLoss / nLengthSample


# Define a t-SNE Plot function
def __draw_tSNE(pTensorTSNE,
                pColorLabel: numpy.ndarray):
    pColorMap = matplotlib.pyplot.cm.rainbow
    matplotlib.pyplot.scatter(pTensorTSNE[:, 0], pTensorTSNE[:, 1], s=4, c=pColorLabel, cmap=pColorMap)
    matplotlib.pyplot.xlim(min(pTensorTSNE[:, 0]), max(pTensorTSNE[:, 0]))
    matplotlib.pyplot.ylim(min(pTensorTSNE[:, 1]), max(pTensorTSNE[:, 1]))
    matplotlib.pyplot.title('D-Vector t-SNE')
    matplotlib.pyplot.show()


def train(nEpoch: int, pTrainData: YoonDataset, pValidationData: YoonDataset, strModelPath: str = None,
          bInitEpoch=False):
    dLearningRate = 0.01
    # Check if we can use a GPU Device
    if torch.cuda.is_available():
        pDevice = torch.device('cuda')
    else:
        pDevice = torch.device('cpu')
    print("{} device activation".format(pDevice.__str__()))
    # Define the training and testing data-set
    pTrainSet = SpeakerDataset(pTrainData)
    pTrainLoader = DataLoader(pTrainSet, batch_size=8, shuffle=True, collate_fn=collate_tensor,
                              num_workers=0, pin_memory=True)
    pValidationSet = SpeakerDataset(pValidationData)
    pValidationLoader = DataLoader(pValidationSet, batch_size=1, shuffle=False, collate_fn=collate_tensor,
                                   num_workers=0, pin_memory=True)
    # Define a network model
    pModel = DVector(pTrainData).to(pDevice)
    # Set the optimizer with adam
    pOptimizer = torch.optim.Adam(pModel.parameters(), lr=dLearningRate)
    # Set the training criterion
    pCriterion = torch.nn.CrossEntropyLoss(reduction='mean')
    # Load pre-trained model
    nStart = 0
    print("Directory of the pre-trained model: {}".format(strModelPath))
    if strModelPath is not None and os.path.exists(strModelPath) and bInitEpoch is False:
        pModelData = torch.load(strModelPath)
        nStart = pModelData['epoch']
        pModel.load_state_dict(pModelData['model'])
        pOptimizer.load_state_dict(pModelData['optimizer'])
        print("## Successfully load the model at {} epochs!".format(nStart))
    # Train and Test Repeat
    dMinLoss = 10000.0
    nCountDecrease = 0
    for iEpoch in range(nStart, nEpoch + 1):
        # Train the network
        __process_train(iEpoch, pModel=pModel, pDataLoader=pTrainLoader, pCriterion=pCriterion,
                        pOptimizer=pOptimizer)
        # Test the network
        dLoss = __process_test(pModel=pModel, pDataLoader=pValidationLoader, pCriterion=pCriterion)
        # Save the optimal model
        if dLoss < dMinLoss:
            dMinLoss = dLoss
            torch.save({'epoch': iEpoch, 'model': pModel.state_dict(), 'optimizer': pOptimizer.state_dict()},
                       strModelPath)
            nCountDecrease = 0
        else:
            nCountDecrease += 1
            # Decrease the learning rate by 2 when the test loss decrease 3 times in a row
            if nCountDecrease == 3:
                pDicOptimizerState = pOptimizer.state_dict()
                pDicOptimizerState['param_groups'][0]['lr'] /= 2
                pOptimizer.load_state_dict(pDicOptimizerState)
                print('learning rate is divided by 2')
                nCountDecrease = 0


def test(pTestData: YoonDataset, strModelPath: str = None):
    # Check if we can use a GPU device
    if torch.cuda.is_available():
        pDevice = torch.device('cuda')
    else:
        pDevice = torch.device('cpu')
    print("{} device activation".format(pDevice.__str__()))
    # Load DVector model
    pModel = DVector(pTestData).to(pDevice)  # Check train data
    pModel.eval()
    pFile = torch.load(strModelPath)
    pModel.load_state_dict(pFile['model'])
    print("Successfully load the Model in path")
    # Define a data path for plot for test
    pDataSet = SpeakerDataset(pTestData)
    pDataLoader = DataLoader(pDataSet, batch_size=1, shuffle=False, collate_fn=collate_tensor,
                             num_workers=0, pin_memory=True)
    pBar = tqdm(pDataLoader)
    print("Length of data = ", len(pBar))
    pListOutput = []
    pListTarget = []
    for i, (pTensorInput, pTensorTarget) in enumerate(pBar):
        pTensorInput = pTensorInput.type(torch.FloatTensor).to(pDevice)
        pTensorOutput = pModel(pTensorInput, bExtract=False)
        pListOutput.append(pTensorOutput.detach().cpu().numpy())
        pListTarget.append(pTensorTarget.detach().cpu().numpy()[0])  # (Batch, Label) to (Label)
    # Prepare embeddings for plot
    pArrayOutput = numpy.concatenate(pListOutput)
    pArrayTarget = numpy.array(pListTarget)
    # Obtain embedding for the t-SNE plot
    pTSNE = TSNE(n_components=2)
    pArrayOutput = pArrayOutput.reshape(len(pArrayOutput), -1)
    pArrayTSNE = pTSNE.fit_transform(pArrayOutput)
    # Draw plot
    __draw_tSNE(pArrayTSNE, pArrayTarget)


def recognition(pSpeech: YoonSpeech, nCountClass: int, strModelPath: str = None, strFeatureType: str = "deltas"):
    # Warp data set
    pTestData = YoonDataset(nCountClass, strFeatureType, None, pSpeech)
    # Check if we can use a GPU device
    if torch.cuda.is_available():
        pDevice = torch.device('cuda')
    else:
        pDevice = torch.device('cpu')
    # Load DVector model
    pModel = DVector(pTestData).to(device=pDevice)
    pModel.eval()
    pFile = torch.load(strModelPath)
    pModel.load_state_dict(pFile['model'])
    # Recognition model
    pTensorInput = torch.from_numpy(pTestData[0].buffer).to(pDevice).unsqueeze(0)
    pArrayOutput = pModel(pTensorInput, bExtract=True).detach().cpu().numpy()
    nLabelEstimated = numpy.argmax(pArrayOutput, -1)  # Index of maximum of output layer
    print("Estimated: {0}, Score : {1:.2f}".format(nLabelEstimated, numpy.max(pArrayOutput)))
    return nLabelEstimated
