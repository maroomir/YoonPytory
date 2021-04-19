import os
import torch
import torch.nn
import numpy
import matplotlib
from tqdm import tqdm
from torch import tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import Module
from torch.nn import CrossEntropyLoss
from sklearn.manifold import TSNE
from yoonspeech.speakerRecognition.parser import YoonParser


# Define a SpeakerDataset class
class SpeakerDataset(Dataset):
    def __init__(self,
                 pParser: YoonParser,
                 strType: str = "train"):
        self.parser = pParser
        self.type = strType

    def __len__(self):
        if self.type == "train":
            return self.parser.get_train_count()
        elif self.type == "test":
            return self.parser.get_test_count()
        else:
            Exception("Speaker Dataset type is mismatching")

    def __getitem__(self, item):  # obtain label and file name
        if self.type == "train":
            nLabel = self.parser.get_train_label(item)
            pArrayFeature = self.parser.get_train_data(item)
            return nLabel, pArrayFeature
        elif self.type == "test":
            nLabel = self.parser.get_test_label(item)
            pArrayFeature = self.parser.get_test_data(item)
            return nLabel, pArrayFeature
        else:
            Exception("Speaker Dataset type is mismatching")


class DVector(Module):
    def __init__(self,
                 pParser: YoonParser,
                 strType: str = "train"):
        super(DVector, self).__init__()
        # Set the number of speakers to classify
        if strType == "train":
            nCountSpeaker = pParser.get_train_count()
        elif strType == "test":
            nCountSpeaker = pParser.get_test_count()
        else:
            Exception("DVector type is mismatching")
        nDimInput = pParser.get_data_dimension()
        nDimOutput = pParser.fftCount
        self.speakersCount = nCountSpeaker
        # Set 4 layers of feed-forward network (FFN) (+Activation)
        self.network = torch.nn.Sequential(torch.nn.Linear(nDimInput, nDimOutput),
                                           torch.nn.LeakyReLU(negative_slope=0.2),
                                           torch.nn.Linear(nDimOutput, nDimOutput),
                                           torch.nn.LeakyReLU(negative_slope=0.2),
                                           torch.nn.Linear(nDimOutput, nDimOutput),
                                           torch.nn.LeakyReLU(negative_slope=0.2),
                                           torch.nn.Linear(nDimOutput, nDimOutput),
                                           torch.nn.LeakyReLU(negative_slope=0.2))
        # Set a classification layer
        self.classificationLayer = torch.nn.Linear(nDimOutput, self.speakersCount)

    def forward(self, pTensorX: tensor, bExtract=False):  # override
        # Normalize input features (zero mean and unit variance).
        pXMean = torch.mean(pTensorX, -1)
        pXStd = torch.std(pTensorX, -1)
        pXStd[pXStd < 0.01] = 0.01
        pTensorX = (pTensorX - pXMean[:, :, None]) / pXStd[:, :, None]
        # Pass FFN Layers
        pTensorResult = self.network(pTensorX)
        # Use a mean pooling approach to obtain a D-Vector
        pTensorResult = pTensorResult.mean(dim=1)
        # Perform a classification task in the training process
        if bExtract:
            pTensorResult = self.classificationLayer(pTensorResult)
        return pTensorResult


# Define a collate function for the data loader
def collate_tensor(pListTensor):
    pListData = []
    pListLabel = []
    nLengthMin = min([len(pTuple[1]) for pTuple in pListTensor]) - 1
    for nLabel, pArrayData in pListTensor:
        nStart = numpy.random.randint(len(pArrayData) - nLengthMin)
        pListData.append(torch.tensor(pArrayData[nStart:nStart + nLengthMin]).unsqueeze(0))
        pListLabel.append(torch.tensor(nLabel))
    pListData = torch.cat(pListData, 0)
    pListLabel = torch.LongTensor(pListLabel)
    return pListLabel, pListData


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
    nTotalLoss = 0
    nTotalAcc = 0
    for i, (pTensorLabel, pTensorData) in pBar:
        # Move data and label to device
        pTensorData = pTensorData.type(torch.FloatTensor).to(pDevice)
        pTensorLabel = pTensorLabel.to(pDevice)
        # Pass the input data through the defined network architecture
        pPrediction = pModel(pTensorData, bExtract=True)  # Module
        # Compute a loss function
        pLoss = pCriterion(pPrediction, pTensorLabel)
        nTotalLoss += pLoss.item() * len(pTensorLabel)
        # Compute speaker recognition accuracy
        nAcc = torch.sum(torch.eq(torch.argmax(pPrediction, -1), pTensorLabel)).item()
        nLengthSample += len(pTensorLabel)
        nTotalAcc += nAcc
        # Perform backpropagation to update network parameters
        pOptimizer.zero_grad()
        pLoss.backward()
        pOptimizer.step()
        pBar.set_description('Epoch:{:3d} [{}/{} {:.2f}%] CE Loss: {:.3f} ACC: {:.2f}%'
                             .format(nEpoch, i, len(pDataLoader), 100.0 * (i / len(pDataLoader)),
                                     nTotalLoss / nLengthSample, (nTotalAcc / nLengthSample) * 100.0))


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
    for i, (pTensorLabel, pTensorData) in pBar:
        # Move data and label to device
        pTensorData = pTensorData.type(torch.FloatTensor).to(pDevice)
        pTensorLabel = pTensorLabel.to(pDevice)
        # Pass the input data through the defined network architecture
        pPrediction = pModel(pTensorData, bExtract=True)  # Module
        # Compute a loss function
        pLoss = pCriterion(pPrediction, pTensorLabel)
        nTotalLoss += pLoss.item() * len(pTensorLabel)
        # Compute speaker recognition accuracy
        nAcc = torch.sum(torch.eq(torch.argmax(pPrediction, -1), pTensorLabel)).item()
        nLengthSample += len(pTensorLabel)
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


def train(nEpoch: int, pParser: YoonParser, strModelPath: str = None):
    dLearningRate = 0.01
    # Check if we can use a GPU Device
    if torch.cuda.is_available():
        pDevice = torch.device('cuda')
    else:
        pDevice = torch.device('cpu')
    print("{} device activation".format(pDevice.__str__()))
    # Define the training and testing data-set
    pTrainSet = SpeakerDataset(pParser, "train")
    pTrainLoader = DataLoader(pTrainSet, batch_size=8, shuffle=True, collate_fn=collate_tensor,
                              num_workers=0, pin_memory=True)
    pTestSet = SpeakerDataset(pParser, "test")
    pTestLoader = DataLoader(pTestSet, batch_size=8, shuffle=True, collate_fn=collate_tensor,
                             num_workers=0, pin_memory=True)
    # Define a network model
    pModel = DVector(pParser, "train").to(pDevice)
    # Set the optimizer with adam
    pOptimizer = torch.optim.Adam(pModel.parameters(), lr=dLearningRate)
    # Set the training criterion
    pCriterion = torch.nn.CrossEntropyLoss(reduction='mean')
    # Load pre-trained model
    nStart = 0
    print("Directory of the pre-trained model: {}".format(strModelPath))
    if strModelPath is not None and os.path.exists(strModelPath):
        pTensorModelData = torch.load(strModelPath)
        nStart = pTensorModelData['epoch']
        pModel.load_state_dict(pTensorModelData['model'])
        pOptimizer.load_state_dict(pTensorModelData['optimizer'])
        print("## Successfully load the model at {} epochs!".format(nStart))
    # Train and Test Repeat
    dMinLoss = 10000.0
    nCountDecrease = 0
    for iEpoch in range(nStart, nEpoch + 1):
        # Train the network
        __process_train(iEpoch, pModel=pModel, pDataLoader=pTrainLoader, pCriterion=pCriterion,
                        pOptimizer=pOptimizer)
        # Test the network
        dLoss = __process_test(pModel=pModel, pDataLoader=pTestLoader, pCriterion=pCriterion)
        # Save the optimal model
        if dLoss < dMinLoss:
            dMinLoss = dLoss
            torch.save({'epoch': iEpoch, 'model': pModel.state_dict(), 'optimizer': pOptimizer.state_dict()},
                       './model_opt.pth')
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


def test(pParser: YoonParser, strModelPath: str = None):
    # Check if we can use a GPU device
    if torch.cuda.is_available():
        pDevice = torch.device('cuda')
    else:
        pDevice = torch.device('cpu')
    print("{} device activation".format(pDevice.__str__()))
    # Load DVector model
    pModel = DVector(pParser, "test").to(pDevice)
    pModel.eval()
    pFile = torch.load(strModelPath)
    pModel.load_state_dict(pFile['model'])
    print("Successfully load the Model in path")
    # Define a data path for plot for test
    pDataSet = SpeakerDataset(pParser, "test")
    pDataLoader = DataLoader(pDataSet, batch_size=8, shuffle=True, collate_fn=collate_tensor,
                             num_workers=0, pin_memory=True)
    pBar = tqdm(pDataLoader)
    print(", Length of data = ", len(pBar))
    pListData = []
    pListLabel = []
    for i, (pTensorLabel, pTensorData) in enumerate(pBar):
        pTensorData = pTensorData.type(torch.FloatTensor).to(pDevice)
        pOutput = pModel(pTensorData, bExtract=False)
        pListData.append(pOutput.detach().cpu().numpy())
        pListLabel.append(pTensorLabel.detach().cpu().numpy()[0])
    # Prepare embeddings for plot
    pArrayData = numpy.concatenate(pListData)
    pArrayLabel = numpy.array(pListLabel)
    # Obtain embedding for the t-SNE plot
    pTSNE = TSNE(n_components=2)
    pArrayData = pArrayData.reshape(len(pArrayData), -1)
    pArrayTSNE = pTSNE.fit_transform(pArrayData)
    # Draw plot
    __draw_tSNE(pArrayTSNE, pArrayLabel)
