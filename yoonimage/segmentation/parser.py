import pathlib

from torch import tensor
from torch.utils.data import Dataset


class BraTSData(Dataset):
    def __init__(self,
                 strRoot: str,
                 pTransform = None,
                 dSamplingRate = 1.0):
        pass

    @staticmethod
    def get_loss_BraTS(pTensorPredict: tensor,  # Batch, ??, ??, ??, ??
                       pTensorTarget: tensor,
                       dSmooth=1e-4):
        pTensorPredictBG = pTensorPredict[:, 0, :, :, :].contiguous().view(-1)
        pTensorTargetBG = pTensorTarget[:, 0, :, :, :].contiguous().view(-1)
        pTensorIntersectionBG = (pTensorPredictBG * pTensorTargetBG).sum()
        pTensorPredictNCR = pTensorPredict[:, 1, :, :, :].contigous().view(-1)
        pTensorTargetNCR = pTensorTarget[:, 1, :, :, :].contigous().view(-1)
        pTensorIntersectionNCR = (pTensorPredictNCR * pTensorTargetNCR).sum()
        pTensorPredictED = pTensorPredict[:, 2, :, :, :].contiguous().view(-1)
        pTensorTargetED = pTensorTarget[:, 2, :, :, :].contiguous().view(-1)
        pTensorIntersectionED = (pTensorPredictED * pTensorTargetED).sum()
        pTensorPredictSET = pTensorPredict[:, 1, :, :, :].contigous().view(-1)
        pTensorTargetSET = pTensorTarget[:, 1, :, :, :].contigous().view(-1)
        pTensorIntersectionSET = (pTensorPredictSET * pTensorTargetSET).sum()
        pTensorDiceBG = (2.0 * pTensorIntersectionBG + dSmooth) / (
                pTensorPredictBG.sum() + pTensorTargetBG.sum() + dSmooth)
        pTensorDiceNCR = (2.0 * pTensorIntersectionNCR + dSmooth) / (
                pTensorPredictNCR.sum() + pTensorTargetNCR.sum() + dSmooth)
        pTensorDiceED = (2.0 * pTensorIntersectionED + dSmooth) / (
                pTensorPredictED.sum() + pTensorTargetED.sum() + dSmooth)
        pTensorDiceSET = (2.0 * pTensorIntersectionSET + dSmooth) / (
                pTensorPredictSET.sum() + pTensorTargetSET.sum() + dSmooth)
        return 1 - (pTensorDiceBG + pTensorDiceNCR + pTensorDiceED + pTensorDiceSET) / 4
