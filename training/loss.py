import torch
import torchmetrics
import torch.nn as nn
import torchvision.models as models
from torchmetrics.image import StructuralSimilarityIndexMeasure

class SSIMLoss(torch.nn.Module):
    def __init__(self, data_range=1.0, channel=1):
        super(SSIMLoss, self).__init__()
        self.ssim = torchmetrics.StructuralSimilarityIndexMeasure(data_range=data_range, channel=channel)

    def forward(self, predicted, target):
        ssim_score = self.ssim(predicted, target)
        return 1 - ssim_score  # Convert similarity to loss
    
class MSESSIMLoss(torch.nn.Module):
    def __init__(self, data_range=1.0, channel=1):
        super(MSESSIMLoss, self).__init__()
        self.mse = torch.nn.MSELoss()
        self.ssim = SSIMLoss(data_range=data_range, channel=channel)

    def forward(self, predicted, target):
        mse_loss = self.mse(predicted, target)
        ssim_loss = self.ssim(predicted, target)
        return mse_loss + ssim_loss

class PerceptualLoss(nn.Module):
    def __init__(self, layers=['3', '8', '15', '22'], device='cuda'):
        super(PerceptualLoss, self).__init__()
        self.layers = layers
        self.device = device
        self.vgg = models.vgg16(pretrained=True).features[:23].to(self.device).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, predicted, target):
        predicted_features = self.vgg(predicted)
        target_features = self.vgg(target)
        return nn.functional.l1_loss(predicted_features, target_features)
    
    # training/losses.py
