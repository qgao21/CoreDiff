import torch
import torch.nn as nn
import torchvision.models as models


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        model = models.vgg16(pretrained=True).cuda()
        model.eval()

        for param in model.parameters():
            param.requires_grad = False
        
        # self.blocks_1 = nn.Sequential(*list(networks.children()))[0][:4]
        self.blocks_1 = nn.Sequential(*list(model.children()))[0][:8]
        self.blocks_2 = nn.Sequential(*list(model.children()))[0][:15]
        self.blocks_3 = nn.Sequential(*list(model.children()))[0][:22]
        
        self.extractors = [self.blocks_1, self.blocks_2, self.blocks_3]
        self.weights = [0.5, 0.5, 0.5]

    def forward(self, x, y):
        x = torch.cat((x, x, x), 1)
        y = torch.cat((y, y, y), 1)
        loss = torch.zeros((1, 1)).cuda()
        for extractor, weight in zip(self.extractors, self.weights):
            loss += weight * (torch.mean(torch.abs(extractor(x) - extractor(y))))
        return loss