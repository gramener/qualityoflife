import torch
import torchvision
import torch.nn.functional as F
from torch import nn, optim
from torchvision import models

from srm_single_dataset import get_data

device = torch.device('cuda')


class Flatten(nn.Module):
    def forward(self, xb):
        return xb.view(xb.size(0), -1)


class SentinelResNet(nn.Module):
    def __init__(self, M, N, targets):
        super().__init__()
        self.rgb_features = nn.Sequential(*(list(M.children())[:-2] + [nn.AdaptiveAvgPool2d(1)]))
        self.nir_features = nn.Sequential(*(list(N.children())[:-2] + [nn.AdaptiveAvgPool2d(1)]))

        self.classifier = nn.Sequential(
            Flatten(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.wealth = nn.Linear(256, len(targets['agriculture_land_bin']['classes']))
        # self.water_src = nn.Linear(256, len(targets['water_src']['classes']))
        # self.toilet_type = nn.Linear(256, len(targets['toilet_type']['classes']))
        # self.roof = nn.Linear(256, len(targets['roof']['classes']))
        # self.cooking_fuel = nn.Linear(256, len(targets['cooking_fuel']['classes']))
        # self.drought = nn.Linear(256, len(targets['drought']['classes']))
        # self.pop_density = nn.Linear(256, len(targets['pop_density']['classes']))
        # self.livestock_bin = nn.Linear(256, len(targets['livestock_bin']['classes']))
        # self.agriculture_land_bin = nn.Linear(256, len(targets['agriculture_land_bin']['classes']))

    def forward(self, xb):
        rgb_out = self.rgb_features(xb[0])
        nir_out = self.nir_features(xb[1])
        out = torch.cat([rgb_out, nir_out], dim=1)

        out = self.classifier(out)

        wealth = self.wealth(out)
        # water_src = self.water_src(out)
        # toilet_type = self.toilet_type(out)
        # roof = self.roof(out)
        # cooking_fuel = self.cooking_fuel(out)
        # drought = self.drought(out)
        # pop_density = self.pop_density(out)
        # livestock_bin = self.livestock_bin(out)
        # agriculture_land_bin = self.agriculture_land_bin(out)

        return wealth

class SentinelDenseNet(nn.Module):
    def __init__(self, M, metric, targets):
        super().__init__()
        self.features = M.features

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.wealth = nn.Linear(256, len(targets[metric]['classes']))
        # self.water_src = nn.Linear(256, len(targets['water_src']['classes']))
        # self.toilet_type = nn.Linear(256, len(targets['toilet_type']['classes']))
        # self.roof = nn.Linear(256, len(targets['roof']['classes']))
        # self.cooking_fuel = nn.Linear(256, len(targets['cooking_fuel']['classes']))
        # self.drought = nn.Linear(256, len(targets['drought']['classes']))
        # self.pop_density = nn.Linear(256, len(targets['pop_density']['classes']))
        # self.livestock_bin = nn.Linear(256, len(targets['livestock_bin']['classes']))
        # self.agriculture_land_bin = nn.Linear(256, len(targets['agriculture_land_bin']['classes']))

    def forward(self, xb):
        xb = self.features(xb)
        xb = self.classifier(xb)

        wealth = self.wealth(xb)
        # water_src = self.water_src(xb)
        # toilet_type = self.toilet_type(xb)
        # roof = self.roof(xb)
        # cooking_fuel = self.cooking_fuel(xb)
        # drought = self.drought(xb)
        # pop_density = self.pop_density(xb)

        return wealth


class Model:
    def __init__(self, M):
        self.model = M

    def __call__(self, xb):
        return self.model(xb)

    @staticmethod
    def freeze(L):
        for p in L.parameters(): p.requires_grad_(False)

    @staticmethod
    def unfreeze(L):
        for p in L.parameters(): p.requires_grad_(True)

    def freeze_features(self, arg=True):
        if arg:
            Model.freeze(self.model.features)
        else:
            Model.unfreeze(self.model.features)

    def freeze_classifier(self, arg=True):
        Model.freeze(self.model.classifier) if arg else Model.unfreeze(self.model.classifier)

    def partial_freeze_features(self, pct=0.2):
        sz = len(list(self.model.features.children()))
        point = int(sz * pct)

        for idx, child in enumerate(self.model.features.children()):
            Model.freeze(child) if idx <= point else Model.unfreeze(child)

    def summary(self):
        print('\n\n')
        for idx, (name, child) in enumerate(self.model.features.named_children()):
            print(f'{idx}: {name}-{child}')
            for param in child.parameters():
                print(f'{param.requires_grad}')

        for idx, (name, child) in enumerate(self.model.classifier.named_children()):
            print(f'{idx}: {name}-{child}')
            for param in child.parameters():
                print(f'{param.requires_grad}')
        print('\n\n')

    @property
    def grads(self):
        return ''.join(str(v.requires_grad)[0].upper() for k,v in self.model.named_parameters())


def get_model(metric):
    d121 = models.densenet121(pretrained=True)

    model = SentinelDenseNet(d121,metric, get_data(metric).targets)
    wrapper = Model(model)

    return wrapper
