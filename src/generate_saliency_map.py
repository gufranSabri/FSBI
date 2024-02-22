import torch
import numpy as np
from torch import nn
import sys
from model import *
from utils.sbi import SBI_Dataset
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

device = torch.device('mps')

model=Detector()
cnn_sd=torch.load("/Users/gufran/Developer/Projects/AI/DeepFakeDetection/weights/4_0.9991_val.tar", map_location=torch.device('cpu'))["model"]
model.load_state_dict(cnn_sd)
model=model.to(device)

image_size = 380

train_dataset=SBI_Dataset(phase='train',image_size=image_size)
val_dataset=SBI_Dataset(phase='val',image_size=image_size)

train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=32//2,shuffle=True,collate_fn=train_dataset.collate_fn,num_workers=4,pin_memory=True,drop_last=True,worker_init_fn=train_dataset.worker_init_fn)
val_loader=torch.utils.data.DataLoader(val_dataset,batch_size=32,shuffle=False,collate_fn=val_dataset.collate_fn,num_workers=4,pin_memory=True,worker_init_fn=val_dataset.worker_init_fn)

for batch in train_loader:
    images, _ = batch
    break

print(images.shape)
    
class ModifiedCNN(nn.Module):
    def __init__(self, original_model):
        super(ModifiedCNN, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-5])
                
    def forward(self, x):
        return self.features(x)

modified_model = ModifiedCNN(model).to(device)
modified_model.eval()

#make random tensor
# images = torch.rand(1, 3, 380, 380)

with torch.no_grad():
    activations = modified_model(images.to(device))

print(activations.shape)

last_conv_layer_activations = activations[0]
heatmap = torch.mean(last_conv_layer_activations, dim=0)
heatmap = heatmap.cpu().numpy()
heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))

plt.imshow(heatmap, cmap='viridis')
plt.axis('off')
plt.show()
