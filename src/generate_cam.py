import torch
import torch.nn as nn
import numpy as np
from utils.sbi import SBI_Dataset
from tqdm import tqdm
from model import *
import cv2
from torchvision import transforms


class CAMGenerator:
  def __init__(self, model, target_layer):
      self.model = model
      self.target_layer = target_layer
      self.gradients = None
      self.activations = None

      # Register hook to get gradients and activations
      self.model.net._conv_head.register_forward_hook(self.save_activation)
      self.model.net._conv_head.register_backward_hook(self.save_gradient)

  def save_activation(self, module, input, output):
      self.activations = output

  def save_gradient(self, module, grad_input, grad_output):
      self.gradients = grad_output[0]

  def generate_cam(self, input_tensor, class_idx):
      self.model.eval()
      self.model.zero_grad()

      # Forward pass to get activations
      output = self.model.net(input_tensor)
      score = output[:, class_idx]

      # Backward pass to get gradients
      score.backward()

      # Global average pooling on gradients
      weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

      # Weighted sum of activations
      cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
      cam = nn.functional.relu(cam, inplace=True)

      # Upsample to input size
      cam = nn.functional.interpolate(cam, size=(input_tensor.size(2), input_tensor.size(3)), mode="bilinear", align_corners=False)

      # Normalize between 0 and 1
      cam_min, cam_max = cam.min(), cam.max()
      cam = (cam - cam_min).div(cam_max - cam_min).squeeze()

      return cam.cpu().detach().numpy()

def overlay_cam(image, cam):
  heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
  heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

  overlayed_img = 0.5 * image + 0.5 * heatmap
  overlayed_img = (overlayed_img * 255).astype(np.uint8)

  return overlayed_img

def main():
  device = torch.device('mps')

  image_size=380
  batch_size=16

  train_dataset=SBI_Dataset(phase='train',image_size=image_size)
  
  train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size//2,shuffle=True,collate_fn=train_dataset.collate_fn,num_workers=4,pin_memory=True,drop_last=True,worker_init_fn=train_dataset.worker_init_fn)
  
  model=Detector()
  cnn_sd=torch.load("/Users/gufran/Developer/Projects/AI/DeepFakeDetection/weights/4_0.9991_val.tar", map_location=torch.device('cpu'))["model"]
  model.load_state_dict(cnn_sd)
  model=model.to(device)
  model.eval()

  for _,data in enumerate((train_loader)):
      img=data['img'].to(device, non_blocking=True).float()[8:]
      
      class_index = 1
      cam_generator = CAMGenerator(model, target_layer=model.net._conv_head)

      count = 0
      for i in img:
        original_image = i.cpu().detach().numpy().squeeze().transpose((1, 2, 0))
        #convert to RGB
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        cam = cam_generator.generate_cam(i.unsqueeze(0), class_index)

        count+=1
        overlayed_image = overlay_cam(original_image, cam)

        original_image *= 255

        cv2.imwrite(f"./fig/cam/original{count}.jpg", original_image)
        cv2.imwrite(f"./fig/cam/cam{count}.jpg", overlayed_image)

      exit()
        
if __name__=='__main__':
  main()
        
