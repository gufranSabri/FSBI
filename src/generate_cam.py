import torch
import torch.nn as nn
import numpy as np
from utils.sbi import SBI_Dataset
from model import *
import cv2
import os
import random

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

def overlay_cam(image, cam, blur_radius=5, intensity=0.2):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    # Invert colors
    heatmap = cv2.bitwise_not(heatmap)

    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Apply Gaussian blur to smooth out the heatmap
    heatmap = cv2.GaussianBlur(heatmap, (blur_radius, blur_radius), 0)

    # Normalize between 0 and 1 after applying blur
    heatmap_min, heatmap_max = heatmap.min(), heatmap.max()
    heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)

    # Adjust intensity to make low attention areas very dark in the original image
    darkened_image = image * (1 - intensity)

    overlayed_img = 0.5 * darkened_image + 0.5 * heatmap
    overlayed_img = (overlayed_img * 255).astype(np.uint8)

    return overlayed_img


def get_dwt_rgb(img):
  b, g, r = cv2.split(img)

  cA_r, (cH_r, cV_r, cD_r) = pywt.dwt2(r, "sym2", mode="reflect")
  cA_g, (cH_g, cV_g, cD_g) = pywt.dwt2(g, "sym2", mode="reflect")
  cA_b, (cH_b, cV_b, cD_b) = pywt.dwt2(b, "sym2", mode="reflect")

  cA_r = cv2.resize(cA_r, (380, 380), interpolation=cv2.INTER_LINEAR).astype('float32')
  cA_g = cv2.resize(cA_g, (380, 380), interpolation=cv2.INTER_LINEAR).astype('float32')
  cA_b = cv2.resize(cA_b, (380, 380), interpolation=cv2.INTER_LINEAR).astype('float32')

  cA_r = (cA_r + r)/2
  cA_g = (cA_g + g)/2
  cA_b = (cA_b + b)/2

  img_dwt = np.array([cA_r, cA_g, cA_b])

  return img_dwt

def main(method, dataset_name):
  device = torch.device('mps')
  seed=5
  random.seed(seed)
  torch.manual_seed(seed)
  np.random.seed(seed)
  torch.mps.manual_seed(seed)

  image_size=380
  batch_size=16

  train_dataset, train_loader = None, None

  train_dataset=SBI_Dataset(phase='train',image_size=image_size)
  train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size//2,shuffle=True,collate_fn=train_dataset.collate_fn,num_workers=4,pin_memory=True,drop_last=True,worker_init_fn=train_dataset.worker_init_fn)
  
  model_path = ""
  models = os.listdir("/Users/gufran/Developer/Projects/AI/DeepFakeDetection/weights")
  for m in models:
    if method.lower() in m.lower() and dataset_name.lower() in m.lower():
      model_path = "/Users/gufran/Developer/Projects/AI/DeepFakeDetection/weights/" + m
      break

  model=Detector()
  cnn_sd=torch.load(model_path, map_location=torch.device('cpu'))["model"]
  model.load_state_dict(cnn_sd)
  model=model.to(device)
  model.eval()

  print(f"Model: {model_path.split('/')[-1]}, method: {method}, dataset: {dataset_name}")
  for _,data in enumerate((train_loader)):
    img=data['img'].to(device, non_blocking=True).float()[batch_size//2:]      

    class_index = 1
    cam_generator = CAMGenerator(model, target_layer=model.net._conv_head)

    count = 0
    for i in img:
      original_image = i.cpu().detach().numpy().squeeze().transpose((1, 2, 0))
      original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

      if method == "ESBI":
        i = i.cpu().detach().numpy().transpose((1, 2, 0))

        i = get_dwt_rgb(i)
        i = torch.tensor(i).to(device, non_blocking=True).float()

      cam = cam_generator.generate_cam(i.unsqueeze(0), class_index)

      count+=1
      overlayed_image = overlay_cam(original_image, cam)

      original_image *= 255

      cv2.imwrite(f"./fig/cam/original{count}_{dataset_name}_{method}.jpg", original_image)
      cv2.imwrite(f"./fig/cam/cam{count}_{dataset_name}_{method}.jpg", overlayed_image)
    
    break
    
  print("Done", end="\n\n")
        
if __name__=='__main__':
  main("SBI", "cdf")
  main("ESBI", "cdf")