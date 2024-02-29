import torch
import torch.nn as nn
import numpy as np
from utils.esbi import ESBI_Dataset
from utils.sbi import SBI_Dataset
from model import *
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def main(model, method, dataset_name):
    device = torch.device('mps')

    image_size=380
    batch_size=16

    train_dataset, train_loader = None, None
    if method == "ESBI":
        train_dataset=ESBI_Dataset(phase='train',image_size=image_size)
        train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size//2,shuffle=True,collate_fn=train_dataset.collate_fn,num_workers=4,pin_memory=True,drop_last=True,worker_init_fn=train_dataset.worker_init_fn)
    elif method == "SBI":
        train_dataset=SBI_Dataset(phase='train',image_size=image_size)
        train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size//2,shuffle=True,collate_fn=train_dataset.collate_fn,num_workers=4,pin_memory=True,drop_last=True,worker_init_fn=train_dataset.worker_init_fn)

    model.eval()
    no_of_batches = 100

    real_feature_tensor = torch.tensor([]).to(device)
    fake_feature_tensor = torch.tensor([]).to(device)
    with torch.no_grad():
        count = 0
        for _,data in enumerate((train_loader)):
            img = data['img'].to(device, non_blocking=True).float()

            img_real = img[:batch_size//2]
            img_fake = img[batch_size//2:]

            outputs_real = model(img_real)
            outputs_fake = model(img_fake)

            real_feature_tensor = torch.cat((real_feature_tensor, outputs_real), 0)
            fake_feature_tensor = torch.cat((fake_feature_tensor, outputs_fake), 0)
            
            count+=1
            if count == no_of_batches + 1: break
            print(f"Batch [{count}/{no_of_batches}]")

    feature_vectors = torch.cat((real_feature_tensor, fake_feature_tensor), 0).cpu().numpy()
    labels = torch.cat((torch.zeros(real_feature_tensor.shape[0]), torch.ones(fake_feature_tensor.shape[0])))
    
    print("\nApplying TSNE...")
    tsne = TSNE(n_components=2, random_state=42)
    embedded_features = tsne.fit_transform(feature_vectors)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedded_features[:, 0], embedded_features[:, 1], c=labels, cmap='viridis')

    handles = scatter.legend_elements()[0]
    labels = ['Real', method]

    plt.legend(handles, labels, title="Classes")
    plt.title(f't-SNE Visualization of Feature Vectors using {method} method, trained on {dataset_name} dataset')

    plt.savefig(f'./fig/plots/{dataset_name.replace("++","")}_{method.lower()}_eb5_feature_space.png')


if __name__=='__main__':
    model=Detector()
    cnn_sd=torch.load("/Users/gufran/Developer/Projects/AI/DeepFakeDetection/weights/61_0.9993_val_sbi_eb5_cdf.tar", map_location=torch.device('mps'))["model"]
    model.load_state_dict(cnn_sd)
    model=model.to("mps")
    model.eval()

    model.cel = nn.Identity()
    model._swish = nn.Identity()
    model.net._fc = nn.Identity()
    model.net._dropout = nn.Identity()

    # print(model)
    main(model, "SBI", "CDF")