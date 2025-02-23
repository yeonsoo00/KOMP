import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
import warnings
import argparse
from skimage.measure import label, regionprops
from scipy import ndimage
import wandb
import numpy as np
from skimage.morphology import binary_closing, binary_opening
from models import UNetBinary
from data import CustomDataset
from utils import *
warnings.filterwarnings('ignore')

"""
TODO 
    - overlap targets and outputs (path2 /home/yec23006/projects/research/KOMP/output/results/)
"""



def train_unet(data_root, target_size, batch_size, lr, num_epochs, gpu):
    
    # gpu set-up
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create U-Net model
    in_channels = 1  # Assuming Binary imgs
    out_channels = 1  # Output is Binary img
    model = UNetBinary(in_channels, out_channels)
    model = nn.DataParallel(model).to(device)

    # Define loss and optimizer
    pos_weight = torch.tensor([10.0]).to(device)

    criterion = nn.BCELoss() #WeightedBCELoss(pos_weight=pos_weight) #DiceLoss() #nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr) #optim.SGD(model.parameters(), lr=lr, momentum=0.9)#optim.RMSprop(model.parameters(), lr=lr)#optim.Adam(model.parameters(), lr=lr)

    # data augmentation
    # Define transformations for data augmentation
    augmentation_transform = v2.Compose([
        v2.RandomRotation(degrees=30),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        #transforms.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0)),
        v2.ToTensor()
    ])
    # Assuming you have a custom dataset class (CustomDataset) for training
    # Adjust the 'root' parameter based on your dataset structure
    train_dataset = CustomDataset(root=data_root, target_size = target_size, transform=augmentation_transform) #transforms.ToTensor())

    # Split the dataset into training and validation sets
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    print("traindata size : ", train_size)
    print("testdata size : ", val_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Training loop
    loss_list = []
    val_loss_list = []
    best_loss = 5

    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # wandb targets 
            # train_examples+=[wandb.Image(im) for im in targets]

        # Validation
        model.eval()
        val_input_list=[]
        val_result_list=[]
        val_target_list =[]
        postprocessed_mask = []
        masks_list=[]
        postprocessed_mask_final = []
        masks_list_final = []
        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)

                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_targets)

                # wandb val output 
                val_input_list += [wandb.Image(im) for im in val_inputs]
                val_result_list += [wandb.Image(im) for im in val_outputs]
                val_target_list += [wandb.Image(im) for im in val_targets]
                for im in val_outputs:
                    postprocessed_mask.append(postprocess_mask(im))
                masks_list += [wandb.Image(im) for im in postprocessed_mask]

                for pred in postprocessed_mask:
                    holeFilled = ndimage.binary_fill_holes(pred).astype(int)
                    postprocessed_mask_final.append(holeFilled)
                masks_list_final += [wandb.Image(im) for im in postprocessed_mask_final]
                    

        wandb.log({"val inputs": val_input_list, "val results": val_result_list, "val targets":val_target_list, "post-processed masks":masks_list, "final post-processed masks":masks_list_final, "epoch": epoch})

        current_loss = loss.item()
        current_val_loss = val_loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {current_loss:.4f}, Val Loss: {current_val_loss:.4f}')
        loss_list.append(current_loss)
        val_loss_list.append(current_val_loss)
        wandb.log({"val_loss": current_val_loss, "loss": current_loss, "epoch":epoch})

        

        if current_val_loss < best_loss:
            print('Save checkpoint e', str(epoch))
            torch.save(model.state_dict(), os.path.join('/home/yec23006/projects/research/KOMP/ckpt', 'unet_model_fix.pth'))
            best_loss = current_val_loss
            
            # pred mask
            for i, mask in enumerate(postprocessed_mask_final):
                mask = np.array(mask)
                np.save('/home/yec23006/projects/research/KOMP/output/results_all/mask/'+str(i) + '.npy', mask)
                # maskfig = plt.figure()
                # plt.imshow(mask)
                # maskfig.savefig('/home/yec23006/projects/research/KOMP/output/results_all/mask/'+str(epoch) + '_' + str(i) + '.png')
                # plt.close()
            # target mask
            for i, mask in enumerate(val_targets):
                mask = np.array(mask.detach().cpu().numpy()[0])
                np.save('/home/yec23006/projects/research/KOMP/output/results_all/target/'+ str(i) + '.npy', mask)
                # maskfig = plt.figure()
                # plt.imshow(mask)
                # maskfig.savefig('/home/yec23006/projects/research/KOMP/output/results_all/target/'+str(epoch) + '_' + str(i) + '.png')
                # plt.close()
            # image
            for i, img in enumerate(val_inputs):
                img = np.array(img.detach().cpu().numpy()[0])
                np.save('/home/yec23006/projects/research/KOMP/output/results_all/img/' + str(i) + '.npy', img)
                # fig = plt.figure()
                # plt.imshow(img)
                # fig.savefig('/home/yec23006/projects/research/KOMP/output/results_all/img/'+str(epoch) + '_' + str(i) + '.png')
                # plt.close()



if __name__ == "__main__":
    # parser
    parser = argparse.ArgumentParser(description="Train U-Net model")
    parser.add_argument("--data_root", type=str, default="/home/yec23006/projects/research/KOMP/Vertebrae/", help="Root directory of the dataset")
    parser.add_argument("--target_size", type=list, nargs=2, default=[256, 256], help="Target size for resizing images")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=500, help="Number of epochs for training")
    parser.add_argument("--gpu", type=str, default="0, 1", help="Number of epochs for training")
    parser.add_argument("--model", type=str, default="unetbin", help="Model")
    parser.add_argument("--run_name", type=str, default="unet_train", help="Number of epochs for training")
    args = parser.parse_args()

    # wandb setting
    config = {"lr":args.lr, "num_epochs":args.num_epochs, "target_size":args.target_size, "batch_size":args.batch_size, "model":args.model}
    wandb.init(config=config, project = "unet KOMP Vertebrae")
    wandb.run.name = args.run_name
    
    # run the model
    train_unet(args.data_root, args.target_size, args.batch_size, args.lr, args.num_epochs, args.gpu)

    wandb.finish()