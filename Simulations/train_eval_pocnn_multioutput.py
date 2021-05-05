from functools import partial
import numpy as np
import pandas as pd
import random
import json 
import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models

from sklearn.metrics import roc_auc_score

from PIL import Image
from itertools import compress 
 


transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])



class PSODataset_train(Dataset):
    
    def __init__(self, images,clinical_data):
        self.images = images
        self.clinical_data = clinical_data
        self.id_images = clinical_data['ID'] 
    def __len__(self):
        return len(self.id_images)
    def __getitem__(self, idx):
        ID = int(self.id_images.iloc[idx])
        image = self.images[ID]
        image = np.transpose(image,(2,0,1))
        po1 = np.array(self.clinical_data.iloc[idx,1] ).astype(np.float32)
        po2 = np.array(self.clinical_data.iloc[idx,2] ).astype(np.float32)
        po3 = np.array(self.clinical_data.iloc[idx,3] ).astype(np.float32)
        po4 = np.array(self.clinical_data.iloc[idx,4] ).astype(np.float32)
        po5 = np.array(self.clinical_data.iloc[idx,5] ).astype(np.float32)        
        return image, po1,po2,po3, po4,po5,  ID


class Dataset_test(Dataset):
    
    def __init__(self, images,clinical_data): 
        self.images = images
        self.clinical_data = clinical_data
        self.id_images = clinical_data['ID']        
    def __len__(self):
        return len(self.id_images)
    def __getitem__(self, idx):
        ID = int(self.id_images.iloc[idx])
        image = self.images[ID]
        image = np.transpose(image,(2,0,1))
        time_point = np.array(self.clinical_data.iloc[idx,8:] ).astype(np.float32)
        time = np.array(self.clinical_data.iloc[idx,1] ).astype(np.float32)
        event = np.array(self.clinical_data.iloc[idx,2] ).astype(np.float32)
        event_1 = np.array(self.clinical_data.iloc[idx,3] ).astype(np.float32)
        event_2 = np.array(self.clinical_data.iloc[idx,4] ).astype(np.float32)
        event_3 = np.array(self.clinical_data.iloc[idx,5] ).astype(np.float32)
        event_4 = np.array(self.clinical_data.iloc[idx,6] ).astype(np.float32)
        event_5 = np.array(self.clinical_data.iloc[idx,7] ).astype(np.float32)        
                
        return image, time, event, event_1, event_2, event_3, event_4, event_5,  ID

       
class Resnet18_Mtl(nn.Module):
    def __init__(self, use_pretrained):
        super(Resnet18_Mtl, self).__init__()
        model_ft = models.resnet18(pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features
        image_modules = list(model_ft.children())[:-1]
        self.modelA = nn.Sequential(*image_modules)
        self.fc = nn.Linear(num_ftrs  + 10, 1)           
        
    def forward(self, image,clin_covariates ):
        x = self.modelA(image)
        x = torch.flatten(x, 1)
        x = torch.cat((x, clin_covariates), dim=1)
        x1 = torch.tanh(self.fc(x))
        x2 = torch.tanh(self.fc(x))
        x3 = torch.tanh(self.fc(x))
        x4 = torch.tanh(self.fc(x))
        x5 = torch.tanh(self.fc(x))
        return x1,x2,x3,x4,x5       

def mean_abs_error(output,label):
    return torch.mean( torch.abs(label-output) )

metrics_train = {'mae': mean_abs_error}


def roc_auc(y_preds, y_targets):
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        raise RuntimeError("This contrib module requires sklearn to be installed.")
    y_true = y_targets
    y_pred = y_preds
    return roc_auc_score(y_true, y_pred)
    
    
def train_eval_resnet_sim(dataset,clindata,lr):
    net = Resnet18_Mtl(use_pretrained=True)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(),lr=lr)       
    batch_size = 2**8 # 2**8 2**9 or 2**7
    ds_train = PSODataset_train(trainset.data,clindata['train_val'])
    trainloader = torch.utils.data.DataLoader(ds_train, batch_size=batch_size,shuffle=True)
    for epoch in range(25):  
        running_loss = 0.0
        epoch_steps = 0
        net.train()
        for i, data in enumerate(trainloader, 0):
            input_batch, po1_batch, po2_batch, po3_batch, po4_batch, po5_batch, slide_id_batch = data          
            input_batch = input_batch.to(device)
            covariates_batch = np.concatenate([np.array(clindata['covariates'][clindata['covariates']['ID']== int(file)].iloc[:,1:]).astype(np.float32) for file in slide_id_batch])            
            covariates_batch = torch.from_numpy(covariates_batch).to(device)
            po1_batch, po2_batch, po3_batch, po4_batch, po5_batch = po1_batch.to(device), po2_batch.to(device), po3_batch.to(device), po4_batch.to(device), po5_batch.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            output_batch1, output_batch2,output_batch3,output_batch4,output_batch5 = net(input_batch.float(), covariates_batch )            
            loss = criterion(output_batch1,torch.unsqueeze(po1_batch,axis=1) )
            loss += criterion(output_batch2,torch.unsqueeze(po2_batch,axis=1)  )
            loss += criterion(output_batch3,torch.unsqueeze(po3_batch,axis=1)  )
            loss += criterion(output_batch4,torch.unsqueeze(po4_batch,axis=1)  )
            loss += criterion(output_batch5,torch.unsqueeze(po5_batch,axis=1)  )
            loss.backward()
            optimizer.step()          
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,running_loss / epoch_steps))
                running_loss = 0.0                 
        # Validation loss       
        time = []
        event = []
        event_1 = []
        event_2 = []
        event_3 = []
        event_4 = []
        event_5 = []
        slide_id =  np.array([])
        output1 =  np.array([])
        output2 =  np.array([])
        output3 =  np.array([])
        output4 =  np.array([])
        output5 =  np.array([])
        batch_size = 2**8 # 2**8 2**9 or 2**7        
        ds_test = Dataset_test(trainset.data,clindata['test'])
        testloader = torch.utils.data.DataLoader(ds_test, batch_size=batch_size,shuffle=False) 
        with torch.no_grad():
            for i,(input_batch, time_batch, event_batch, event_batch_1, event_batch_2, event_batch_3, event_batch_4, event_batch_5,  slide_id_batch) in enumerate(testloader,0):           
                input_batch = input_batch.to(device)
                covariates_batch = np.concatenate([np.array(clindata['covariates'][clindata['covariates']['ID']== int(file)].iloc[:,1:]).astype(np.float32) for file in slide_id_batch])            
                covariates_batch = torch.from_numpy(covariates_batch).to(device)
                output_batch1, output_batch2,output_batch3,output_batch4,output_batch5 = net(input_batch.float(),covariates_batch)                
                output_batch1 = output_batch1.data.cpu()
                output1 = np.concatenate([output1, np.squeeze(output_batch1.detach().numpy(),axis=1)])
                output_batch2 = output_batch2.data.cpu()
                output2 = np.concatenate([output2, np.squeeze(output_batch2.detach().numpy(),axis=1)])
                output_batch3 = output_batch3.data.cpu()
                output3 = np.concatenate([output3, np.squeeze(output_batch3.detach().numpy(),axis=1)])
                output_batch4 = output_batch4.data.cpu()
                output4 = np.concatenate([output4, np.squeeze(output_batch4.detach().numpy(),axis=1)])
                output_batch5 = output_batch5.data.cpu() 
                output5 = np.concatenate([output5, np.squeeze(output_batch5.detach().numpy(),axis=1)])
                slide_id = np.concatenate([slide_id, np.array(slide_id_batch)])           
                time.append(time_batch.numpy())
                event.append(event_batch.numpy())
                event_1.append(event_batch_1)
                event_2.append(event_batch_2)
                event_3.append(event_batch_3)
                event_4.append(event_batch_4)
                event_5.append(event_batch_5)
        time = np.concatenate(time)
        event = np.concatenate(event)
        event1 = np.concatenate(event_1)
        event2 = np.concatenate(event_2)
        event3 = np.concatenate(event_3)
        event4 = np.concatenate(event_4)
        event5 = np.concatenate(event_5)            
        auc1 = roc_auc(output1,event1)
        auc2 = roc_auc(output2,event2)
        auc3 = roc_auc(output3,event3)
        auc4 = roc_auc(output4,event4)
        auc5 = roc_auc(output5,event5)
        auc = {'cutoff_1':auc1,'cutoff_2': auc2, 'cutoff_3': auc3, 'cutoff_4':auc4, 'cutoff_5':auc5}
        
    return auc1, auc2, auc3, auc4, auc5
    
parser = argparse.ArgumentParser()
parser.add_argument('--niter', type=int, default=25)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--sample_size', type=int, default=1000)
parser.add_argument('--nsim', type=int, default=100)
parser.add_argument('--case', type=int, default=1, help='Any case considered in the paper') 
parser.add_argument('--data_dir', default='./dataset_cifar', help="Directory to save CIFAR10 dataset")
parser.add_argument('--po', default='po', help="data generation using PO or IPCW-PO")


            
if __name__ == "__main__":

    args = parser.parse_args()
    if args.po == 'po':
        import data_generating.data_generating_po_multioutput as po
    else:
        import data_generating.data_generating_ipcwpo_multioutput as po
    num_sim = args.nsim
    sample_size = args.sample_size
    cifardataset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False,download=True, transform=transform)
    
    auc1 = []
    auc2 = []
    auc3 = []
    auc4 = []
    auc5 = []
    for i in range(num_sim):
    
        if args.case == 1:
            clindata = po.sim_event_times_case1(cifardataset, args.sample_size)
        elif args.case == 2:
            clindata = sim_event_times_case2(cifardataset, args.sample_size)
        elif args.case == 3:
            clindata = sim_event_times_case3(cifardataset, args.sample_size)
        elif args.case == 4:
            clindata = sim_event_times_case4(cifardataset, args.sample_size)
        elif args.case == 5:
            clindata = po.sim_event_times_case5(cifardataset, args.sample_size)
        else :
            clindata = sim_event_times_case6(cifardataset, args.sample_size)
            
        auc1_j, auc2_j, auc3_j, auc4_j, auc5_j = train_eval_resnet_sim(cifardataset, clindata, args.lr,args.niter) 
        auc1.append(auc1_j)
        auc2.append(auc2_j)
        auc3.append(auc3_j)
        auc4.append(auc4_j)
        auc5.append(auc5_j)
        if i % 25 == 0:
            print(i)