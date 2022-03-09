from functools import partial
import numpy as np
import pandas as pd
import random
import json 
import os
import argparse

from PIL import Image
from itertools import compress

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from ray import tune
from ray.tune import CLIReporter, ExperimentAnalysis
from ray.tune.schedulers import ASHAScheduler

from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index

from sksurv.util import Surv
from sksurv.util import Surv
from sksurv.metrics import (concordance_index_censored,
                            concordance_index_ipcw,
                            cumulative_dynamic_auc)

from sklearn.metrics import roc_auc_score
#from lifelines import KaplanMeierFitter
#from lifelines.utils import concordance_index

import clinical_data as df_c




def how_many_tiles(data_dir,id_patient):
    foldernames = [os.path.join(data_dir,f) for f in id_patient]
    filenames = []
    for i,data_dir_image in enumerate(foldernames):
        filenames_patient= os.listdir(data_dir_image)
        filenames_patient = [os.path.join(data_dir_image, f) for f in filenames_patient if f.endswith('.jpg')   ]               
        filenames.append(len(filenames_patient))
    return filenames

    
def get_filenames(data_dir,id_patient,max_tiles):
    foldernames = [os.path.join(data_dir,f) for f in id_patient]
    filenames = []
    for i,data_dir_image in enumerate(foldernames):
        filenames_patient= os.listdir(data_dir_image)
        filenames_patient = [os.path.join(data_dir_image, f) for f in filenames_patient if f.endswith('.jpg')   ]        
        random.shuffle(filenames_patient)
        delta = max_tiles - len(filenames_patient)
        if delta>0:
           filenames_patient_completion = np.random.choice(filenames_patient, size=delta, replace=True, p=None) 
           filenames.extend(filenames_patient_completion)           
        filenames.extend(filenames_patient)
    return filenames

    
train_transformer = transforms.Compose([   
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=(-90,90)),    
    transforms.ToTensor()])

eval_transformer = transforms.Compose([
    transforms.ToTensor()]) 

class PSODataset_train(Dataset):
    
    def __init__(self, filenames,clinical_data, transform):
        '''
        Args:
        filenames: location where the image of the train or val or test is located
        clinical_data: clinical data of train or val or test
        transform: transformation to apply on image
        
        Return: float32
        
        ''' 
        self.filenames = filenames
        self.slide_id = [(os.path.split(filename)[-1][0:16]) for filename in filenames]
        self.slide_id = pd.Series(self.slide_id,name='ID_slide')
        self.filenames = pd.Series(self.filenames,name='file')
        self.id_file = pd.concat([self.slide_id, self.filenames], axis=1)
        self.clinical_data = pd.merge(left=self.id_file, right=clinical_data, how='left',left_on='ID_slide' ,right_on='ID_slide')
        self.transform = transform
    def __len__(self):
        return len(self.filenames)
    def __getitem__(self, idx):
        image = Image.open(self.clinical_data.iloc[idx,1]) # PIL image
        image = self.transform(image)
        po1 = np.array(self.clinical_data.iloc[idx,2] ).astype(np.float32)
        po2 = np.array(self.clinical_data.iloc[idx,3] ).astype(np.float32)
        po3 = np.array(self.clinical_data.iloc[idx,4] ).astype(np.float32)
        po4 = np.array(self.clinical_data.iloc[idx,5] ).astype(np.float32)
        po5 = np.array(self.clinical_data.iloc[idx,6] ).astype(np.float32)
        slide_id = self.clinical_data.iloc[idx,0]
        
        return image, po1, po2,po3, po4,po5,  slide_id


class Dataset_test(Dataset):
    
    def __init__(self, filenames,clinical_data, transform):
        '''
        Args:
        filenames: location where the image of the train or val or test is located
        clinical_data: clinical data of train or val or test
        transform: transformation to apply on image
        ''' 
        self.filenames = filenames
        self.slide_id = [(os.path.split(filename)[-1][0:16]) for filename in filenames]
        self.slide_id = pd.Series(self.slide_id,name='ID_slide')
        self.filenames = pd.Series(self.filenames,name='file')
        self.id_file = pd.concat([self.slide_id, self.filenames], axis=1)
        self.clinical_data = pd.merge(left=self.id_file, right=clinical_data, how='left',left_on='ID_slide' ,right_on='ID_slide')
        #self.filenames = list(compress(self.filenames, self.clinical_data['age_at_index'].isnull()==False))
        #self.clinical_data = self.clinical_data[self.clinical_data['age_at_index'].isnull()==False]
        self.transform = transform
    def __len__(self):
        return len(self.filenames)
    def __getitem__(self, idx):
        #ID = self.slide_id[idx]
        image = Image.open(self.clinical_data.iloc[idx,1]) # PIL image
        image = self.transform(image)
        #time_point = np.array(self.clinical_data.iloc[idx,4:] ).astype(np.float32)
        time = np.array(self.clinical_data.iloc[idx,2] ).astype(np.float32)
        event = np.array(self.clinical_data.iloc[idx,3] ).astype(np.float32)        
        slide_id = self.clinical_data.iloc[idx,0]
                
        return image, time, event,  slide_id    

       
        

def mean_abs_error(output,label):
    return torch.mean( torch.abs(label-output) )

metrics_train = {'mae': mean_abs_error}


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


feature_extract = True

###########################  Single Output #######################################################
class Resnet18_so(nn.Module):
	def __init__(self, use_pretrained):
		super(Resnet18_so, self).__init__()
		model_ft = models.resnet18(pretrained=use_pretrained)
		set_parameter_requires_grad(model_ft, feature_extract)
		num_final_in = model_ft.fc.in_features
		model_ft.fc = nn.Linear(num_final_in, 1)
		self.vismodel = model_ft
		self.projective = nn.Linear(1 + 20,1)
		self.nonlinearity = nn.LeakyReLU(inplace=True)
		
	def forward(self, image,clin_covariates):
		x = self.vismodel(image)
		x = torch.flatten(x, 1)
		x = torch.cat((x, clin_covariates), dim=1)
		x = self.projective(x)
		x = self.nonlinearity(x)
				
		return x

def train_resnet_so(config, checkpoint_dir=None, data_dir=None):
    net = Resnet18_so(use_pretrained=True)    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net.to(device)
    criterion = nn.MSELoss()
    params_to_update = net.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in net.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in net.named_parameters():
            if param.requires_grad == True:
                print("\t",name)
    optimizer = optim.Adam(params_to_update,lr=config['lr'])
    #optimizer = optim.SGD(params_to_update, lr=0.0001, momentum=0.9)
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)        
    batch_size = 2**2 # 2**8 2**9 or 2**7
    id_patient = [(os.path.split(filename)[-1][8:12]) for filename in clindata['train']['ID_slide']]
    max_tiles = np.max(how_many_tiles(args.data_dir_train,id_patient))
    filenames = get_filenames(args.data_dir_train,id_patient, max_tiles)
    ds_train = PSODataset_train(filenames,clindata['train'],train_transformer)
    train_losses = []
    val_losses = []
    for epoch in range(30):  # loop over the dataset multiple times
        trainloader = DataLoader(ds_train,batch_size=batch_size,shuffle=True, num_workers=4,pin_memory=True) # because we use 50000 samples shuffled
        running_loss = 0.0
        epoch_steps = 0
        net.train()
        for i, data in enumerate(trainloader, 0):
            input_batch, po1_batch, po2_batch, po3_batch, po4_batch, po5_batch, slide_id_batch = data
            # from wide to long pseudo-observations per tile, so each tile has its five points
            po_dict = {'po1':po1_batch,'po2':po2_batch,'po3':po3_batch,'po4':po4_batch,'po5':po5_batch, 'ID_slide':slide_id_batch}
            po_df_batch = pd.DataFrame(po_dict)
            po_long_batch = pd.melt(po_df_batch, id_vars=['ID_slide'],value_vars=['po1','po2','po3','po4','po5'] )
            po_long_batch.rename(columns={'variable': 'time_point','value': 'ps_risk'}, inplace=True)
            mymap= {'po1': 2, 'po2': 3.5, 'po3': 5, 'po4': 8, 'po5': 10  }
            po_long_batch = po_long_batch.applymap(lambda s : mymap.get(s) if s in mymap else s)
            po_long_batch = pd.get_dummies(po_long_batch, columns=['time_point'])
            input_batch_long = torch.cat([input_batch,input_batch,input_batch,input_batch,input_batch], axis=0)
            input_batch_long = input_batch_long.to(device)
            covariates_batch = np.concatenate([np.array(clindata['covariates'][clindata['covariates']['ID_slide']== file].iloc[:,1:]).astype(np.float32) for file in po_long_batch['ID_slide']])            
            covariates_batch = torch.from_numpy(covariates_batch).to(device)
            time_point_batch = torch.from_numpy(np.array(po_long_batch.iloc[:,2:]).astype(np.float32)).to(device)
            po_long_batch = torch.from_numpy(np.array(po_long_batch['ps_risk']).astype(np.float32)).to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            output_batch = net(input_batch_long,torch.cat((time_point_batch, covariates_batch), 1))
            loss = criterion(output_batch,torch.unsqueeze(po_long_batch,axis=1)  )
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            #if i % 2000 == 1999:  # print every 2000 mini-batches
            #    print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,running_loss / epoch_steps))
            #    running_loss = 0.0  
            if i == int(50000/batch_size):
                break
        train_loss_epoch = running_loss / ( int(50000/batch_size) * batch_size ) 
        train_losses.append(train_loss_epoch)
        print("[%d] loss: %.3f" % (epoch + 1, train_loss_epoch))                
        # Validation loss       
        slide_id =  np.array([])
        time =  np.array([])
        val_loss = 0.0
        val_mae = np.array([])
        val_steps = 0.0
        batch_size = 2**2 # 2**8 2**9 or 2**7
        id_patient = [(os.path.split(filename)[-1][8:12]) for filename in clindata['val']['ID_slide']]
        max_tiles = np.max(how_many_tiles(args.data_dir_train,id_patient))
        filenames = get_filenames(args.data_dir_train,id_patient, max_tiles)
        ds_val = PSODataset_train(filenames,clindata['val'],train_transformer)       
        valloader = DataLoader(ds_val,batch_size=batch_size,shuffle=True, num_workers=4,pin_memory=True)        
        net.eval()
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                input_batch, po1_batch, po2_batch, po3_batch, po4_batch, po5_batch, slide_id_batch = data               
                po_dict = {'po1':po1_batch,'po2':po2_batch,'po3':po3_batch,'po4':po4_batch,'po5':po5_batch, 'ID_slide':slide_id_batch}
                po_df_batch = pd.DataFrame(po_dict)
                po_long_batch = pd.melt(po_df_batch, id_vars=['ID_slide'],value_vars=['po1','po2','po3','po4','po5'] )
                po_long_batch.rename(columns={'variable': 'time_point','value': 'ps_risk'}, inplace=True)
                mymap= {'po1': 2, 'po2': 3.5, 'po3': 5, 'po4': 8, 'po5': 10  }
                po_long_batch = po_long_batch.applymap(lambda s : mymap.get(s) if s in mymap else s)
                po_long_batch = pd.get_dummies(po_long_batch, columns=['time_point'])
                input_batch_long = torch.cat([input_batch,input_batch,input_batch,input_batch,input_batch], axis=0)
                input_batch_long = input_batch_long.to(device)
                covariates_batch = np.concatenate([np.array(clindata['covariates'][clindata['covariates']['ID_slide']== file].iloc[:,1:]).astype(np.float32) for file in po_long_batch['ID_slide']])            
                covariates_batch = torch.from_numpy(covariates_batch).to(device)
                time_point_batch = torch.from_numpy(np.array(po_long_batch.iloc[:,2:]).astype(np.float32)).to(device)
                po_long_batch = torch.from_numpy(np.array(po_long_batch['ps_risk']).astype(np.float32))
                output_batch = net(input_batch_long,torch.cat((time_point_batch, covariates_batch), 1))                
                output_batch = output_batch.data.cpu()  
                val_loss += criterion(output_batch,torch.unsqueeze(po_long_batch,axis=1)  )
                val_mae = np.concatenate([val_mae, np.squeeze(abs(output_batch-torch.unsqueeze(po_long_batch,axis=1) ),axis=1)] )
                #val_mae += mean_abs_error(output_batch,torch.unsqueeze(po_long_batch,axis=1) ) 
                val_steps += 1
                if i == int(50000/batch_size):
                    break
        val_loss_epoch = val_loss / ( int(50000/batch_size) * batch_size ) 
        val_losses.append(val_loss_epoch)
        val_mae_epoch = np.mean(val_mae)
        print("[%d] val loss: %.3f" % (epoch + 1, val_loss_epoch))
        print("[%d] val accur : %.3f" % (epoch + 1, val_mae_epoch) ) 
        print(train_losses)
        print( val_losses  )        
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((net.state_dict(), optimizer.state_dict()), path)
        tune.report( training_loss = train_loss_epoch ,validation_loss = val_loss_epoch, mae_target = val_mae_epoch.item())
        #tune.report(mae_target = (val_mae / val_steps).item())                
    print("Finished Training")

    
def test_accuracy_so(net,device='cpu'):
    pred1_ind_ave = []
    pred2_ind_ave = []
    pred3_ind_ave = []
    pred4_ind_ave = []
    pred5_ind_ave = []
    pred1_ind_quant = []
    pred2_ind_quant = []
    pred3_ind_quant = []
    pred4_ind_quant = []
    pred5_ind_quant = []
    time = []
    event = []
    correct = 0
    step = 0
    batch_size = 1 # 2**8 2**9 or 2**7
    for file in clindata['slide_id_test']:
        time_point = []
        output1 =  np.array([])
        output2 =  np.array([])
        output3 =  np.array([])
        output4 =  np.array([])
        output5 =  np.array([])
        slide_id =  np.array([])
        id_patient = (os.path.split(file)[-1][8:12]) 
        foldernames = os.path.join(args.data_dir_test,id_patient)
        filenames_patient= os.listdir(foldernames)
        filenames_patient = [os.path.join(foldernames, f) for f in filenames_patient if f.endswith('.jpg')   ]
        dl = DataLoader(Dataset_test(filenames_patient,clindata['test'],eval_transformer),batch_size=batch_size,shuffle=True, num_workers=4,pin_memory=True) 
        net.eval()
        with torch.no_grad():
            for i,(input_batch ,time_batch, event_batch,  slide_id_batch) in enumerate(dl,0):           
                input_batch_long = torch.cat([input_batch,input_batch,input_batch,input_batch,input_batch], axis=0)
                input_batch_long = input_batch_long.to(device)               
                covariates_batch = np.concatenate([np.array(clindata['covariates'][clindata['covariates']['ID_slide']== file].iloc[:,1:]).astype(np.float32) for file in slide_id_batch])            
                covariates_batch_long = np.concatenate([covariates_batch,covariates_batch,covariates_batch,covariates_batch,covariates_batch], axis=0)
                covariates_batch_long = torch.from_numpy(covariates_batch_long).to(device)
                time_point_batch = np.eye(5).astype(np.float32)
                time_point_batch = torch.from_numpy(time_point_batch).to(device)                              
                output_batch = net(input_batch_long, torch.cat((time_point_batch, covariates_batch_long), 1))            
                output_batch = output_batch.data.cpu()           
                output1 = np.concatenate([output1, output_batch[0].detach().numpy()])
                output2 = np.concatenate([output2, output_batch[1].detach().numpy()])
                output3 = np.concatenate([output3, output_batch[2].detach().numpy()])
                output4 = np.concatenate([output4, output_batch[3].detach().numpy()])
                output5 = np.concatenate([output5, output_batch[4].detach().numpy()])
                slide_id = np.concatenate([slide_id, np.array(slide_id_batch)])
                if i == int(1000/batch_size):
                      break
        time.append(time_batch.numpy()[0])
        event.append(event_batch.numpy()[0])
        pred1_ind_ave.append(np.mean(output1))    
        pred2_ind_ave.append(np.mean(output2)) 
        pred3_ind_ave.append(np.mean(output3)) 
        pred4_ind_ave.append(np.mean(output4))        
        pred5_ind_ave.append(np.mean(output5))
        pred1_ind_quant.append(np.quantile(output1,.75))    
        pred2_ind_quant.append(np.quantile(output2,.75)) 
        pred3_ind_quant.append(np.quantile(output3,.75)) 
        pred4_ind_quant.append(np.quantile(output4,.75))        
        pred5_ind_quant.append(np.quantile(output5,.75))        
    y_surv_test = Surv.from_arrays(event=event, time=time)
    y_surv_train = Surv.from_arrays(event=np.squeeze(clindata['event_train'],axis=1), time=np.squeeze(clindata['time_train'],axis=1))
    auc1 = cumulative_dynamic_auc( y_surv_train, y_surv_test, pred1_ind_ave, 730)[0]  
    auc2 = cumulative_dynamic_auc( y_surv_train, y_surv_test, pred2_ind_ave, 1277)[0]
    auc3 = cumulative_dynamic_auc( y_surv_train, y_surv_test, pred3_ind_ave, 1825)[0]
    auc4 = cumulative_dynamic_auc( y_surv_train, y_surv_test, pred4_ind_ave, 2920)[0]
    auc5 = cumulative_dynamic_auc( y_surv_train, y_surv_test, pred5_ind_ave, 3650)[0]
    auc_average = {'2_year':auc1,'3.5_year': auc2, '5_year': auc3, '8_year':auc4, '10_year':auc5}   
    auc1 = cumulative_dynamic_auc( y_surv_train, y_surv_test, pred1_ind_quant, 730)[0]  
    auc2 = cumulative_dynamic_auc( y_surv_train, y_surv_test, pred2_ind_quant, 1277)[0]
    auc3 = cumulative_dynamic_auc( y_surv_train, y_surv_test, pred3_ind_quant, 1825)[0]
    auc4 = cumulative_dynamic_auc( y_surv_train, y_surv_test, pred4_ind_quant, 2920)[0]
    auc5 = cumulative_dynamic_auc( y_surv_train, y_surv_test, pred5_ind_quant, 3650)[0]
    auc_quantile = {'2_year':auc1,'3.5_year': auc2, '5_year': auc3, '8_year':auc4, '10_year':auc5}
    return auc_average, auc_quantile

###########################  Multiple Output #######################################################
    
class Resnet18_mo(nn.Module):
    def __init__(self, use_pretrained):
        super(Resnet18_mo, self).__init__()
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_final_in = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_final_in, 1)
        self.vismodel = model_ft
        self.projective1 = nn.Linear(1 + 15,1)
        self.projective2 = nn.Linear(1 + 15,1)
        self.projective3 = nn.Linear(1 + 15,1)
        self.projective4 = nn.Linear(1 + 15,1)
        self.projective5 = nn.Linear(1 + 15,1)
        self.nonlinearity = nn.LeakyReLU(inplace=True)
        
    def forward(self, image,clin_covariates):
        x = self.vismodel(image)
        x = torch.flatten(x, 1)
        x = torch.cat((x, clin_covariates), dim=1)
        x1 = self.projective1(x)
        x2 = self.projective2(x)
        x3 = self.projective3(x)
        x4 = self.projective4(x)
        x5 = self.projective5(x)
        x1 = self.nonlinearity(x1)
        x2 = self.nonlinearity(x2)
        x3 = self.nonlinearity(x3)
        x4 = self.nonlinearity(x4)
        x5 = self.nonlinearity(x5)
                
        return x1, x2, x3, x4, x5		

     

def train_resnet_mo(config, checkpoint_dir=None, data_dir=None):
    net = Resnet18_mo(use_pretrained=True)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)
    criterion = nn.MSELoss()
    params_to_update = net.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in net.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in net.named_parameters():
            if param.requires_grad == True:
                print("\t",name)
    optimizer = optim.Adam(params_to_update,lr=config['lr'])
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)        
    batch_size = 2**2 # 2**8 2**9 or 2**7
    id_patient = [(os.path.split(filename)[-1][8:12]) for filename in clindata['train']['ID_slide']]
    max_tiles = np.max(how_many_tiles(args.data_dir_train,id_patient))
    filenames = get_filenames(args.data_dir_train,id_patient, max_tiles)
    ds_train = PSODataset_train(filenames,clindata['train'],train_transformer)
    train_losses = []
    val_losses = []
    for epoch in range(30):  # loop over the dataset multiple times
        trainloader = DataLoader(ds_train,batch_size=batch_size,shuffle=True, num_workers=4,pin_memory=True)       
        train_running_loss = 0.0
        epoch_steps = 0
        net.train()
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            input_batch, po1_batch, po2_batch, po3_batch, po4_batch, po5_batch, slide_id_batch = data
            covariates_batch = np.concatenate([np.array(clindata['covariates'][clindata['covariates']['ID_slide']== file].iloc[:,1:]).astype(np.float32) for file in slide_id_batch])            
            covariates_batch = torch.from_numpy(covariates_batch).to(device)
            input_batch = input_batch.to(device)
            po1_batch, po2_batch, po3_batch, po4_batch, po5_batch = po1_batch.to(device), po2_batch.to(device), po3_batch.to(device), po4_batch.to(device), po5_batch.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            output_batch1, output_batch2,output_batch3,output_batch4,output_batch5 = net(input_batch.float(), covariates_batch )  
            train_loss = criterion(output_batch1,torch.unsqueeze(po1_batch,axis=1) )
            train_loss += criterion(output_batch2,torch.unsqueeze(po2_batch,axis=1)  )
            train_loss += criterion(output_batch3,torch.unsqueeze(po3_batch,axis=1)  )
            train_loss += criterion(output_batch4,torch.unsqueeze(po4_batch,axis=1)  )
            train_loss += criterion(output_batch5,torch.unsqueeze(po5_batch,axis=1)  )
            train_loss.backward()
            optimizer.step()
            # print statistics
            train_running_loss += train_loss.item() 
            if i == int(50000/batch_size):
                break
        train_loss_epoch = train_running_loss / ( int(50000/batch_size) * batch_size ) 
        train_losses.append(train_loss_epoch)
        print("[%d] loss: %.3f" % (epoch + 1, train_loss_epoch))        
        # Validation loss       
        slide_id =  np.array([])
        time =  np.array([])
        val_mae = 0.0
        val_steps = 0.0
        batch_size = 2**2 # 2**8 2**9 or 2**7
        id_patient = [(os.path.split(filename)[-1][8:12]) for filename in clindata['val']['ID_slide']]
        max_tiles = np.max(how_many_tiles(args.data_dir_train,id_patient))
        filenames = get_filenames(args.data_dir_train,id_patient, max_tiles)
        #ds_val = PSODataset_train(filenames,clindata['val'],eval_transformer)
        ds_val = PSODataset_train(filenames,clindata['val'],train_transformer)
        valloader = DataLoader(ds_val,batch_size=batch_size,shuffle=True, num_workers=4,pin_memory=True)        
        net.eval()
        val_mae1 =  np.array([])
        val_mae2 =  np.array([])
        val_mae3 =  np.array([])
        val_mae4 =  np.array([])
        val_mae5 =  np.array([])
        val_running_loss = 0.0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                input_batch, po1_batch, po2_batch, po3_batch, po4_batch, po5_batch, slide_id_batch = data
                input_batch = input_batch.to(device)
                covariates_batch = np.concatenate([np.array(clindata['covariates'][clindata['covariates']['ID_slide']== file].iloc[:,1:]).astype(np.float32) for file in slide_id_batch])            
                covariates_batch = torch.from_numpy(covariates_batch).to(device)
                output_batch1, output_batch2,output_batch3,output_batch4,output_batch5 = net(input_batch,covariates_batch)                              
                output_batch1 = output_batch1.data.cpu()
                output_batch2 = output_batch2.data.cpu()
                output_batch3 = output_batch3.data.cpu()
                output_batch4 = output_batch4.data.cpu()
                output_batch5 = output_batch5.data.cpu()
                val_loss = criterion(output_batch1,torch.unsqueeze(po1_batch,axis=1) )
                val_loss += criterion(output_batch2,torch.unsqueeze(po2_batch,axis=1)  )
                val_loss += criterion(output_batch3,torch.unsqueeze(po3_batch,axis=1)  )
                val_loss += criterion(output_batch4,torch.unsqueeze(po4_batch,axis=1)  )
                val_loss += criterion(output_batch5,torch.unsqueeze(po5_batch,axis=1)  ) 
                val_running_loss += val_loss.item()              
                val_mae1 = np.concatenate([val_mae1, np.squeeze(abs(output_batch1-torch.unsqueeze(po1_batch,axis=1) ),axis=1)] )
                val_mae2 = np.concatenate([val_mae2,np.squeeze(abs(output_batch2-torch.unsqueeze(po2_batch,axis=1) ),axis=1)] ) 
                val_mae3 = np.concatenate([val_mae3,np.squeeze(abs(output_batch3-torch.unsqueeze(po3_batch,axis=1) ),axis=1)] ) 
                val_mae4 = np.concatenate([val_mae4,np.squeeze(abs(output_batch4-torch.unsqueeze(po4_batch,axis=1) ),axis=1)] ) 
                val_mae5 = np.concatenate([val_mae5,np.squeeze(abs(output_batch5-torch.unsqueeze(po5_batch,axis=1) ),axis=1)] )
                val_steps += 1
                if i == int(50000/batch_size):
                    break                
        val_loss_epoch = val_running_loss / ( int(50000/batch_size) * batch_size ) 
        val_losses.append(val_loss_epoch)
        print("[%d] val loss: %.3f" % (epoch + 1, val_loss_epoch))
        val_mae_ave = ( np.mean(val_mae1 ) + np.mean(val_mae2 ) + np.mean(val_mae3 ) + np.mean(val_mae4 ) + np.mean(val_mae5 ) ) 
        print("[%d] val accur : %.3f" % (epoch + 1, val_mae_ave) ) 
        print(train_losses)
        print( val_losses  )
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((net.state_dict(), optimizer.state_dict()), path)
        tune.report( training_loss = train_loss_epoch ,validation_loss = val_loss_epoch, mae_target = val_mae_ave.item())
    print("Finished Training")
    
           
     
def test_accuracy_mo(net,device='cpu'):
	pred1_ind_ave = []
	pred2_ind_ave = []
	pred3_ind_ave = []
	pred4_ind_ave = []
	pred5_ind_ave = []
	pred1_ind_quant = []
	pred2_ind_quant = []
	pred3_ind_quant = []
	pred4_ind_quant = []
	pred5_ind_quant = []
	time = []
	event = []
	correct = 0
	step = 0
	batch_size = 2**2 # 2**8 2**9 or 2**7
	for file in clindata['slide_id_test']:
		time_point = []
		output1 =  np.array([])
		output2 =  np.array([])
		output3 =  np.array([])
		output4 =  np.array([])
		output5 =  np.array([])
		slide_id =  np.array([])
		id_patient = (os.path.split(file)[-1][8:12]) 
		foldernames = os.path.join(args.data_dir_test,id_patient)
		filenames_patient= os.listdir(foldernames)
		filenames_patient = [os.path.join(foldernames, f) for f in filenames_patient if f.endswith('.jpg')   ]
		dl = DataLoader(Dataset_test(filenames_patient,clindata['test'],eval_transformer),batch_size=batch_size,shuffle=True, num_workers=4,pin_memory=True) 
		with torch.no_grad():
			for i,(input_batch ,time_batch, event_batch,  slide_id_batch) in enumerate(dl,0):           
				input_batch = input_batch.to(device)
				covariates_batch = np.concatenate([np.array(clindata['covariates'][clindata['covariates']['ID_slide']== file].iloc[:,1:]).astype(np.float32) for file in slide_id_batch])            
				covariates_batch = torch.from_numpy(covariates_batch).to(device)
				output_batch1, output_batch2,output_batch3,output_batch4,output_batch5 = net(input_batch.float(),covariates_batch)                                              
				output_batch1 = output_batch1.data.cpu()
				output_batch2 = output_batch2.data.cpu()
				output_batch3 = output_batch3.data.cpu()
				output_batch4 = output_batch4.data.cpu()
				output_batch5 = output_batch5.data.cpu()
				output1 = np.concatenate([output1, np.squeeze(output_batch1.detach().numpy(),axis=1)])
				output2 = np.concatenate([output2, np.squeeze(output_batch2.detach().numpy(),axis=1)])
				output3 = np.concatenate([output3, np.squeeze(output_batch3.detach().numpy(),axis=1)])
				output4 = np.concatenate([output4, np.squeeze(output_batch4.detach().numpy(),axis=1)])
				output5 = np.concatenate([output5, np.squeeze(output_batch5.detach().numpy(),axis=1)])				 
				slide_id = np.concatenate([slide_id, np.array(slide_id_batch)])                
				if i == int(10000/batch_size):
					break                           
		time.append(time_batch.numpy()[0])
		event.append(event_batch.numpy()[0])
		pred1_ind_ave.append(np.mean(output1))    
		pred2_ind_ave.append(np.mean(output2)) 
		pred3_ind_ave.append(np.mean(output3)) 
		pred4_ind_ave.append(np.mean(output4))        
		pred5_ind_ave.append(np.mean(output5))
		pred1_ind_quant.append(np.quantile(output1,.75))    
		pred2_ind_quant.append(np.quantile(output2,.75)) 
		pred3_ind_quant.append(np.quantile(output3,.75)) 
		pred4_ind_quant.append(np.quantile(output4,.75))        
		pred5_ind_quant.append(np.quantile(output5,.75))        
	y_surv_test = Surv.from_arrays(event=event, time=time)
	y_surv_train = Surv.from_arrays(event=np.squeeze(clindata['event_train'],axis=1), time=np.squeeze(clindata['time_train'],axis=1))
	auc1 = cumulative_dynamic_auc( y_surv_train, y_surv_test, pred1_ind_ave, 730)[0]  
	auc2 = cumulative_dynamic_auc( y_surv_train, y_surv_test, pred2_ind_ave, 1277)[0]
	auc3 = cumulative_dynamic_auc( y_surv_train, y_surv_test, pred3_ind_ave, 1825)[0]
	auc4 = cumulative_dynamic_auc( y_surv_train, y_surv_test, pred4_ind_ave, 2920)[0]
	auc5 = cumulative_dynamic_auc( y_surv_train, y_surv_test, pred5_ind_ave, 3650)[0]
	auc_average = {'2_year':auc1,'3.5_year': auc2, '5_year': auc3, '8_year':auc4, '10_year':auc5}   
	auc1 = cumulative_dynamic_auc( y_surv_train, y_surv_test, pred1_ind_quant, 730)[0]  
	auc2 = cumulative_dynamic_auc( y_surv_train, y_surv_test, pred2_ind_quant, 1277)[0]
	auc3 = cumulative_dynamic_auc( y_surv_train, y_surv_test, pred3_ind_quant, 1825)[0]
	auc4 = cumulative_dynamic_auc( y_surv_train, y_surv_test, pred4_ind_quant, 2920)[0]
	auc5 = cumulative_dynamic_auc( y_surv_train, y_surv_test, pred5_ind_quant, 3650)[0]
	auc_quantile = {'2_year':auc1,'3.5_year': auc2, '5_year': auc3, '8_year':auc4, '10_year':auc5}
	return auc_average, auc_quantile  
    
##################################################################################################
    
parser = argparse.ArgumentParser()
parser.add_argument('--max_num_epochs', type=int, default=30)
parser.add_argument('--num_samples', type=int, default=3)
parser.add_argument('--gpus_per_trial', type=int, default=1)
parser.add_argument('--cpus_per_trial', type=int, default=4)
parser.add_argument('--po_cnn', default='po') 
parser.add_argument('--implementation', default='single_output')
parser.add_argument('--data_dir_train', default = './data/Macenko_new_normed_TCGA_size_1196_stride_598_resize_1' )
parser.add_argument('--data_dir_test', default = './data/test_set/Macenko_new_normed_TCGA_size_1196_stride_598_resize_1')


if __name__ == "__main__":

    args = parser.parse_args()
    
    ################# clinical data  ########################
    df = pd.read_csv('./TCGA_clinical.csv', dtype='object', header=0)
    tcga_clin_orig = df[['Sample ID Slide']+['days_to_death']+['days_to_last_follow_up']+['vital_status']+['race']+
                     ['ethnicity']+['age_at_index']+['ajcc_pathologic_stage']+['BRCA_Subtype_PAM50','Test']]
    tcga_clin=tcga_clin_orig.assign(time_to_event = np.where(pd.isna(tcga_clin_orig['days_to_last_follow_up']),tcga_clin_orig['days_to_death'],tcga_clin_orig['days_to_last_follow_up']).astype(float),  
                     event = np.where(tcga_clin_orig['vital_status'] =='Dead',1,0).astype(float),
                     race_group= np.where(tcga_clin_orig['race']=='white','white', np.where(tcga_clin_orig['race']=='black or african american','black','other')),
                     ethnicity_group= np.where(tcga_clin_orig['ethnicity']=='not hispanic or latino','not_hispanic_latino','other'),
                     pathologic_stage =np.where(tcga_clin_orig['ajcc_pathologic_stage']=='Stage IA','Stage I',
                                                   np.where(tcga_clin_orig['ajcc_pathologic_stage']=='Stage IB','Stage I',    
                                                   np.where(tcga_clin_orig['ajcc_pathologic_stage']=='Stage IIA','Stage II',
                                                   np.where(tcga_clin_orig['ajcc_pathologic_stage']=='Stage IIB','Stage II',
                                                   np.where(tcga_clin_orig['ajcc_pathologic_stage']=='Stage IIIA','Stage III',
                                                   np.where(tcga_clin_orig['ajcc_pathologic_stage']=='Stage IIIB','Stage III',
                                                   np.where(tcga_clin_orig['ajcc_pathologic_stage']=='Stage IIIC','Stage III', 'StageX'
                                                   ) ) ) ) ) ) )                
                     )                                                   
    tcga_clin = tcga_clin.drop(['vital_status','race','ethnicity','days_to_last_follow_up','days_to_death','ajcc_pathologic_stage'],axis=1)
    tcga_clin = tcga_clin.rename(columns={"race_group": "race", 'ethnicity_group': 'ethnicity','Sample ID Slide' : 'ID_slide', 'age_at_index': 'age','BRCA_Subtype_PAM50':'molecular_subtype'})
    tcga_clin = tcga_clin[ ['ID_slide'] + ['Test'] + ['time_to_event'] + ['event']+ 
                           ['age'] + ['race'] + ['ethnicity'] + ['pathologic_stage'] + ['molecular_subtype'] ]                                            
    tcga_clin=tcga_clin.assign(time_to_event = np.where(tcga_clin['time_to_event']==0,0.5,tcga_clin['time_to_event']))
    tcga_clin.drop_duplicates(subset = 'ID_slide',inplace=True)
    cutoff = np.array([730,1277,1825, 2920, 3650])
    
    if args.po_cnn == 'po':
        clindata = df_c.get_clinical_data_po(tcga_clin, cutoff)
    else :
        clindata = df_c.get_clinical_data_ipcwpo(tcga_clin, cutoff)
    ####################################################
    
    config = {
        "lr": tune.loguniform(1e-5, 1e-3)        
    }
    scheduler = ASHAScheduler(
        metric="mae_target",
        mode="min",
        max_t= args.max_num_epochs,
        grace_period=10,
        reduction_factor=4)
    reporter = CLIReporter(
        metric_columns=["mae_target", "training_iteration"])
    if args.implementation == 'single_output':
        result = tune.run(
            partial(train_resnet_so, data_dir=args.data_dir_train),
            resources_per_trial={"cpu": args.cpus_per_trial, "gpu": args.gpus_per_trial},
            config= config,
            num_samples= args.num_samples,
            name="po_cnn_multi_output",
            scheduler= scheduler,
            progress_reporter=reporter)   
        best_trial = result.get_best_trial("mae_target", "min", "last")
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final validation loss: {}".format(best_trial.last_result["mae_target"]))
        best_trained_model = Resnet18_so(use_pretrained=False) # single output resnet
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
            if gpus_per_trial > 1:
                best_trained_model = nn.DataParallel(best_trained_model)
        best_trained_model.to(device)
        best_checkpoint_dir = best_trial.checkpoint.value
        model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))            
        best_trained_model.load_state_dict(model_state)
        auc_average, auc_quantile = test_accuracy_so(best_trained_model, args.data_dir_test ,device)
        print("Best Trial test set accuracy average: {}".format(auc_average) )
        print("Best Trial test set accuracy quantile: {}".format(auc_quantile) )
    else :
        result = tune.run(
            partial(train_resnet_mo, data_dir=args.data_dir_train),
            resources_per_trial={"cpu": args.cpus_per_trial, "gpu": args.gpus_per_trial},
            config= config,
            num_samples= args.num_samples,
            name="po_cnn_multi_output",
            scheduler= scheduler,
            progress_reporter=reporter)   
        best_trial = result.get_best_trial("mae_target", "min", "last")
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final validation loss: {}".format(best_trial.last_result["mae_target"]))
        best_trained_model = Resnet18_mo(use_pretrained=False) # multi output resnet
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
            if gpus_per_trial > 1:
                best_trained_model = nn.DataParallel(best_trained_model)
        best_trained_model.to(device)
        best_checkpoint_dir = best_trial.checkpoint.value
        model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))            
        best_trained_model.load_state_dict(model_state)
        auc_average, auc_quantile = test_accuracy_mo(best_trained_model, args.data_dir_test ,device)
        print("Best Trial test set accuracy average: {}".format(auc_average) )
        print("Best Trial test set accuracy quantile: {}".format(auc_quantile) )
