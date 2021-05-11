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



class CoxDataset(Dataset):
    
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
        self.transform = transform
    def __len__(self):
        return len(self.filenames)
    def __getitem__(self, idx):
        #ID = self.slide_id[idx]
        image = Image.open(self.clinical_data.iloc[idx,1]) # PIL image
        image = self.transform(image)
        time = np.array(self.clinical_data.iloc[idx,2] ).astype(np.float32)
        event = np.array(self.clinical_data.iloc[idx,3] ).astype(np.float32)        
        slide_id = self.clinical_data.iloc[idx,0]
                
        return image,time, event,  slide_id
        
class Resnet18(nn.Module):
    def __init__(self, use_pretrained):
        super(Resnet18, self).__init__()
        model_ft = models.resnet18(pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features
        image_modules = list(model_ft.children())[:-1]
        self.modelA = nn.Sequential(*image_modules)
        self.fc = nn.Linear(num_ftrs  + 15, 1) # cov 
        
    def forward(self, image,clin_covariates ):
        x = self.modelA(image)
        x = torch.flatten(x, 1)
        x = torch.cat((x, clin_covariates), dim=1)
        x1 = self.fc(x)
        x1 = torch.tanh(x1)        
        
        return x1
        
#COX 
#time must be order in descending time
def make_riskset(time):
         o = np.argsort(-time, kind="mergesort")
         n_samples = len(time)
         risk_set = np.zeros((n_samples, n_samples), dtype=np.bool_)
         for i_org, i_sort in enumerate(o):
            ti = time[i_sort]
            k = i_org
            while k < n_samples and ti <= time[o[k]]:
                k += 1
            risk_set[i_sort, o[:k]] = True         
         return torch.from_numpy(risk_set)

def logsumexp_masked(risk_scores,
                     riskset):                     
    """Compute logsumexp across `axis` for entries where `mask` is true."""
    risk_score_masked = torch.mul(risk_scores, riskset).float()
    # for numerical stability, substract the maximum value
    # before taking the exponential
    amax = torch.max(risk_score_masked, 1)[0]
    risk_score_shift = risk_score_masked.sub(amax)
    exp_masked = torch.mul(risk_score_shift.exp(), riskset)
    exp_sum = torch.sum(exp_masked,1)
    output = amax + torch.log(exp_sum)      
    return output


class CoxPH_Loss(torch.nn.Module):    
    def __init__(self):
        super(CoxPH_Loss,self).__init__()        
    def forward(self,outputs,labels):
        num_examples = outputs.size()[0]
        events = labels[0]
        riskset = labels[1]
        if events.dtype is torch.bool:
            events = events.type(outputs.dtype)
        outputs_t = torch.transpose(outputs,0,1)
        log_cumsum_h = logsumexp_masked(outputs_t, riskset)
        neg_loss =  torch.sum(torch.mul(events, log_cumsum_h-outputs.view(-1)))/num_examples
        return neg_loss

########################  Using the aggregation criteria Average   ####################################################      

def train_resnet_cox(config, checkpoint_dir=None, data_dir=None):
    net = Resnet18(use_pretrained=True)
    if torch.cuda.is_available():
        device = "cuda"  
    net.to(device)
    criterion = CoxPH_Loss()
    optimizer = optim.Adam(net.parameters(),lr=config['lr'])
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)        
    batch_size = 2**4 # 2**8 2**9 or 2**7
    id_patient = [(os.path.split(filename)[-1][8:12]) for filename in clindata['train']['ID_slide']]
    max_tiles = np.max(how_many_tiles(args.data_dir_train,id_patient))
    filenames = get_filenames(args.data_dir_train,id_patient, max_tiles)
    ds = CoxDataset(filenames,clindata['train'],train_transformer)  
    for epoch in range(30):  # loop over the dataset multiple times
        trainloader = DataLoader(ds,batch_size=batch_size,shuffle=True, num_workers=4,pin_memory=True) # because we use 50000 samples shuffled
        running_loss = 0.0
        epoch_steps = 0
        net.train()
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            input_batch, time_batch, event_batch,  slide_id_batch = data          
            input_batch = input_batch.to(device)
            covariates_batch = np.concatenate([np.array(clindata['covariates'][clindata['covariates']['ID_slide']== file].iloc[:,1:]).astype(np.float32) for file in slide_id_batch])            
            covariates_batch = torch.from_numpy(covariates_batch).to(device)
            riskset = make_riskset(time_batch)
            riskset = riskset.to(device)
            event_batch = event_batch.to(device)
            optimizer.zero_grad()
            output_batch = net(input_batch, covariates_batch)
            loss = criterion(output_batch, [event_batch, riskset] )
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,running_loss / epoch_steps))
                running_loss = 0.0
            if i == int(50000/batch_size):
                break                
        # Validation loss       
        slide_id =  np.array([])
        time =  np.array([])
        event = np.array([])
        output =  np.array([])
        val_mae = 0.0
        val_steps = 0.0
        batch_size = 2**2 # 2**8 2**9 or 2**7
        id_patient = [(os.path.split(filename)[-1][8:12]) for filename in clindata['val']['ID_slide']]
        max_tiles = np.max(how_many_tiles(args.data_dir_train,id_patient))
        filenames = get_filenames(args.data_dir_train,id_patient, max_tiles)
        ds_val = CoxDataset(filenames,clindata['val'],eval_transformer)       
        valloader = DataLoader(ds_val,batch_size=batch_size,shuffle=True, num_workers=4,pin_memory=True)        
        net.eval()
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                input_batch, time_batch, event_batch,  slide_id_batch = data
                input_batch = input_batch.to(device)
                covariates_batch = np.concatenate([np.array(clindata['covariates'][clindata['covariates']['ID_slide']== file].iloc[:,1:]).astype(np.float32) for file in slide_id_batch])            
                covariates_batch = torch.from_numpy(covariates_batch).to(device)
                slide_id = np.concatenate([slide_id, slide_id_batch])
                time = np.concatenate([time, time_batch])
                event = np.concatenate([event, event_batch])
                output_batch = net(input_batch, covariates_batch)                
                output_batch = output_batch.data.cpu() 
                output = np.concatenate([output, np.squeeze(output_batch.detach().numpy(),axis=1)])                
                val_steps += 1
                if i == int(50000/batch_size):
                    break
        pred_ind = []
        time_r = []
        event_r = []
        for file in list(set(slide_id)):
            time_r.append( np.array(clindata['val']['time_to_event'][clindata['val']['ID_slide']== file ]) )
            event_r.append(np.array( clindata['val']['event'][clindata['val']['ID_slide']== file ])  )    
            pred_ind.append(np.mean(output[np.array(slide_id)== file]))    
        y_surv_test = Surv.from_arrays(event=np.concatenate(event_r), time=np.concatenate(time_r))
        y_surv_train = Surv.from_arrays(event= np.array(clindata['train']['event'])  , time= np.array(clindata['train']['time_to_event'])  )
        auc1 = cumulative_dynamic_auc( y_surv_train, y_surv_test, pred_ind, 730)[0]  
        auc2 = cumulative_dynamic_auc( y_surv_train, y_surv_test, pred_ind, 1277)[0]
        auc3 = cumulative_dynamic_auc( y_surv_train, y_surv_test, pred_ind, 1825)[0]
        auc4 = cumulative_dynamic_auc( y_surv_train, y_surv_test, pred_ind, 2920)[0]
        auc5 = cumulative_dynamic_auc( y_surv_train, y_surv_test, pred_ind, 3650)[0]
        auc = (auc1 + auc2 + auc3 + auc4 + auc5) / 5
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((net.state_dict(), optimizer.state_dict()), path)
        tune.report(auc_target = auc.item())                
    print("Finished Training")


def test_accuracy(net,data_dir_test,device='cpu'):
    pred_ind_ave = []
    time = []
    event = []
    batch_size = 1 # 2**8 2**9 or 2**7
    for file in clindata['slide_id_test']:
        output =  np.array([])       
        slide_id =  np.array([])
        id_patient = (os.path.split(file)[-1][8:12]) 
        foldernames = os.path.join(data_dir_test,id_patient)
        filenames_patient= os.listdir(foldernames)
        filenames_patient = [os.path.join(foldernames, f) for f in filenames_patient if f.endswith('.jpg')   ]
        dl = DataLoader(CoxDataset(filenames_patient,clindata['test'],eval_transformer),batch_size=batch_size,shuffle=True, num_workers=4,pin_memory=True) 
        net.eval()
        with torch.no_grad():
            for i,(input_batch ,time_batch, event_batch,  slide_id_batch) in enumerate(dl,0):           
                input_batch = input_batch.to(device)       
                covariates_batch = np.concatenate([np.array(clindata['covariates'][clindata['covariates']['ID_slide']== file].iloc[:,1:]).astype(np.float32) for file in slide_id_batch])            
                covariates_batch = torch.from_numpy(covariates_batch).to(device)
                slide_id = np.concatenate([slide_id, slide_id_batch])
                output_batch = net(input_batch, covariates_batch)                
                output_batch = output_batch.data.cpu()                                                                                
                output = np.concatenate([output, np.squeeze(output_batch.detach().numpy(),axis=1)])              
                slide_id = np.concatenate([slide_id, np.array(slide_id_batch)])
                if i == int(1000/batch_size):
                      break
        time.append(time_batch.numpy()[0])
        event.append(event_batch.numpy()[0])
        pred_ind_ave.append(np.mean(output))                 
    y_surv_test = Surv.from_arrays(event=event, time=time)
    y_surv_train = Surv.from_arrays(event=np.squeeze(clindata['event_train'],axis=1), time=np.squeeze(clindata['time_train'],axis=1))
    auc1 = cumulative_dynamic_auc( y_surv_train, y_surv_test, pred_ind_ave, 730)[0]  
    auc2 = cumulative_dynamic_auc( y_surv_train, y_surv_test, pred_ind_ave, 1277)[0]
    auc3 = cumulative_dynamic_auc( y_surv_train, y_surv_test, pred_ind_ave, 1825)[0]
    auc4 = cumulative_dynamic_auc( y_surv_train, y_surv_test, pred_ind_ave, 2920)[0]
    auc5 = cumulative_dynamic_auc( y_surv_train, y_surv_test, pred_ind_ave, 3650)[0]
    auc_average = {'2_year':auc1,'3.5_year': auc2, '5_year': auc3, '8_year':auc4, '10_year':auc5}     
    return auc_average 

##################################### Using the aggregation criteria 75th percentile  ####################################################################################################

def train_resnet_quantile(config, checkpoint_dir=None, data_dir=None):
    net = Resnet18(use_pretrained=True)
    if torch.cuda.is_available():
        device = "cuda"  
    net.to(device)
    criterion = CoxPH_Loss()
    optimizer = optim.Adam(net.parameters(),lr=config['lr'])
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)        
    batch_size = 2**4 # 2**8 2**9 or 2**7
    id_patient = [(os.path.split(filename)[-1][8:12]) for filename in clindata['train']['ID_slide']]
    max_tiles = np.max(how_many_tiles(args.data_dir_train,id_patient))
    filenames = get_filenames(args.data_dir_train,id_patient, max_tiles)
    ds = CoxDataset(filenames,clindata['train'],train_transformer)  
    for epoch in range(30):  # loop over the dataset multiple times
        trainloader = DataLoader(ds,batch_size=batch_size,shuffle=True, num_workers=4,pin_memory=True) # because we use 50000 samples shuffled
        running_loss = 0.0
        epoch_steps = 0
        net.train()
        for i, data in enumerate(trainloader, 0):
            input_batch, time_batch, event_batch,  slide_id_batch = data          
            input_batch = input_batch.to(device)
            covariates_batch = np.concatenate([np.array(clindata['covariates'][clindata['covariates']['ID_slide']== file].iloc[:,1:]).astype(np.float32) for file in slide_id_batch])            
            covariates_batch = torch.from_numpy(covariates_batch).to(device)
            riskset = make_riskset(time_batch)
            riskset = riskset.to(device)
            event_batch = event_batch.to(device)
            optimizer.zero_grad()
            output_batch = net(input_batch, covariates_batch)
            loss = criterion(output_batch, [event_batch, riskset] )
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,running_loss / epoch_steps))
                running_loss = 0.0
            if i == int(50000/batch_size):
                break                
        # Validation loss       
        slide_id =  np.array([])
        time =  np.array([])
        event = np.array([])
        output =  np.array([])
        val_mae = 0.0
        val_steps = 0.0
        batch_size = 2**2 # 2**8 2**9 or 2**7
        id_patient = [(os.path.split(filename)[-1][8:12]) for filename in clindata['val']['ID_slide']]
        max_tiles = np.max(how_many_tiles(args.data_dir_train,id_patient))
        filenames = get_filenames(args.data_dir_train,id_patient, max_tiles)
        ds_val = CoxDataset(filenames,clindata['val'],eval_transformer)       
        valloader = DataLoader(ds_val,batch_size=batch_size,shuffle=True, num_workers=4,pin_memory=True)        
        net.eval()
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                input_batch, time_batch, event_batch,  slide_id_batch = data
                input_batch = input_batch.to(device)
                covariates_batch = np.concatenate([np.array(clindata['covariates'][clindata['covariates']['ID_slide']== file].iloc[:,1:]).astype(np.float32) for file in slide_id_batch])            
                covariates_batch = torch.from_numpy(covariates_batch).to(device)
                slide_id = np.concatenate([slide_id, slide_id_batch])
                time = np.concatenate([time, time_batch])
                event = np.concatenate([event, event_batch])
                output_batch = net(input_batch, covariates_batch)                
                output_batch = output_batch.data.cpu() 
                output = np.concatenate([output, np.squeeze(output_batch.detach().numpy(),axis=1)])                
                val_steps += 1
                if i == int(50000/batch_size):
                    break
        pred_ind = []
        time_r = []
        event_r = []
        for file in list(set(slide_id)):
            time_r.append( np.array(clindata['val']['time_to_event'][clindata['val']['ID_slide']== file ]) )
            event_r.append(np.array( clindata['val']['event'][clindata['val']['ID_slide']== file ])  )    
            pred_ind.append(np.quantile(output[np.array(slide_id)== file],.75))    
        y_surv_test = Surv.from_arrays(event=np.concatenate(event_r), time=np.concatenate(time_r))
        y_surv_train = Surv.from_arrays(event= np.array(clindata['train']['event'])  , time= np.array(clindata['train']['time_to_event'])  )
        auc1 = cumulative_dynamic_auc( y_surv_train, y_surv_test, pred_ind, 730)[0]  
        auc2 = cumulative_dynamic_auc( y_surv_train, y_surv_test, pred_ind, 1277)[0]
        auc3 = cumulative_dynamic_auc( y_surv_train, y_surv_test, pred_ind, 1825)[0]
        auc4 = cumulative_dynamic_auc( y_surv_train, y_surv_test, pred_ind, 2920)[0]
        auc5 = cumulative_dynamic_auc( y_surv_train, y_surv_test, pred_ind, 3650)[0]
        auc = (auc1 + auc2 + auc3 + auc4 + auc5) / 5
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((net.state_dict(), optimizer.state_dict()), path)
        tune.report(auc_target = auc.item())                
    print("Finished Training")



def test_accuracy_quantile(net,data_dir_test,device='cpu'):
    pred_ind_quant = []
    time = []
    event = []
    batch_size = 1 # 2**8 2**9 or 2**7
    for file in clindata['slide_id_test']:
        output =  np.array([])       
        slide_id =  np.array([])
        id_patient = (os.path.split(file)[-1][8:12]) 
        foldernames = os.path.join(data_dir_test,id_patient)
        filenames_patient= os.listdir(foldernames)
        filenames_patient = [os.path.join(foldernames, f) for f in filenames_patient if f.endswith('.jpg')   ]
        dl = DataLoader(CoxDataset(filenames_patient,clindata['test'],eval_transformer),batch_size=batch_size,shuffle=True, num_workers=4,pin_memory=True) 
        net.eval()
        with torch.no_grad():
            for i,(input_batch ,time_batch, event_batch,  slide_id_batch) in enumerate(dl,0):           
                input_batch = input_batch.to(device)       
                covariates_batch = np.concatenate([np.array(clindata['covariates'][clindata['covariates']['ID_slide']== file].iloc[:,1:]).astype(np.float32) for file in slide_id_batch])            
                covariates_batch = torch.from_numpy(covariates_batch).to(device)
                slide_id = np.concatenate([slide_id, slide_id_batch])
                output_batch = net(input_batch, covariates_batch)                
                output_batch = output_batch.data.cpu()                                                                                
                output = np.concatenate([output, np.squeeze(output_batch.detach().numpy(),axis=1)])              
                slide_id = np.concatenate([slide_id, np.array(slide_id_batch)])
                if i == int(1000/batch_size):
                      break
        time.append(time_batch.numpy()[0])
        event.append(event_batch.numpy()[0])
        pred_ind_quant.append(np.quantile(output,.75))                 
    y_surv_test = Surv.from_arrays(event=event, time=time)
    y_surv_train = Surv.from_arrays(event=np.squeeze(clindata['event_train'],axis=1), time=np.squeeze(clindata['time_train'],axis=1))
    auc1 = cumulative_dynamic_auc( y_surv_train, y_surv_test, pred_ind_quant, 730)[0]  
    auc2 = cumulative_dynamic_auc( y_surv_train, y_surv_test, pred_ind_quant, 1277)[0]
    auc3 = cumulative_dynamic_auc( y_surv_train, y_surv_test, pred_ind_quant, 1825)[0]
    auc4 = cumulative_dynamic_auc( y_surv_train, y_surv_test, pred_ind_quant, 2920)[0]
    auc5 = cumulative_dynamic_auc( y_surv_train, y_surv_test, pred_ind_quant, 3650)[0]
    auc_average = {'2_year':auc1,'3.5_year': auc2, '5_year': auc3, '8_year':auc4, '10_year':auc5}     
    return auc_average 

#########################################################################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--max_num_epochs', type=int, default=30)
parser.add_argument('--num_samples', type=int, default=3)
parser.add_argument('--gpus_per_trial', type=int, default=1)
parser.add_argument('--cpus_per_trial', type=int, default=4)
parser.add_argument('--aggregation_criteria', default = 'average')
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
    clindata = df_c.get_clinical_data_cox(tcga_clin, cutoff)

    ####################################################
    
    config = {
        "lr": tune.loguniform(1e-5, 1e-3)        
    }
    if args.aggregation_criteria == 'average':   
        scheduler = ASHAScheduler(
            metric="auc_target",
            mode="max",
            max_t= args.max_num_epochs,
            grace_period=10,
            reduction_factor=4)
        reporter = CLIReporter(
            metric_columns=["auc_target", "training_iteration"])
        result = tune.run(
            partial(train_resnet, data_dir=args.data_dir_train),
            resources_per_trial={"cpu": args.cpus_per_trial, "gpu": args.gpus_per_trial},
            config=config,
            num_samples= args.num_samples,
            name="coxcnn_resnet18",
            scheduler=scheduler,
            progress_reporter=reporter)   
        best_trial = result.get_best_trial("auc_target", "max", "last")
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final validation loss: {}".format(best_trial.last_result["auc_target"]))
        best_trained_model = Resnet18(use_pretrained=False)
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
            if gpus_per_trial > 1:
                best_trained_model = nn.DataParallel(best_trained_model)
        best_trained_model.to(device)
        best_checkpoint_dir = best_trial.checkpoint.value
        model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))            
        best_trained_model.load_state_dict(model_state)
        test_acc = test_accuracy(best_trained_model, args.data_dir_test,device)
        print("Best Trial test set accuracy: {}".format(test_acc) )
    else:
        scheduler = ASHAScheduler(
            metric="auc_target",
            mode="max",
            max_t= args.max_num_epochs,
            grace_period=10,
            reduction_factor=4)
        reporter = CLIReporter(
            metric_columns=["auc_target", "training_iteration"])
        result = tune.run(
            partial(train_resnet_quantile, data_dir=args.data_dir_train),
            resources_per_trial={"cpu": args.cpus_per_trial, "gpu": args.gpus_per_trial},
            config=config,
            num_samples= args.num_samples,
            name="coxcnn_quantile_resnet18",
            scheduler=scheduler,
            progress_reporter=reporter)   
        best_trial = result.get_best_trial("auc_target", "max", "last")
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final validation loss: {}".format(best_trial.last_result["auc_target"]))
        best_trained_model = Resnet18(use_pretrained=False)
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
            if gpus_per_trial > 1:
                best_trained_model = nn.DataParallel(best_trained_model)
        best_trained_model.to(device)
        best_checkpoint_dir = best_trial.checkpoint.value
        model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))            
        best_trained_model.load_state_dict(model_state)
        test_acc = test_accuracy_quantile(best_trained_model, args.data_dir_test,device)
        print("Best Trial test set accuracy: {}".format(test_acc) )
