import numpy as np
import pandas as pd
import random

from rpy2.robjects.packages import importr
utils = importr('utils')
#utils.install_packages('prodlim')
prodlim = importr('prodlim')
eventglm = importr('eventglm')
#utils.install_packages('eventglm')
import rpy2.robjects as robjects
from rpy2.robjects import r

def get_clinical_data_ipcwpo(df, cutoff):
    data_converted = pd.get_dummies(df,columns= ['race' ,'ethnicity' ,'pathologic_stage' ,'molecular_subtype'],dtype=float)
    data_converted['event1'] = data_converted['event'] * np.where(data_converted['time_to_event']<cutoff[0],1,0)
    data_converted['event2'] = data_converted['event'] * np.where(data_converted['time_to_event']<cutoff[1],1,0)
    data_converted['event3'] = data_converted['event'] * np.where(data_converted['time_to_event']<cutoff[2],1,0)
    data_converted['event4'] = data_converted['event'] * np.where(data_converted['time_to_event']<cutoff[3],1,0)
    data_converted['event5'] = data_converted['event'] * np.where(data_converted['time_to_event']<cutoff[4],1,0)
    train_clindata_all = data_converted[data_converted['Test']=='False']
    test_clindata = data_converted[data_converted['Test']=='True']
    cols_cov = data_converted.columns.tolist()[4:-5]
    covariates = data_converted[['ID_slide'] + cols_cov]
    #split thge data
    idx_train = np.arange(len(train_clindata_all))
    random.seed(5)
    random.shuffle(idx_train)
    split = int(.8*len(idx_train))
    train_idx = idx_train[:split]
    val_idx = idx_train[split:]
    train_clindata = train_clindata_all.iloc[train_idx,]
    val_clindata = train_clindata_all.iloc[val_idx,]
    # Train
    order_time_train = np.argsort(train_clindata['time_to_event'])
    train_clindata = train_clindata.iloc[order_time_train,:]
    time_r = robjects.FloatVector(train_clindata['time_to_event'])
    event_r = robjects.BoolVector(train_clindata['event'])
    cutoff_r = robjects.FloatVector(cutoff)
    # this is for making that variable available to R
    robjects.globalenv["time_r"] = time_r
    robjects.globalenv["event_r"] = event_r
    robjects.globalenv["cutoff"] = cutoff_r
    #Pseudo Observation IPCW
    for i in (np.arange(15)+4):
        exec(f'cov_{i-3} = robjects.FloatVector(train_clindata.iloc[:,i]) ')
        exec(f'robjects.globalenv["cov_{i-3}"] = cov_{i-3}')
    r('xnam <- paste("cov_", 0:9, sep="")')
    r('data_surv_pso <- data.frame("time_r"=time_r,"event_r"=event_r ,  "cov_0"=cov_1, "cov_1"=cov_2, "cov_2"=cov_3, "cov_3"=cov_4,"cov_4"=cov_5,"cov_5"=cov_6,"cov_6"=cov_7, "cov_7"=cov_8, "cov_8"=cov_9,"cov_9"=cov_10, "cov_10"=cov_11,"cov_11"=cov_12,"cov_12"=cov_13,"cov_13"=cov_14,"cov_14"=cov_15) ' )
    r('POi_1 <- eventglm::pseudo_coxph(Surv(time_r, event_r) ~ 1, cutoff[1], cause = 1,data = data_surv_pso, type = "cuminc", formula.censoring = ~ cov_0+cov_1+cov_2+cov_3+cov_4+cov_5+cov_6+cov_7+cov_8+cov_9+cov_10+cov_11+cov_12+cov_13+cov_14  , ipcw.method = "hajek" )' )
    r('POi_2 <- eventglm::pseudo_coxph(Surv(time_r, event_r) ~ 1, cutoff[2], cause = 1,data = data_surv_pso, type = "cuminc", formula.censoring = ~ cov_0+cov_1+cov_2+cov_3+cov_4+cov_5+cov_6+cov_7+cov_8+cov_9+cov_10+cov_11+cov_12+cov_13+cov_14 , ipcw.method = "hajek" )' )
    r('POi_3 <- eventglm::pseudo_coxph(Surv(time_r, event_r) ~ 1, cutoff[3], cause = 1,data = data_surv_pso, type = "cuminc", formula.censoring = ~ cov_0+cov_1+cov_2+cov_3+cov_4+cov_5+cov_6+cov_7+cov_8+cov_9+cov_10+cov_11+cov_12+cov_13+cov_14 , ipcw.method = "hajek" )' )
    r('POi_4 <- eventglm::pseudo_coxph(Surv(time_r, event_r) ~ 1, cutoff[4], cause = 1,data = data_surv_pso, type = "cuminc", formula.censoring = ~ cov_0+cov_1+cov_2+cov_3+cov_4+cov_5+cov_6+cov_7+cov_8+cov_9+cov_10+cov_11+cov_12+cov_13+cov_14 , ipcw.method = "hajek" )' )
    r('POi_5 <- eventglm::pseudo_coxph(Surv(time_r, event_r) ~ 1, cutoff[5], cause = 1,data = data_surv_pso, type = "cuminc", formula.censoring = ~ cov_0+cov_1+cov_2+cov_3+cov_4+cov_5+cov_6+cov_7+cov_8+cov_9+cov_10+cov_11+cov_12+cov_13+cov_14 , ipcw.method = "hajek" )' )
    POi_1 = r('matrix(POi_1)')
    POi_2 = r('matrix(POi_2)')
    POi_3 = r('matrix(POi_3)')
    POi_4 = r('matrix(POi_4)')
    POi_5 = r('matrix(POi_5)')
    train_clindata = train_clindata.assign(risk_pso1 = np.array(POi_1,dtype=np.float64) ,
        risk_pso2 = np.array(POi_2,dtype=np.float64),
        risk_pso3 = np.array(POi_3,dtype=np.float64),
        risk_pso4 = np.array(POi_4,dtype=np.float64),
        risk_pso5 = np.array(POi_5,dtype=np.float64)
        )
    train_clindata = train_clindata[['ID_slide']+['risk_pso1']+['risk_pso2']+['risk_pso3']+['risk_pso4']+ ['risk_pso5']]
    # Validation
    order_time_val = np.argsort(val_clindata['time_to_event'])
    val_clindata = val_clindata.iloc[order_time_val,:]
    time_r = robjects.FloatVector(val_clindata['time_to_event'])
    event_r = robjects.BoolVector(val_clindata['event'])
    # this is for making that variable available to R
    robjects.globalenv["time_r"] = time_r
    robjects.globalenv["event_r"] = event_r
    #Pseudo Observation IPCW
    for i in (np.arange(15)+4):
        exec(f'cov_{i-3} = robjects.FloatVector(val_clindata.iloc[:,i]) ')
        exec(f'robjects.globalenv["cov_{i-3}"] = cov_{i-3}')
    r('xnam <- paste("cov_", 0:9, sep="")')
    r('data_surv_pso <- data.frame("time_r"=time_r,"event_r"=event_r ,  "cov_0"=cov_1, "cov_1"=cov_2, "cov_2"=cov_3, "cov_3"=cov_4,"cov_4"=cov_5,"cov_5"=cov_6,"cov_6"=cov_7, "cov_7"=cov_8, "cov_8"=cov_9,"cov_9"=cov_10, "cov_10"=cov_11,"cov_11"=cov_12,"cov_12"=cov_13,"cov_13"=cov_14,"cov_14"=cov_15) ' )
    r('POi_1 <- eventglm::pseudo_coxph(Surv(time_r, event_r) ~ 1, cutoff[1], cause = 1,data = data_surv_pso, type = "cuminc", formula.censoring = ~ cov_0+cov_1+cov_2+cov_3+cov_4+cov_5+cov_6+cov_7+cov_8+cov_9+cov_10+cov_11+cov_12+cov_13+cov_14  , ipcw.method = "hajek" )' )
    r('POi_2 <- eventglm::pseudo_coxph(Surv(time_r, event_r) ~ 1, cutoff[2], cause = 1,data = data_surv_pso, type = "cuminc", formula.censoring = ~ cov_0+cov_1+cov_2+cov_3+cov_4+cov_5+cov_6+cov_7+cov_8+cov_9+cov_10+cov_11+cov_12+cov_13+cov_14 , ipcw.method = "hajek" )' )
    r('POi_3 <- eventglm::pseudo_coxph(Surv(time_r, event_r) ~ 1, cutoff[3], cause = 1,data = data_surv_pso, type = "cuminc", formula.censoring = ~ cov_0+cov_1+cov_2+cov_3+cov_4+cov_5+cov_6+cov_7+cov_8+cov_9+cov_10+cov_11+cov_12+cov_13+cov_14 , ipcw.method = "hajek" )' )
    r('POi_4 <- eventglm::pseudo_coxph(Surv(time_r, event_r) ~ 1, cutoff[4], cause = 1,data = data_surv_pso, type = "cuminc", formula.censoring = ~ cov_0+cov_1+cov_2+cov_3+cov_4+cov_5+cov_6+cov_7+cov_8+cov_9+cov_10+cov_11+cov_12+cov_13+cov_14 , ipcw.method = "hajek" )' )
    r('POi_5 <- eventglm::pseudo_coxph(Surv(time_r, event_r) ~ 1, cutoff[5], cause = 1,data = data_surv_pso, type = "cuminc", formula.censoring = ~ cov_0+cov_1+cov_2+cov_3+cov_4+cov_5+cov_6+cov_7+cov_8+cov_9+cov_10+cov_11+cov_12+cov_13+cov_14 , ipcw.method = "hajek" )' )
    POi_1 = r('matrix(POi_1)')
    POi_2 = r('matrix(POi_2)')
    POi_3 = r('matrix(POi_3)')
    POi_4 = r('matrix(POi_4)')
    POi_5 = r('matrix(POi_5)')
    val_clindata = val_clindata.assign(risk_pso1 = np.array(POi_1,dtype=np.float64) ,
        risk_pso2 = np.array(POi_2,dtype=np.float64),
        risk_pso3 = np.array(POi_3,dtype=np.float64),
        risk_pso4 = np.array(POi_4,dtype=np.float64),
        risk_pso5 = np.array(POi_5,dtype=np.float64)
        )
    val_clindata = val_clindata[['ID_slide']+['risk_pso1']+['risk_pso2']+['risk_pso3']+['risk_pso4']+ ['risk_pso5']]
    test_clindata = test_clindata[ ['ID_slide'] +  ['time_to_event'] + ['event'] ]
    events = data_converted[['ID_slide'] + ['event1'] + ['event2'] + ['event3'] + ['event4'] +['event5']  ]
    clindata = {'train': train_clindata ,'val': val_clindata, 'test':test_clindata, 'covariates': covariates, 'time_train': train_clindata_all[['time_to_event']], 'event_train': train_clindata_all[['event']], 'slide_id_test': test_clindata['ID_slide'] , 'events': events }
    return clindata

	
	
from functools import partial
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
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



import torch
import torch.nn as nn
from torchvision import models

from sklearn.metrics import roc_auc_score
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index


import pandas as pd
import random

import json 
import os


import argparse
import random
import os

from PIL import Image


from itertools import compress 


from rpy2.robjects.packages import importr
utils = importr('utils')
#utils.install_packages('pseudo')
pseudo = importr('pseudo')
#utils.install_packages('prodlim')
prodlim = importr('prodlim')
#utils.install_packages('survival')
survival = importr('survival')
#utils.install_packages('KMsurv')
KMsurv = importr('KMsurv')
#utils.install_packages('pROC')
cvAUC = importr('pROC')
survivalROC = importr('survivalROC')
reshape2 = importr('reshape2')
import rpy2.robjects as robjects
from rpy2.robjects import r

#Load clinical data
df = pd.read_csv('/home/data/TCGA_clinical.csv', dtype='object', header=0)

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
clindata = get_clinical_data_ipcwpo(tcga_clin, cutoff)

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

# loader for evaluation, no horizontal flip
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

        
feature_extract = True

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
    

parser = argparse.ArgumentParser()
parser.add_argument('--max_num_epochs', type=int, default=30)
parser.add_argument('--num_samples', type=int, default=3)
parser.add_argument('--gpus_per_trial', type=int, default=1)
parser.add_argument('--cpus_per_trial', type=int, default=4)

#parser.add_argument('--implementation', default='single_output')

parser.add_argument('--data_dir_train', default = '/home/data/Macenko_new_normed_TCGA_size_1196_stride_598_resize_1' )
parser.add_argument('--data_dir_test', default = '/home/data/test_set/Macenko_new_normed_TCGA_size_1196_stride_598_resize_1')
args = parser.parse_args()


config = {
        "lr": tune.loguniform(1e-7, 1e-4)        
    }

scheduler = ASHAScheduler(
	metric="mae_target",
	mode="min",
	max_t= args.max_num_epochs,
	grace_period=10,
	reduction_factor=4)

reporter = CLIReporter(
	metric_columns=["training_loss","validation_loss","mae_target", "training_iteration"])
	
result = tune.run(
            partial(train_resnet_so, data_dir=args.data_dir_train),
            resources_per_trial={"cpu": args.cpus_per_trial, "gpu": args.gpus_per_trial},
            config= config,
            num_samples= args.num_samples,
            name="ipcwpocnn_single_output_freezing",
            scheduler= scheduler,
            progress_reporter=reporter)
            
best_trial = result.get_best_trial("mae_target", "min", "last")
print("Best trial config: {}".format(best_trial.config))
print("Best trial final validation loss: {}".format(best_trial.last_result["mae_target"]))
best_trained_model = Resnet18_so(use_pretrained=False)
device = "cpu"
if torch.cuda.is_available():
	device = "cuda"
	if gpus_per_trial > 1:
		best_trained_model = nn.DataParallel(best_trained_model)
best_trained_model.to(device)
best_checkpoint_dir = best_trial.checkpoint.value
model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))  

model_state, optimizer_state = torch.load(os.path.join('', "checkpoint"))        

best_trained_model.load_state_dict(model_state)
auc_average, auc_quantile = test_accuracy_so(best_trained_model, device)