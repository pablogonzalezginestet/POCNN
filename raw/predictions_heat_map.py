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

def get_clinical_data_po(df, cutoff):
    data_converted = pd.get_dummies(df,columns= ['race' ,'ethnicity' ,'pathologic_stage' ,'molecular_subtype'],dtype=float)
    data_converted['event1'] = data_converted['event'] * np.where(data_converted['time_to_event']<cutoff[0],1,0)
    data_converted['event2'] = data_converted['event'] * np.where(data_converted['time_to_event']<cutoff[1],1,0)
    data_converted['event3'] = data_converted['event'] * np.where(data_converted['time_to_event']<cutoff[2],1,0)
    data_converted['event4'] = data_converted['event'] * np.where(data_converted['time_to_event']<cutoff[3],1,0)
    data_converted['event5'] = data_converted['event'] * np.where(data_converted['time_to_event']<cutoff[4],1,0)
    train_clindata_all = data_converted[data_converted['Test']=='False']
    test_clindata = data_converted[data_converted['Test']=='True']
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
    #Pseudo Observation
    r('km_out <- prodlim(Hist(time_r,event_r)~1)')  
    r(' surv_pso <- jackknife(km_out,times=cutoff) ' )
    risk_pso1 = r('1-surv_pso[,1]')
    risk_pso2 = r('1-surv_pso[,2]')
    risk_pso3 = r('1-surv_pso[,3]')
    risk_pso4 = r('1-surv_pso[,4]')
    risk_pso5 = r('1-surv_pso[,5]')
    train_clindata = train_clindata.assign(risk_pso1 = np.array(risk_pso1,dtype=np.float64),
        risk_pso2 = np.array(risk_pso2,dtype=np.float64),
        risk_pso3 = np.array(risk_pso3,dtype=np.float64),
        risk_pso4 = np.array(risk_pso4,dtype=np.float64),
        risk_pso5 = np.array(risk_pso5,dtype=np.float64)
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
    #Pseudo Observation
    r('km_out <- prodlim(Hist(time_r,event_r)~1)')  
    r(' surv_pso <- jackknife(km_out,times=cutoff) ' )
    risk_pso1 = r('1-surv_pso[,1]')
    risk_pso2 = r('1-surv_pso[,2]')
    risk_pso3 = r('1-surv_pso[,3]')
    risk_pso4 = r('1-surv_pso[,4]')
    risk_pso5 = r('1-surv_pso[,5]')
    val_clindata = val_clindata.assign(risk_pso1 = np.array(risk_pso1,dtype=np.float64),
        risk_pso2 = np.array(risk_pso2,dtype=np.float64),
        risk_pso3 = np.array(risk_pso3,dtype=np.float64),
        risk_pso4 = np.array(risk_pso4,dtype=np.float64),
        risk_pso5 = np.array(risk_pso5,dtype=np.float64)
        )
    val_clindata = val_clindata[['ID_slide']+['risk_pso1']+['risk_pso2']+['risk_pso3']+['risk_pso4']+ ['risk_pso5']]
    test_clindata = test_clindata[ ['ID_slide'] +  ['time_to_event'] + ['event'] ]
    cols_cov = data_converted.columns.tolist()[4:-5]
    covariates = data_converted[['ID_slide'] + cols_cov]
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

#delete cache cuda
torch.cuda.empty_cache()
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
clindata = get_clinical_data_po(tcga_clin, cutoff)

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
        time = np.array(self.clinical_data.iloc[idx,2] ).astype(np.float32)
        event = np.array(self.clinical_data.iloc[idx,3] ).astype(np.float32)        
        slide_id = self.clinical_data.iloc[idx,0]
        file_id = self.filenames[idx]
        
                
        return image, time, event,  slide_id ,file_id  

       
        

def mean_abs_error(output,label):
    return torch.mean( torch.abs(label-output) )

metrics_train = {'mae': mean_abs_error}


###########################  Single Output #######################################################

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


parser = argparse.ArgumentParser()
parser.add_argument('--max_num_epochs', type=int, default=30)
parser.add_argument('--num_samples', type=int, default=3)
parser.add_argument('--gpus_per_trial', type=int, default=1)
parser.add_argument('--cpus_per_trial', type=int, default=4)
parser.add_argument('--po_cnn', default='po') 
#parser.add_argument('--implementation', default='single_output')

parser.add_argument('--data_dir_train', default = '/home/data/Macenko_new_normed_TCGA_size_1196_stride_598_resize_1' )
parser.add_argument('--data_dir_test', default = '/home/data/test_set/Macenko_new_normed_TCGA_size_1196_stride_598_resize_1')
args = parser.parse_args()

#>>> best_checkpoint_dir
'/home/pablo/ray_results/pocnn_single_output_freezing/DEFAULT_71497_00000_0_lr=0.0005905_2022-01-18_00-30-23/checkpoint_29/checkpoint'
            
best_trained_model = Resnet18_so(use_pretrained=False)
device = "cuda"
best_trained_model.to(device)
best_checkpoint_dir = '/home/pablo/ray_results/pocnn_single_output_freezing/DEFAULT_71497_00000_0_lr=0.0005905_2022-01-18_00-30-23/checkpoint_29/'
model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))  
#model_state, optimizer_state = torch.load(os.path.join('', "checkpoint"))        
best_trained_model.load_state_dict(model_state)

    
idx_split = np.arange(len(clindata['slide_id_test']))
random.seed(5)
random.shuffle(idx_split)
test_clindata_id = clindata['slide_id_test'].iloc[idx_split[:50],]   

batch_size = 1 # 2**8 2**9 or 2**7
output1 =  np.array([])
output2 =  np.array([])
output3 =  np.array([])
output4 =  np.array([])
slide_id =  np.array([])
file_id =  np.array([])
for file in test_clindata_id:
    id_patient = (os.path.split(file)[-1][8:12]) 
    foldernames = os.path.join(args.data_dir_test,id_patient)
    filenames_patient = os.listdir(foldernames)
    filenames_patient = [os.path.join(foldernames, f) for f in filenames_patient if f.endswith('.jpg')   ]
    dl = DataLoader(Dataset_test(filenames_patient,clindata['test'],eval_transformer),batch_size=batch_size,shuffle=False, num_workers=4,pin_memory=True) 
    net = best_trained_model
    net.eval()
    with torch.no_grad():
        for i,(input_batch ,time_batch, event_batch,  slide_id_batch, file_id_batch) in enumerate(dl,0):           
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
            slide_id = np.concatenate([slide_id, np.array(slide_id_batch)])
            file_id = np.concatenate([file_id, np.array(file_id_batch)])
   
#auc1_json_path = os.path.join(model_dir,"auc1_po_so")
np.save('/home/pablo/ray_results/ipcwpocnn_single_output_freezing/prediction1_heat_map',output1)           
np.save('/home/pablo/ray_results/ipcwpocnn_single_output_freezing/prediction2_heat_map',output2)     
np.save('/home/pablo/ray_results/ipcwpocnn_single_output_freezing/prediction3_heat_map',output3)
np.save('/home/pablo/ray_results/ipcwpocnn_single_output_freezing/prediction4_heat_map',output4)
np.save('/home/pablo/ray_results/ipcwpocnn_single_output_freezing/file_id_heat_map',file_id)
np.save('/home/pablo/ray_results/ipcwpocnn_single_output_freezing/slide_id_heat_map',slide_id)
    
