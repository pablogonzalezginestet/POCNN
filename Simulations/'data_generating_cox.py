import numpy as np
import pandas as pd
import random


def sim_event_times_case1(trainset, num_samples):    
    train_n = int( .8 * num_samples)
    test_n = int( (.2) * num_samples) 
    cov =  np.random.standard_normal(size=(num_samples, 9))
    treatment = np.random.binomial(n=1,p=.5,size=num_samples)
    treatment=np.expand_dims(treatment,1)
    clinical_data = np.concatenate((treatment, cov), axis=1)
    index = np.arange(len(trainset.targets))
    idx_sample = np.random.choice(index, num_samples,replace=False)
    digits = np.array(trainset.targets)[idx_sample]
    denom = np.exp( 1.7* digits+ .6*np.cos(digits)*clinical_data[:,0]+.2*clinical_data[:,1]+.3*clinical_data[:,0] )
    true_times = np.sqrt(-np.log( np.random.uniform(low=0,high=1,size=num_samples) )/ denom )
    censored_times = np.random.uniform(low=0,high=true_times)
    censored_indicator = np.random.binomial(n=1,p=.3,size=digits.shape[0])
    times = np.where(censored_indicator==1, censored_times,true_times) 
    event = np.where(censored_indicator==1,0,1) 
    cutoff = np.array(np.quantile(true_times,(.2,.3,.4,.5,.6)))
    event_1= np.where(true_times<= cutoff[0],1,0)
    event_2= np.where(true_times<= cutoff[1],1,0)
    event_3= np.where(true_times<= cutoff[2],1,0)
    event_4= np.where(true_times<= cutoff[3],1,0)
    event_5= np.where(true_times<= cutoff[4],1,0)    
    cens_perc = np.sum(censored_indicator)/num_samples
    cens_perc_train = np.sum(censored_indicator[:train_n])/train_n
    df = np.concatenate((np.expand_dims(idx_sample,axis=1), np.expand_dims(times,axis=1),np.expand_dims(event,axis=1),
    np.expand_dims(event_1,axis=1),np.expand_dims(event_2,axis=1),np.expand_dims(event_3,axis=1),np.expand_dims(event_4,axis=1),np.expand_dims(event_5,axis=1),clinical_data),axis=1)
    df = pd.DataFrame(df,columns= ('ID','time','event','event_1','event_2','event_3','event_4','event_5','cov1','cov2','cov3','cov4','cov5','cov6','cov7','cov8','cov9','cov10')) # the ID is the image chosen
    train_clindata_all =  df.iloc[0:train_n,:]
    test_clindata_all = df.iloc[(train_n+1):,:]
    test_clindata = test_clindata_all[ ['ID']  + ['time'] + ['event'] +['event_1'] + ['event_2'] + ['event_3'] + ['event_4'] + ['event_5']]
    covariates = df[['ID'] +  df.columns.tolist()[8:]]
    idx_train = np.arange(len(train_clindata_all))
    random.seed(5)
    random.shuffle(idx_train)
    split = int(.8*len(idx_train))
    train_idx = idx_train[:split]
    val_idx = idx_train[split:]
    train_clindata = train_clindata_all.iloc[train_idx,0:3]
    val_clindata = train_clindata_all.iloc[val_idx,0:8]
    clindata = {'train_val':train_clindata_all ,'train': train_clindata ,'val': val_clindata, 'test':test_clindata, 'covariates': covariates,'time_train': train_clindata_all['time'], 'event_train': train_clindata_all['event'], 'slide_id_test': test_clindata_all['ID'], 'cutoff': cutoff , 'cens': cens_perc, 'cens_train': cens_perc_train}
    return clindata


def sim_event_times_case2(trainset, num_samples):    
    train_n = int( .8 * num_samples)
    test_n = int( (.2) * num_samples) 
    cov =  np.random.standard_normal(size=(num_samples, 9))
    treatment = np.random.binomial(n=1,p=.5,size=num_samples)
    treatment=np.expand_dims(treatment,1)
    clinical_data = np.concatenate((treatment, cov), axis=1)
    index = np.arange(len(trainset.targets))
    idx_sample = np.random.choice(index, num_samples,replace=False)
    digits = np.array(trainset.targets)[idx_sample]    
    denom = np.exp( 1.7* digits+ .6*np.cos(digits)*clinical_data[:,0]+.2*clinical_data[:,1]+.3*clinical_data[:,0] )
    true_times = np.sqrt(-np.log( np.random.uniform(low=0,high=1,size=num_samples) )/ denom )   
    denom = np.exp( 1.4*clinical_data[:,0]+2.6*clinical_data[:,1] -.2*clinical_data[:,2]   )*6
    censored_times = np.sqrt(-np.log(np.random.uniform(low=0,high=1,size=num_samples))/denom )
    censored_indicator = (true_times > censored_times)*1  
    times = np.where(censored_indicator==1, censored_times,true_times) 
    event = np.where(censored_indicator==1,0,1)
    cutoff = np.array(np.quantile(true_times,(.2,.3,.4,.5,.6)))
    event_1= np.where(true_times<= cutoff[0],1,0)
    event_2= np.where(true_times<= cutoff[1],1,0)
    event_3= np.where(true_times<= cutoff[2],1,0)
    event_4= np.where(true_times<= cutoff[3],1,0)
    event_5= np.where(true_times<= cutoff[4],1,0)    
    cens_perc = np.sum(censored_indicator)/num_samples
    cens_perc_train = np.sum(censored_indicator[:train_n])/train_n
    df = np.concatenate((np.expand_dims(idx_sample,axis=1), np.expand_dims(times,axis=1),np.expand_dims(event,axis=1),
    np.expand_dims(event_1,axis=1),np.expand_dims(event_2,axis=1),np.expand_dims(event_3,axis=1),np.expand_dims(event_4,axis=1),np.expand_dims(event_5,axis=1),clinical_data),axis=1)
    df = pd.DataFrame(df,columns= ('ID','time','event','event_1','event_2','event_3','event_4','event_5','cov1','cov2','cov3','cov4','cov5','cov6','cov7','cov8','cov9','cov10')) # the ID is the image chosen
    train_clindata_all =  df.iloc[0:train_n,:]
    test_clindata_all = df.iloc[(train_n+1):,:]
    test_clindata = test_clindata_all[ ['ID']  + ['time'] + ['event'] +['event_1'] + ['event_2'] + ['event_3'] + ['event_4'] + ['event_5']]
    covariates = df[['ID'] +  df.columns.tolist()[8:]]
    idx_train = np.arange(len(train_clindata_all))
    random.seed(5)
    random.shuffle(idx_train)
    split = int(.8*len(idx_train))
    train_idx = idx_train[:split]
    val_idx = idx_train[split:]
    train_clindata = train_clindata_all.iloc[train_idx,0:3]
    val_clindata = train_clindata_all.iloc[val_idx,0:8]
    clindata = {'train_val':train_clindata_all ,'train': train_clindata ,'val': val_clindata, 'test':test_clindata, 'covariates': covariates,'time_train': train_clindata_all['time'], 'event_train': train_clindata_all['event'], 'slide_id_test': test_clindata_all['ID'], 'cutoff': cutoff, 'cens': cens_perc, 'cens_train': cens_perc_train }
    return clindata 



def sim_event_times_case3(trainset, num_samples):
    train_n = int( .8 * num_samples)
    test_n = int( (.2) * num_samples) 
    cov =  np.random.standard_normal(size=(num_samples, 9))
    treatment = np.random.binomial(n=1,p=.5,size=num_samples)
    treatment=np.expand_dims(treatment,1)
    clinical_data = np.concatenate((treatment, cov), axis=1)
    index = np.arange(len(trainset.targets))
    idx_sample = np.random.choice(index, num_samples,replace=False)
    digits = np.array(trainset.targets)[idx_sample]    
    denom = np.exp( 1* digits- 1.6*np.cos(digits)*clinical_data[:,0]+.3*clinical_data[:,1]*clinical_data[:,0] )* (.7/2)
    true_times = np.sqrt(-np.log( np.random.uniform(low=0,high=1,size=num_samples) )/ denom )   
    denom = np.exp( 1.4*clinical_data[:,0]+2.6*clinical_data[:,1] -.2*clinical_data[:,2]   )*6
    shape_c = np.maximum(0.001,np.exp(-1.8*clinical_data[:,0]+1.4*clinical_data[:,1]+1.5 *clinical_data[:,0]*clinical_data[:,1]))
    censored_times = np.random.gamma(shape_c,digits, num_samples)
    censored_indicator = (true_times > censored_times)*1
    times = np.where(censored_indicator==1, censored_times,true_times) 
    event = np.where(censored_indicator==1,0,1)
    cutoff = np.array(np.quantile(true_times,(.2,.3,.4,.5,.6)))
    event_1= np.where(true_times<= cutoff[0],1,0)
    event_2= np.where(true_times<= cutoff[1],1,0)
    event_3= np.where(true_times<= cutoff[2],1,0)
    event_4= np.where(true_times<= cutoff[3],1,0)
    event_5= np.where(true_times<= cutoff[4],1,0)    
    cens_perc = np.sum(censored_indicator)/num_samples
    cens_perc_train = np.sum(censored_indicator[:train_n])/train_n
    df = np.concatenate((np.expand_dims(idx_sample,axis=1), np.expand_dims(times,axis=1),np.expand_dims(event,axis=1),
    np.expand_dims(event_1,axis=1),np.expand_dims(event_2,axis=1),np.expand_dims(event_3,axis=1),np.expand_dims(event_4,axis=1),np.expand_dims(event_5,axis=1),clinical_data),axis=1)
    df = pd.DataFrame(df,columns= ('ID','time','event','event_1','event_2','event_3','event_4','event_5','cov1','cov2','cov3','cov4','cov5','cov6','cov7','cov8','cov9','cov10')) # the ID is the image chosen
    train_clindata_all =  df.iloc[0:train_n,:]
    test_clindata_all = df.iloc[(train_n+1):,:]
    test_clindata = test_clindata_all[ ['ID']  + ['time'] + ['event'] +['event_1'] + ['event_2'] + ['event_3'] + ['event_4'] + ['event_5']]
    covariates = df[['ID'] +  df.columns.tolist()[8:]]
    idx_train = np.arange(len(train_clindata_all))
    random.seed(5)
    random.shuffle(idx_train)
    split = int(.8*len(idx_train))
    train_idx = idx_train[:split]
    val_idx = idx_train[split:]
    train_clindata = train_clindata_all.iloc[train_idx,0:3]
    val_clindata = train_clindata_all.iloc[val_idx,0:8]
    clindata = {'train_val':train_clindata_all ,'train': train_clindata ,'val': val_clindata, 'test':test_clindata, 'covariates': covariates,'time_train': train_clindata_all['time'], 'event_train': train_clindata_all['event'], 'slide_id_test': test_clindata_all['ID'], 'cutoff': cutoff, 'cens': cens_perc, 'cens_train': cens_perc_train }
    return clindata
	
def sim_event_times_case4(trainset, num_samples):    
    train_n = int( .8 * num_samples)
    test_n = int( (.2) * num_samples) 
    cov =  np.random.standard_normal(size=(num_samples, 9))
    treatment = np.random.binomial(n=1,p=.5,size=num_samples)
    treatment=np.expand_dims(treatment,1)
    clinical_data = np.concatenate((treatment, cov), axis=1)
    index = np.arange(len(trainset.targets))
    idx_sample = np.random.choice(index, num_samples,replace=False)
    digits = np.array(trainset.targets)[idx_sample]
    shape = np.maximum(0.001,np.exp(.5*digits+.2*clinical_data[:,0] * np.cos(digits)+1.5*clinical_data[:,1]+1.2*clinical_data[:,0]))
    true_times = np.random.gamma(shape,digits, num_samples) # shape = shape; scale = digits
    censored_times = np.random.uniform(low=0,high=true_times)
    censored_indicator = np.random.binomial(n=1,p=.3,size=digits.shape[0])
    times = np.where(censored_indicator==1, censored_times,true_times) 
    event = np.where(censored_indicator==1,0,1)
    cutoff = np.array(np.quantile(true_times,(.2,.3,.4,.5,.6)))
    event_1= np.where(true_times<= cutoff[0],1,0)
    event_2= np.where(true_times<= cutoff[1],1,0)
    event_3= np.where(true_times<= cutoff[2],1,0)
    event_4= np.where(true_times<= cutoff[3],1,0)
    event_5= np.where(true_times<= cutoff[4],1,0)    
    cens_perc = np.sum(censored_indicator)/num_samples
    cens_perc_train = np.sum(censored_indicator[:train_n])/train_n
    df = np.concatenate((np.expand_dims(idx_sample,axis=1), np.expand_dims(times,axis=1),np.expand_dims(event,axis=1),
    np.expand_dims(event_1,axis=1),np.expand_dims(event_2,axis=1),np.expand_dims(event_3,axis=1),np.expand_dims(event_4,axis=1),np.expand_dims(event_5,axis=1),clinical_data),axis=1)
    df = pd.DataFrame(df,columns= ('ID','time','event','event_1','event_2','event_3','event_4','event_5','cov1','cov2','cov3','cov4','cov5','cov6','cov7','cov8','cov9','cov10')) # the ID is the image chosen
    train_clindata_all =  df.iloc[0:train_n,:]
    test_clindata_all = df.iloc[(train_n+1):,:]
    test_clindata = test_clindata_all[ ['ID']  + ['time'] + ['event'] +['event_1'] + ['event_2'] + ['event_3'] + ['event_4'] + ['event_5']]
    covariates = df[['ID'] +  df.columns.tolist()[8:]]
    idx_train = np.arange(len(train_clindata_all))
    random.seed(5)
    random.shuffle(idx_train)
    split = int(.8*len(idx_train))
    train_idx = idx_train[:split]
    val_idx = idx_train[split:]
    train_clindata = train_clindata_all.iloc[train_idx,0:3]
    val_clindata = train_clindata_all.iloc[val_idx,0:8]
    clindata = {'train_val':train_clindata_all ,'train': train_clindata ,'val': val_clindata, 'test':test_clindata, 'covariates': covariates,'time_train': train_clindata_all['time'], 'event_train': train_clindata_all['event'], 'slide_id_test': test_clindata_all['ID'], 'cutoff': cutoff, 'cens': cens_perc, 'cens_train': cens_perc_train }
    return clindata
    

def sim_event_times_case5(trainset, num_samples):    
    train_n = int( .8 * num_samples)
    test_n = int( (.2) * num_samples) 
    cov =  np.random.standard_normal(size=(num_samples, 9))
    treatment = np.random.binomial(n=1,p=.5,size=num_samples)
    treatment=np.expand_dims(treatment,1)
    clinical_data = np.concatenate((treatment, cov), axis=1)
    index = np.arange(len(trainset.targets))
    idx_sample = np.random.choice(index, num_samples,replace=False)
    digits = np.array(trainset.targets)[idx_sample]
    shape = np.maximum(0.001,np.exp(.5*digits+.2*clinical_data[:,0] * np.cos(digits)+1.5*clinical_data[:,1]+1.2*clinical_data[:,0]))	
    true_times = np.random.gamma(shape,digits, num_samples) # shape = shape; scale = digits  
    denom = np.exp( -3.4*clinical_data[:,0]+.6*clinical_data[:,1] -2.2*clinical_data[:,2]   ) * .005
    censored_times = np.sqrt(-np.log(np.random.uniform(low=0,high=1,size=num_samples))/denom )
    censored_indicator = (true_times > censored_times)*1
    times = np.where(censored_indicator==1, censored_times,true_times) 
    event = np.where(censored_indicator==1,0,1)
    cutoff = np.array(np.quantile(true_times,(.2,.3,.4,.5,.6)))
    event_1= np.where(true_times<= cutoff[0],1,0)
    event_2= np.where(true_times<= cutoff[1],1,0)
    event_3= np.where(true_times<= cutoff[2],1,0)
    event_4= np.where(true_times<= cutoff[3],1,0)
    event_5= np.where(true_times<= cutoff[4],1,0)    
    cens_perc = np.sum(censored_indicator)/num_samples
    cens_perc_train = np.sum(censored_indicator[:train_n])/train_n
    df = np.concatenate((np.expand_dims(idx_sample,axis=1), np.expand_dims(times,axis=1),np.expand_dims(event,axis=1),
    np.expand_dims(event_1,axis=1),np.expand_dims(event_2,axis=1),np.expand_dims(event_3,axis=1),np.expand_dims(event_4,axis=1),np.expand_dims(event_5,axis=1),clinical_data),axis=1)
    df = pd.DataFrame(df,columns= ('ID','time','event','event_1','event_2','event_3','event_4','event_5','cov1','cov2','cov3','cov4','cov5','cov6','cov7','cov8','cov9','cov10')) # the ID is the image chosen
    train_clindata_all =  df.iloc[0:train_n,:]
    test_clindata_all = df.iloc[(train_n+1):,:]
    test_clindata = test_clindata_all[ ['ID']  + ['time'] + ['event'] +['event_1'] + ['event_2'] + ['event_3'] + ['event_4'] + ['event_5']]
    covariates = df[['ID'] +  df.columns.tolist()[8:]]
    idx_train = np.arange(len(train_clindata_all))
    random.seed(5)
    random.shuffle(idx_train)
    split = int(.8*len(idx_train))
    train_idx = idx_train[:split]
    val_idx = idx_train[split:]
    train_clindata = train_clindata_all.iloc[train_idx,0:3]
    val_clindata = train_clindata_all.iloc[val_idx,0:8]
    clindata = {'train_val':train_clindata_all ,'train': train_clindata ,'val': val_clindata, 'test':test_clindata, 'covariates': covariates,'time_train': train_clindata_all['time'], 'event_train': train_clindata_all['event'], 'slide_id_test': test_clindata_all['ID'], 'cutoff': cutoff, 'cens': cens_perc, 'cens_train': cens_perc_train }
    return clindata

	
	
def sim_event_times_case6(trainset, num_samples):    
    train_n = int( .8 * num_samples)
    test_n = int( (.2) * num_samples) 
    cov =  np.random.standard_normal(size=(num_samples, 9))
    treatment = np.random.binomial(n=1,p=.5,size=num_samples)
    treatment=np.expand_dims(treatment,1)
    clinical_data = np.concatenate((treatment, cov), axis=1)
    index = np.arange(len(trainset.targets))
    idx_sample = np.random.choice(index, num_samples,replace=False)
    digits = np.array(trainset.targets)[idx_sample]
    shape = np.maximum(0.001,np.exp(.7*digits+ .4*clinical_data[:,0] * digits-.1*clinical_data[:,1]*clinical_data[:,0]+ .1*digits *clinical_data[:,1]))	
    true_times = np.random.gamma(shape,digits, num_samples) # shape = shape; scale = digits  
    shape_c = np.maximum(0.001,np.exp(3.8*clinical_data[:,0]+5.2*clinical_data[:,1]-3.3 *clinical_data[:,0]*clinical_data[:,1]))
    censored_times = np.random.gamma(shape_c,digits, num_samples)
    censored_indicator = (true_times > censored_times)*1
    times = np.where(censored_indicator==1, censored_times,true_times) 
    event = np.where(censored_indicator==1,0,1)
    cutoff = np.array(np.quantile(true_times,(.2,.3,.4,.5,.6)))
    event_1= np.where(true_times<= cutoff[0],1,0)
    event_2= np.where(true_times<= cutoff[1],1,0)
    event_3= np.where(true_times<= cutoff[2],1,0)
    event_4= np.where(true_times<= cutoff[3],1,0)
    event_5= np.where(true_times<= cutoff[4],1,0)    
    cens_perc = np.sum(censored_indicator)/num_samples
    cens_perc_train = np.sum(censored_indicator[:train_n])/train_n
    df = np.concatenate((np.expand_dims(idx_sample,axis=1), np.expand_dims(times,axis=1),np.expand_dims(event,axis=1),
    np.expand_dims(event_1,axis=1),np.expand_dims(event_2,axis=1),np.expand_dims(event_3,axis=1),np.expand_dims(event_4,axis=1),np.expand_dims(event_5,axis=1),clinical_data),axis=1)
    df = pd.DataFrame(df,columns= ('ID','time','event','event_1','event_2','event_3','event_4','event_5','cov1','cov2','cov3','cov4','cov5','cov6','cov7','cov8','cov9','cov10')) # the ID is the image chosen
    train_clindata_all =  df.iloc[0:train_n,:]
    test_clindata_all = df.iloc[(train_n+1):,:]
    test_clindata = test_clindata_all[ ['ID']  + ['time'] + ['event'] +['event_1'] + ['event_2'] + ['event_3'] + ['event_4'] + ['event_5']]
    covariates = df[['ID'] +  df.columns.tolist()[8:]]
    idx_train = np.arange(len(train_clindata_all))
    random.seed(5)
    random.shuffle(idx_train)
    split = int(.8*len(idx_train))
    train_idx = idx_train[:split]
    val_idx = idx_train[split:]
    train_clindata = train_clindata_all.iloc[train_idx,0:3]
    val_clindata = train_clindata_all.iloc[val_idx,0:8]
    clindata = {'train_val':train_clindata_all ,'train': train_clindata ,'val': val_clindata, 'test':test_clindata, 'covariates': covariates,'time_train': train_clindata_all['time'], 'event_train': train_clindata_all['event'], 'slide_id_test': test_clindata_all['ID'], 'cutoff': cutoff, 'cens': cens_perc, 'cens_train': cens_perc_train }
    return clindata	

      


    