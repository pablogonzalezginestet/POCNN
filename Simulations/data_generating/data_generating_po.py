import numpy as np
import pandas as pd
import random
from rpy2.robjects.packages import importr
utils = importr('utils')
prodlim = importr('prodlim')
survival = importr('survival')
#KMsurv = importr('KMsurv')
#cvAUC = importr('pROC')
#utils.install_packages('pseudo')
#utils.install_packages('prodlim')
#utils.install_packages('survival')
#utils.install_packages('KMsurv')
#utils.install_packages('pROC')
import rpy2.robjects as robjects
from rpy2.robjects import r

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
    #split data
    train_clindata_all =  df.iloc[0:train_n,:]
    order_time = np.argsort(train_clindata_all['time'])
    train_clindata_all = train_clindata_all.iloc[order_time,:]
    test_clindata_all = df.iloc[train_n:,:]
    time_r = robjects.FloatVector(train_clindata_all['time'])
    event_r = robjects.BoolVector(train_clindata_all['event'])
    cutoff_r = robjects.FloatVector(cutoff)
    robjects.globalenv["time_r"] = time_r
    robjects.globalenv["event_r"] = event_r
    robjects.globalenv["cutoff"] = cutoff_r
    r('km_out <- prodlim(Hist(time_r,event_r)~1)')  
    r(' surv_pso <- jackknife(km_out,times=cutoff) ' )
    risk_pso1 = r('1-surv_pso[,1]')
    risk_pso2 = r('1-surv_pso[,2]')
    risk_pso3 = r('1-surv_pso[,3]')
    risk_pso4 = r('1-surv_pso[,4]')
    risk_pso5 = r('1-surv_pso[,5]')
    train_clindata_all = train_clindata_all.assign(risk_pso1 = np.array(risk_pso1,dtype=np.float64),
        risk_pso2 = np.array(risk_pso2,dtype=np.float64),
        risk_pso3 = np.array(risk_pso3,dtype=np.float64),
        risk_pso4 = np.array(risk_pso4,dtype=np.float64),
        risk_pso5 = np.array(risk_pso5,dtype=np.float64)
        )
    long_df = pd.melt(train_clindata_all, id_vars=['ID'],value_vars=['risk_pso1','risk_pso2','risk_pso3','risk_pso4','risk_pso5'] )
    long_df.rename(columns={'variable': 'time_point','value': 'ps_risk'}, inplace=True)
    mymap= {'risk_pso1': 'time1', 'risk_pso2': 'time2', 'risk_pso3': 'time3', 'risk_pso4': 'time4', 'risk_pso5': 'time5'  }
    long_df = long_df.applymap(lambda s : mymap.get(s) if s in mymap else s)
    train_val_clindata = pd.get_dummies(long_df, columns=['time_point'])
    test_clindata_all = test_clindata_all.assign( time_point1=1,time_point2=2,time_point3=3,time_point4=4,time_point5=5 )
    long_test_df = pd.melt(test_clindata_all, id_vars=['ID'],value_vars=['time_point1','time_point2','time_point3','time_point4','time_point5'] )
    long_test_df.rename(columns={'value': 'time_point'}, inplace=True)
    long_test_clindata_all = pd.merge(left=long_test_df, right=test_clindata_all, how='left',left_on='ID' ,right_on='ID')
    cols_test = long_test_clindata_all.columns.tolist()
    long_test_clindata = long_test_clindata_all[ ['ID'] + ['time_point'] + ['time'] + ['event'] + ['event_1'] + ['event_2'] + ['event_3'] + ['event_4'] + ['event_5']]
    long_test_clindata = pd.get_dummies(long_test_clindata, columns=['time_point'])
    covariates = df[['ID'] +  df.columns.tolist()[8:]]
    clindata = {'train_val':train_val_clindata , 'test':long_test_clindata, 'covariates': covariates,'time_train': train_clindata_all['time'], 'event_train': train_clindata_all['event'], 'slide_id_test': test_clindata_all['ID'], 'cutoff': cutoff , 'cens': cens_perc, 'cens_train': cens_perc_train}
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
    order_time = np.argsort(train_clindata_all['time'])
    train_clindata_all = train_clindata_all.iloc[order_time,:]
    test_clindata_all = df.iloc[train_n:,:]
    time_r = robjects.FloatVector(train_clindata_all['time'])
    event_r = robjects.BoolVector(train_clindata_all['event'])
    cutoff_r = robjects.FloatVector(cutoff)
    robjects.globalenv["time_r"] = time_r
    robjects.globalenv["event_r"] = event_r
    robjects.globalenv["cutoff"] = cutoff_r
    r('km_out <- prodlim(Hist(time_r,event_r)~1)')  
    r(' surv_pso <- jackknife(km_out,times=cutoff) ' )
    risk_pso1 = r('1-surv_pso[,1]')
    risk_pso2 = r('1-surv_pso[,2]')
    risk_pso3 = r('1-surv_pso[,3]')
    risk_pso4 = r('1-surv_pso[,4]')
    risk_pso5 = r('1-surv_pso[,5]')
    train_clindata_all = train_clindata_all.assign(risk_pso1 = np.array(risk_pso1,dtype=np.float64),
        risk_pso2 = np.array(risk_pso2,dtype=np.float64),
        risk_pso3 = np.array(risk_pso3,dtype=np.float64),
        risk_pso4 = np.array(risk_pso4,dtype=np.float64),
        risk_pso5 = np.array(risk_pso5,dtype=np.float64)
        )
    long_df = pd.melt(train_clindata_all, id_vars=['ID'],value_vars=['risk_pso1','risk_pso2','risk_pso3','risk_pso4','risk_pso5'] )
    long_df.rename(columns={'variable': 'time_point','value': 'ps_risk'}, inplace=True)
    mymap= {'risk_pso1': 'time1', 'risk_pso2': 'time2', 'risk_pso3': 'time3', 'risk_pso4': 'time4', 'risk_pso5': 'time5'  }
    long_df = long_df.applymap(lambda s : mymap.get(s) if s in mymap else s)
    train_val_clindata = pd.get_dummies(long_df, columns=['time_point'])
    test_clindata_all = test_clindata_all.assign( time_point1=1,time_point2=2,time_point3=3,time_point4=4,time_point5=5 )    
    long_test_df = pd.melt(test_clindata_all, id_vars=['ID'],value_vars=['time_point1','time_point2','time_point3','time_point4','time_point5'] )
    long_test_df.rename(columns={'value': 'time_point'}, inplace=True)
    long_test_clindata_all = pd.merge(left=long_test_df, right=test_clindata_all, how='left',left_on='ID' ,right_on='ID')
    cols_test = long_test_clindata_all.columns.tolist()
    long_test_clindata = long_test_clindata_all[ ['ID'] + ['time_point'] + ['time'] + ['event'] + ['event_1'] + ['event_2'] + ['event_3'] + ['event_4'] + ['event_5']]
    long_test_clindata = pd.get_dummies(long_test_clindata, columns=['time_point'])
    covariates = df[['ID'] +  df.columns.tolist()[8:]]
    clindata = {'train_val':train_val_clindata , 'test':long_test_clindata, 'covariates': covariates,'time_train': train_clindata_all['time'], 'event_train': train_clindata_all['event'], 'slide_id_test': test_clindata_all['ID'], 'cutoff': cutoff , 'cens': cens_perc, 'cens_train': cens_perc_train}
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
    #denom = np.exp( 1.4*clinical_data[:,0]+2.6*clinical_data[:,1] -.2*clinical_data[:,2]   )*6
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
    order_time = np.argsort(train_clindata_all['time'])
    train_clindata_all = train_clindata_all.iloc[order_time,:]
    test_clindata_all = df.iloc[train_n:,:]
    time_r = robjects.FloatVector(train_clindata_all['time'])
    event_r = robjects.BoolVector(train_clindata_all['event'])
    cutoff_r = robjects.FloatVector(cutoff)
    robjects.globalenv["time_r"] = time_r
    robjects.globalenv["event_r"] = event_r
    robjects.globalenv["cutoff"] = cutoff_r
    r('km_out <- prodlim(Hist(time_r,event_r)~1)')  
    r(' surv_pso <- jackknife(km_out,times=cutoff) ' )
    risk_pso1 = r('1-surv_pso[,1]')
    risk_pso2 = r('1-surv_pso[,2]')
    risk_pso3 = r('1-surv_pso[,3]')
    risk_pso4 = r('1-surv_pso[,4]')
    risk_pso5 = r('1-surv_pso[,5]')
    train_clindata_all = train_clindata_all.assign(risk_pso1 = np.array(risk_pso1,dtype=np.float64),
        risk_pso2 = np.array(risk_pso2,dtype=np.float64),
        risk_pso3 = np.array(risk_pso3,dtype=np.float64),
        risk_pso4 = np.array(risk_pso4,dtype=np.float64),
        risk_pso5 = np.array(risk_pso5,dtype=np.float64)
        )
    long_df = pd.melt(train_clindata_all, id_vars=['ID'],value_vars=['risk_pso1','risk_pso2','risk_pso3','risk_pso4','risk_pso5'] )
    long_df.rename(columns={'variable': 'time_point','value': 'ps_risk'}, inplace=True)
    mymap= {'risk_pso1': 'time1', 'risk_pso2': 'time2', 'risk_pso3': 'time3', 'risk_pso4': 'time4', 'risk_pso5': 'time5'  }
    long_df = long_df.applymap(lambda s : mymap.get(s) if s in mymap else s)
    train_val_clindata = pd.get_dummies(long_df, columns=['time_point'])
    test_clindata_all = test_clindata_all.assign( time_point1=1,time_point2=2,time_point3=3,time_point4=4,time_point5=5 )    
    long_test_df = pd.melt(test_clindata_all, id_vars=['ID'],value_vars=['time_point1','time_point2','time_point3','time_point4','time_point5'] )
    long_test_df.rename(columns={'value': 'time_point'}, inplace=True)
    long_test_clindata_all = pd.merge(left=long_test_df, right=test_clindata_all, how='left',left_on='ID' ,right_on='ID')
    cols_test = long_test_clindata_all.columns.tolist()
    long_test_clindata = long_test_clindata_all[ ['ID'] + ['time_point'] + ['time'] + ['event'] + ['event_1'] + ['event_2'] + ['event_3'] + ['event_4'] + ['event_5']]
    long_test_clindata = pd.get_dummies(long_test_clindata, columns=['time_point'])
    covariates = df[['ID'] +  df.columns.tolist()[8:]]
    clindata = {'train_val':train_val_clindata , 'test':long_test_clindata, 'covariates': covariates,'time_train': train_clindata_all['time'], 'event_train': train_clindata_all['event'], 'slide_id_test': test_clindata_all['ID'], 'cutoff': cutoff , 'cens': cens_perc, 'cens_train': cens_perc_train}
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
    order_time = np.argsort(train_clindata_all['time'])
    train_clindata_all = train_clindata_all.iloc[order_time,:]
    test_clindata_all = df.iloc[train_n:,:]
    time_r = robjects.FloatVector(train_clindata_all['time'])
    event_r = robjects.BoolVector(train_clindata_all['event'])
    cutoff_r = robjects.FloatVector(cutoff)
    robjects.globalenv["time_r"] = time_r
    robjects.globalenv["event_r"] = event_r
    robjects.globalenv["cutoff"] = cutoff_r
    r('km_out <- prodlim(Hist(time_r,event_r)~1)')  
    r(' surv_pso <- jackknife(km_out,times=cutoff) ' )
    risk_pso1 = r('1-surv_pso[,1]')
    risk_pso2 = r('1-surv_pso[,2]')
    risk_pso3 = r('1-surv_pso[,3]')
    risk_pso4 = r('1-surv_pso[,4]')
    risk_pso5 = r('1-surv_pso[,5]')
    train_clindata_all = train_clindata_all.assign(risk_pso1 = np.array(risk_pso1,dtype=np.float64),
        risk_pso2 = np.array(risk_pso2,dtype=np.float64),
        risk_pso3 = np.array(risk_pso3,dtype=np.float64),
        risk_pso4 = np.array(risk_pso4,dtype=np.float64),
        risk_pso5 = np.array(risk_pso5,dtype=np.float64)
        )
    long_df = pd.melt(train_clindata_all, id_vars=['ID'],value_vars=['risk_pso1','risk_pso2','risk_pso3','risk_pso4','risk_pso5'] )
    long_df.rename(columns={'variable': 'time_point','value': 'ps_risk'}, inplace=True)
    mymap= {'risk_pso1': 'time1', 'risk_pso2': 'time2', 'risk_pso3': 'time3', 'risk_pso4': 'time4', 'risk_pso5': 'time5'  }
    long_df = long_df.applymap(lambda s : mymap.get(s) if s in mymap else s)
    train_val_clindata = pd.get_dummies(long_df, columns=['time_point'])
    test_clindata_all = test_clindata_all.assign( time_point1=1,time_point2=2,time_point3=3,time_point4=4,time_point5=5 )    
    long_test_df = pd.melt(test_clindata_all, id_vars=['ID'],value_vars=['time_point1','time_point2','time_point3','time_point4','time_point5'] )
    long_test_df.rename(columns={'value': 'time_point'}, inplace=True)
    long_test_clindata_all = pd.merge(left=long_test_df, right=test_clindata_all, how='left',left_on='ID' ,right_on='ID')
    cols_test = long_test_clindata_all.columns.tolist()
    long_test_clindata = long_test_clindata_all[ ['ID'] + ['time_point'] + ['time'] + ['event'] + ['event_1'] + ['event_2'] + ['event_3'] + ['event_4'] + ['event_5']]
    long_test_clindata = pd.get_dummies(long_test_clindata, columns=['time_point'])
    covariates = df[['ID'] +  df.columns.tolist()[8:]]
    clindata = {'train_val':train_val_clindata , 'test':long_test_clindata, 'covariates': covariates,'time_train': train_clindata_all['time'], 'event_train': train_clindata_all['event'], 'slide_id_test': test_clindata_all['ID'], 'cutoff': cutoff , 'cens': cens_perc, 'cens_train': cens_perc_train}
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
    order_time = np.argsort(train_clindata_all['time'])
    train_clindata_all = train_clindata_all.iloc[order_time,:]
    test_clindata_all = df.iloc[train_n:,:]
    time_r = robjects.FloatVector(train_clindata_all['time'])
    event_r = robjects.BoolVector(train_clindata_all['event'])
    cutoff_r = robjects.FloatVector(cutoff)
    robjects.globalenv["time_r"] = time_r
    robjects.globalenv["event_r"] = event_r
    robjects.globalenv["cutoff"] = cutoff_r
    r('km_out <- prodlim(Hist(time_r,event_r)~1)')  
    r(' surv_pso <- jackknife(km_out,times=cutoff) ' )
    risk_pso1 = r('1-surv_pso[,1]')
    risk_pso2 = r('1-surv_pso[,2]')
    risk_pso3 = r('1-surv_pso[,3]')
    risk_pso4 = r('1-surv_pso[,4]')
    risk_pso5 = r('1-surv_pso[,5]')
    train_clindata_all = train_clindata_all.assign(risk_pso1 = np.array(risk_pso1,dtype=np.float64),
        risk_pso2 = np.array(risk_pso2,dtype=np.float64),
        risk_pso3 = np.array(risk_pso3,dtype=np.float64),
        risk_pso4 = np.array(risk_pso4,dtype=np.float64),
        risk_pso5 = np.array(risk_pso5,dtype=np.float64)
        )
    long_df = pd.melt(train_clindata_all, id_vars=['ID'],value_vars=['risk_pso1','risk_pso2','risk_pso3','risk_pso4','risk_pso5'] )
    long_df.rename(columns={'variable': 'time_point','value': 'ps_risk'}, inplace=True)
    mymap= {'risk_pso1': 'time1', 'risk_pso2': 'time2', 'risk_pso3': 'time3', 'risk_pso4': 'time4', 'risk_pso5': 'time5'  }
    long_df = long_df.applymap(lambda s : mymap.get(s) if s in mymap else s)
    train_val_clindata = pd.get_dummies(long_df, columns=['time_point'])
    test_clindata_all = test_clindata_all.assign( time_point1=1,time_point2=2,time_point3=3,time_point4=4,time_point5=5 )    
    long_test_df = pd.melt(test_clindata_all, id_vars=['ID'],value_vars=['time_point1','time_point2','time_point3','time_point4','time_point5'] )
    long_test_df.rename(columns={'value': 'time_point'}, inplace=True)
    long_test_clindata_all = pd.merge(left=long_test_df, right=test_clindata_all, how='left',left_on='ID' ,right_on='ID')
    cols_test = long_test_clindata_all.columns.tolist()
    long_test_clindata = long_test_clindata_all[ ['ID'] + ['time_point'] + ['time'] + ['event'] + ['event_1'] + ['event_2'] + ['event_3'] + ['event_4'] + ['event_5']]
    long_test_clindata = pd.get_dummies(long_test_clindata, columns=['time_point'])
    covariates = df[['ID'] +  df.columns.tolist()[8:]]
    clindata = {'train_val':train_val_clindata , 'test':long_test_clindata, 'covariates': covariates,'time_train': train_clindata_all['time'], 'event_train': train_clindata_all['event'], 'slide_id_test': test_clindata_all['ID'], 'cutoff': cutoff , 'cens': cens_perc, 'cens_train': cens_perc_train}
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
    order_time = np.argsort(train_clindata_all['time'])
    train_clindata_all = train_clindata_all.iloc[order_time,:]
    test_clindata_all = df.iloc[train_n:,:]
    time_r = robjects.FloatVector(train_clindata_all['time'])
    event_r = robjects.BoolVector(train_clindata_all['event'])
    cutoff_r = robjects.FloatVector(cutoff)
    robjects.globalenv["time_r"] = time_r
    robjects.globalenv["event_r"] = event_r
    robjects.globalenv["cutoff"] = cutoff_r
    r('km_out <- prodlim(Hist(time_r,event_r)~1)')  
    r(' surv_pso <- jackknife(km_out,times=cutoff) ' )
    risk_pso1 = r('1-surv_pso[,1]')
    risk_pso2 = r('1-surv_pso[,2]')
    risk_pso3 = r('1-surv_pso[,3]')
    risk_pso4 = r('1-surv_pso[,4]')
    risk_pso5 = r('1-surv_pso[,5]')
    train_clindata_all = train_clindata_all.assign(risk_pso1 = np.array(risk_pso1,dtype=np.float64),
        risk_pso2 = np.array(risk_pso2,dtype=np.float64),
        risk_pso3 = np.array(risk_pso3,dtype=np.float64),
        risk_pso4 = np.array(risk_pso4,dtype=np.float64),
        risk_pso5 = np.array(risk_pso5,dtype=np.float64)
        )
    long_df = pd.melt(train_clindata_all, id_vars=['ID'],value_vars=['risk_pso1','risk_pso2','risk_pso3','risk_pso4','risk_pso5'] )
    long_df.rename(columns={'variable': 'time_point','value': 'ps_risk'}, inplace=True)
    mymap= {'risk_pso1': 'time1', 'risk_pso2': 'time2', 'risk_pso3': 'time3', 'risk_pso4': 'time4', 'risk_pso5': 'time5'  }
    long_df = long_df.applymap(lambda s : mymap.get(s) if s in mymap else s)
    train_val_clindata = pd.get_dummies(long_df, columns=['time_point'])
    test_clindata_all = test_clindata_all.assign( time_point1=1,time_point2=2,time_point3=3,time_point4=4,time_point5=5 )	    
    long_test_df = pd.melt(test_clindata_all, id_vars=['ID'],value_vars=['time_point1','time_point2','time_point3','time_point4','time_point5'] )
    long_test_df.rename(columns={'value': 'time_point'}, inplace=True)
    long_test_clindata_all = pd.merge(left=long_test_df, right=test_clindata_all, how='left',left_on='ID' ,right_on='ID')
    cols_test = long_test_clindata_all.columns.tolist()
    long_test_clindata = long_test_clindata_all[ ['ID'] + ['time_point'] + ['time'] + ['event'] + ['event_1'] + ['event_2'] + ['event_3'] + ['event_4'] + ['event_5']]
    long_test_clindata = pd.get_dummies(long_test_clindata, columns=['time_point'])
    covariates = df[['ID'] +  df.columns.tolist()[8:]]
    clindata = {'train_val':train_val_clindata , 'test':long_test_clindata, 'covariates': covariates,'time_train': train_clindata_all['time'], 'event_train': train_clindata_all['event'], 'slide_id_test': test_clindata_all['ID'], 'cutoff': cutoff , 'cens': cens_perc, 'cens_train': cens_perc_train}
    return clindata