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
#utils.install_packages('eventglm')
eventglm = importr('eventglm', lib_loc="C:/Users/pabgon/Documents/R/R-4.0.4/library")
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
    #event_5= np.where(true_times<= cutoff[4],1,0)    
    cens_perc = np.sum(censored_indicator)/num_samples
    cens_perc_train = np.sum(censored_indicator[:train_n])/train_n
    #df = np.concatenate((np.expand_dims(idx_sample,axis=1), np.expand_dims(times,axis=1),np.expand_dims(event,axis=1),
    #np.expand_dims(event_1,axis=1),np.expand_dims(event_2,axis=1),np.expand_dims(event_3,axis=1),np.expand_dims(event_4,axis=1),np.expand_dims(event_5,axis=1),clinical_data),axis=1)
    df = np.concatenate((np.expand_dims(idx_sample,axis=1), np.expand_dims(times,axis=1),np.expand_dims(event,axis=1),
    np.expand_dims(event_1,axis=1),np.expand_dims(event_2,axis=1),np.expand_dims(event_3,axis=1),np.expand_dims(event_4,axis=1),clinical_data),axis=1)
    #df = pd.DataFrame(df,columns= ('ID','time','event','event_1','event_2','event_3','event_4','event_5','cov1','cov2','cov3','cov4','cov5','cov6','cov7','cov8','cov9','cov10')) # the ID is the image chosen
    df = pd.DataFrame(df,columns= ('ID','time','event','event_1','event_2','event_3','event_4','cov1','cov2','cov3','cov4','cov5','cov6','cov7','cov8','cov9','cov10')) # the ID is the image chosen
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
    # IPCW Pseudo Observations 
    for i in (np.arange(10)+7):
        exec(f'cov_{i} = robjects.FloatVector(train_clindata_all.iloc[:,i]) ')
        exec(f'robjects.globalenv["cov_{i}"] = cov_{i}')
    r('xnam <- paste("cov_", 0:9, sep="")')
    r('data_surv_pso <- data.frame("time_r"=time_r,"event_r"=event_r , "cov_0"=cov_7, "cov_1"=cov_8, "cov_2"=cov_9, "cov_3"=cov_10,"cov_4"=cov_11,"cov_5"=cov_12,"cov_6"=cov_13, "cov_7"=cov_14, "cov_8"=cov_15,"cov_9"=cov_16) ' )
    r('POi_1 <- eventglm::pseudo_coxph(Surv(time_r, event_r) ~ 1, cutoff[1], cause = 1,data = data_surv_pso, type = "cuminc", formula.censoring = ~ cov_0+cov_1+cov_2+cov_3+cov_4+cov_5+cov_6+cov_7+cov_8+cov_9 , ipcw.method = "hajek" )' )
    r('POi_2 <- eventglm::pseudo_coxph(Surv(time_r, event_r) ~ 1, cutoff[2], cause = 1,data = data_surv_pso, type = "cuminc", formula.censoring = ~ cov_0+cov_1+cov_2+cov_3+cov_4+cov_5+cov_6+cov_7+cov_8+cov_9 , ipcw.method = "hajek" )' )
    r('POi_3 <- eventglm::pseudo_coxph(Surv(time_r, event_r) ~ 1, cutoff[3], cause = 1,data = data_surv_pso, type = "cuminc", formula.censoring = ~ cov_0+cov_1+cov_2+cov_3+cov_4+cov_5+cov_6+cov_7+cov_8+cov_9 , ipcw.method = "hajek" )' )
    r('POi_4 <- eventglm::pseudo_coxph(Surv(time_r, event_r) ~ 1, cutoff[4], cause = 1,data = data_surv_pso, type = "cuminc", formula.censoring = ~ cov_0+cov_1+cov_2+cov_3+cov_4+cov_5+cov_6+cov_7+cov_8+cov_9 , ipcw.method = "hajek" )' )
    #r('POi_5 <- eventglm::pseudo_coxph(Surv(time_r, event_r) ~ 1, cutoff[5], cause = 1,data = data_surv_pso, type = "cuminc", formula.censoring = ~ cov_0+cov_1+cov_2+cov_3+cov_4+cov_5+cov_6+cov_7+cov_8+cov_9 , ipcw.method = "hajek" )' )
    POi_1 = r('matrix(POi_1)')
    POi_2 = r('matrix(POi_2)')
    POi_3 = r('matrix(POi_3)')
    POi_4 = r('matrix(POi_4)')
    #POi_5 = r('matrix(POi_5)')
    ###################################################
    train_clindata_all = train_clindata_all.assign(risk_pso1 = np.array(POi_1,dtype=np.float64) ,
        risk_pso2 = np.array(POi_2,dtype=np.float64),
        risk_pso3 = np.array(POi_3,dtype=np.float64),
        risk_pso4 = np.array(POi_4,dtype=np.float64)       
        )
    #long_df = pd.melt(train_clindata_all, id_vars=['ID'],value_vars=['risk_pso1','risk_pso2','risk_pso3','risk_pso4','risk_pso5'] )
    long_df = pd.melt(train_clindata_all, id_vars=['ID'],value_vars=['risk_pso1','risk_pso2','risk_pso3','risk_pso4'] )   
    long_df.rename(columns={'variable': 'time_point','value': 'ps_risk'}, inplace=True)
    #mymap= {'risk_pso1': 'time1', 'risk_pso2': 'time2', 'risk_pso3': 'time3', 'risk_pso4': 'time4', 'risk_pso5': 'time5'  }
    mymap= {'risk_pso1': 'time1', 'risk_pso2': 'time2', 'risk_pso3': 'time3', 'risk_pso4': 'time4'  }
    long_df = long_df.applymap(lambda s : mymap.get(s) if s in mymap else s)
    train_val_clindata = pd.get_dummies(long_df, columns=['time_point'])
    #test_clindata_all = test_clindata_all.assign( time_point1=1,time_point2=2,time_point3=3,time_point4=4,time_point5=5 )
    test_clindata_all = test_clindata_all.assign( time_point1=1,time_point2=2,time_point3=3,time_point4=4 )
    #long_test_df = pd.melt(test_clindata_all, id_vars=['ID'],value_vars=['time_point1','time_point2','time_point3','time_point4','time_point5'] )
    long_test_df = pd.melt(test_clindata_all, id_vars=['ID'],value_vars=['time_point1','time_point2','time_point3','time_point4'] )
    long_test_df.rename(columns={'value': 'time_point'}, inplace=True)
    long_test_clindata_all = pd.merge(left=long_test_df, right=test_clindata_all, how='left',left_on='ID' ,right_on='ID')
    cols_test = long_test_clindata_all.columns.tolist()
    #long_test_clindata = long_test_clindata_all[ ['ID'] + ['time_point'] + ['time'] + ['event'] + ['event_1'] + ['event_2'] + ['event_3'] + ['event_4'] + ['event_5']]
    long_test_clindata = long_test_clindata_all[ ['ID'] + ['time_point'] + ['time'] + ['event'] + ['event_1'] + ['event_2'] + ['event_3'] + ['event_4'] ]
    long_test_clindata = pd.get_dummies(long_test_clindata, columns=['time_point'])
    covariates = df[['ID'] +  df.columns.tolist()[7:]]
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
    #event_5= np.where(true_times<= cutoff[4],1,0)    
    cens_perc = np.sum(censored_indicator)/num_samples
    cens_perc_train = np.sum(censored_indicator[:train_n])/train_n
    #df = np.concatenate((np.expand_dims(idx_sample,axis=1), np.expand_dims(times,axis=1),np.expand_dims(event,axis=1),
    #np.expand_dims(event_1,axis=1),np.expand_dims(event_2,axis=1),np.expand_dims(event_3,axis=1),np.expand_dims(event_4,axis=1),np.expand_dims(event_5,axis=1),clinical_data),axis=1)
    df = np.concatenate((np.expand_dims(idx_sample,axis=1), np.expand_dims(times,axis=1),np.expand_dims(event,axis=1),
    np.expand_dims(event_1,axis=1),np.expand_dims(event_2,axis=1),np.expand_dims(event_3,axis=1),np.expand_dims(event_4,axis=1),clinical_data),axis=1)
    #df = pd.DataFrame(df,columns= ('ID','time','event','event_1','event_2','event_3','event_4','event_5','cov1','cov2','cov3','cov4','cov5','cov6','cov7','cov8','cov9','cov10')) # the ID is the image chosen
    df = pd.DataFrame(df,columns= ('ID','time','event','event_1','event_2','event_3','event_4','cov1','cov2','cov3','cov4','cov5','cov6','cov7','cov8','cov9','cov10')) # the ID is the image chosen
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
    # IPCW Pseudo Observations 
    for i in (np.arange(10)+7):
        exec(f'cov_{i} = robjects.FloatVector(train_clindata_all.iloc[:,i]) ')
        exec(f'robjects.globalenv["cov_{i}"] = cov_{i}')
    r('xnam <- paste("cov_", 0:9, sep="")')
    r('data_surv_pso <- data.frame("time_r"=time_r,"event_r"=event_r , "cov_0"=cov_7, "cov_1"=cov_8, "cov_2"=cov_9, "cov_3"=cov_10,"cov_4"=cov_11,"cov_5"=cov_12,"cov_6"=cov_13, "cov_7"=cov_14, "cov_8"=cov_15,"cov_9"=cov_16) ' )
    r('POi_1 <- eventglm::pseudo_coxph(Surv(time_r, event_r) ~ 1, cutoff[1], cause = 1,data = data_surv_pso, type = "cuminc", formula.censoring = ~ cov_0+cov_1+cov_2+cov_3+cov_4+cov_5+cov_6+cov_7+cov_8+cov_9 , ipcw.method = "hajek" )' )
    r('POi_2 <- eventglm::pseudo_coxph(Surv(time_r, event_r) ~ 1, cutoff[2], cause = 1,data = data_surv_pso, type = "cuminc", formula.censoring = ~ cov_0+cov_1+cov_2+cov_3+cov_4+cov_5+cov_6+cov_7+cov_8+cov_9 , ipcw.method = "hajek" )' )
    r('POi_3 <- eventglm::pseudo_coxph(Surv(time_r, event_r) ~ 1, cutoff[3], cause = 1,data = data_surv_pso, type = "cuminc", formula.censoring = ~ cov_0+cov_1+cov_2+cov_3+cov_4+cov_5+cov_6+cov_7+cov_8+cov_9 , ipcw.method = "hajek" )' )
    r('POi_4 <- eventglm::pseudo_coxph(Surv(time_r, event_r) ~ 1, cutoff[4], cause = 1,data = data_surv_pso, type = "cuminc", formula.censoring = ~ cov_0+cov_1+cov_2+cov_3+cov_4+cov_5+cov_6+cov_7+cov_8+cov_9 , ipcw.method = "hajek" )' )
    #r('POi_5 <- eventglm::pseudo_coxph(Surv(time_r, event_r) ~ 1, cutoff[5], cause = 1,data = data_surv_pso, type = "cuminc", formula.censoring = ~ cov_0+cov_1+cov_2+cov_3+cov_4+cov_5+cov_6+cov_7+cov_8+cov_9 , ipcw.method = "hajek" )' )
    POi_1 = r('matrix(POi_1)')
    POi_2 = r('matrix(POi_2)')
    POi_3 = r('matrix(POi_3)')
    POi_4 = r('matrix(POi_4)')
    #POi_5 = r('matrix(POi_5)')
    ###################################################
    train_clindata_all = train_clindata_all.assign(risk_pso1 = np.array(POi_1,dtype=np.float64) ,
        risk_pso2 = np.array(POi_2,dtype=np.float64),
        risk_pso3 = np.array(POi_3,dtype=np.float64),
        risk_pso4 = np.array(POi_4,dtype=np.float64)       
        )
    #long_df = pd.melt(train_clindata_all, id_vars=['ID'],value_vars=['risk_pso1','risk_pso2','risk_pso3','risk_pso4','risk_pso5'] )
    long_df = pd.melt(train_clindata_all, id_vars=['ID'],value_vars=['risk_pso1','risk_pso2','risk_pso3','risk_pso4'] )   
    long_df.rename(columns={'variable': 'time_point','value': 'ps_risk'}, inplace=True)
    #mymap= {'risk_pso1': 'time1', 'risk_pso2': 'time2', 'risk_pso3': 'time3', 'risk_pso4': 'time4', 'risk_pso5': 'time5'  }
    mymap= {'risk_pso1': 'time1', 'risk_pso2': 'time2', 'risk_pso3': 'time3', 'risk_pso4': 'time4'  }
    long_df = long_df.applymap(lambda s : mymap.get(s) if s in mymap else s)
    train_val_clindata = pd.get_dummies(long_df, columns=['time_point'])
    #test_clindata_all = test_clindata_all.assign( time_point1=1,time_point2=2,time_point3=3,time_point4=4,time_point5=5 )
    test_clindata_all = test_clindata_all.assign( time_point1=1,time_point2=2,time_point3=3,time_point4=4 )
    #long_test_df = pd.melt(test_clindata_all, id_vars=['ID'],value_vars=['time_point1','time_point2','time_point3','time_point4','time_point5'] )
    long_test_df = pd.melt(test_clindata_all, id_vars=['ID'],value_vars=['time_point1','time_point2','time_point3','time_point4'] )
    long_test_df.rename(columns={'value': 'time_point'}, inplace=True)
    long_test_clindata_all = pd.merge(left=long_test_df, right=test_clindata_all, how='left',left_on='ID' ,right_on='ID')
    cols_test = long_test_clindata_all.columns.tolist()
    #long_test_clindata = long_test_clindata_all[ ['ID'] + ['time_point'] + ['time'] + ['event'] + ['event_1'] + ['event_2'] + ['event_3'] + ['event_4'] + ['event_5']]
    long_test_clindata = long_test_clindata_all[ ['ID'] + ['time_point'] + ['time'] + ['event'] + ['event_1'] + ['event_2'] + ['event_3'] + ['event_4'] ]
    long_test_clindata = pd.get_dummies(long_test_clindata, columns=['time_point'])
    covariates = df[['ID'] +  df.columns.tolist()[7:]]
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
    #event_5= np.where(true_times<= cutoff[4],1,0)    
    cens_perc = np.sum(censored_indicator)/num_samples
    cens_perc_train = np.sum(censored_indicator[:train_n])/train_n
    #df = np.concatenate((np.expand_dims(idx_sample,axis=1), np.expand_dims(times,axis=1),np.expand_dims(event,axis=1),
    #np.expand_dims(event_1,axis=1),np.expand_dims(event_2,axis=1),np.expand_dims(event_3,axis=1),np.expand_dims(event_4,axis=1),np.expand_dims(event_5,axis=1),clinical_data),axis=1)
    df = np.concatenate((np.expand_dims(idx_sample,axis=1), np.expand_dims(times,axis=1),np.expand_dims(event,axis=1),
    np.expand_dims(event_1,axis=1),np.expand_dims(event_2,axis=1),np.expand_dims(event_3,axis=1),np.expand_dims(event_4,axis=1),clinical_data),axis=1)
    #df = pd.DataFrame(df,columns= ('ID','time','event','event_1','event_2','event_3','event_4','event_5','cov1','cov2','cov3','cov4','cov5','cov6','cov7','cov8','cov9','cov10')) # the ID is the image chosen
    df = pd.DataFrame(df,columns= ('ID','time','event','event_1','event_2','event_3','event_4','cov1','cov2','cov3','cov4','cov5','cov6','cov7','cov8','cov9','cov10')) # the ID is the image chosen
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
    # IPCW Pseudo Observations 
    for i in (np.arange(10)+7):
        exec(f'cov_{i} = robjects.FloatVector(train_clindata_all.iloc[:,i]) ')
        exec(f'robjects.globalenv["cov_{i}"] = cov_{i}')
    r('xnam <- paste("cov_", 0:9, sep="")')
    r('data_surv_pso <- data.frame("time_r"=time_r,"event_r"=event_r , "cov_0"=cov_7, "cov_1"=cov_8, "cov_2"=cov_9, "cov_3"=cov_10,"cov_4"=cov_11,"cov_5"=cov_12,"cov_6"=cov_13, "cov_7"=cov_14, "cov_8"=cov_15,"cov_9"=cov_16) ' )
    r('POi_1 <- eventglm::pseudo_coxph(Surv(time_r, event_r) ~ 1, cutoff[1], cause = 1,data = data_surv_pso, type = "cuminc", formula.censoring = ~ cov_0+cov_1+cov_2+cov_3+cov_4+cov_5+cov_6+cov_7+cov_8+cov_9 , ipcw.method = "hajek" )' )
    r('POi_2 <- eventglm::pseudo_coxph(Surv(time_r, event_r) ~ 1, cutoff[2], cause = 1,data = data_surv_pso, type = "cuminc", formula.censoring = ~ cov_0+cov_1+cov_2+cov_3+cov_4+cov_5+cov_6+cov_7+cov_8+cov_9 , ipcw.method = "hajek" )' )
    r('POi_3 <- eventglm::pseudo_coxph(Surv(time_r, event_r) ~ 1, cutoff[3], cause = 1,data = data_surv_pso, type = "cuminc", formula.censoring = ~ cov_0+cov_1+cov_2+cov_3+cov_4+cov_5+cov_6+cov_7+cov_8+cov_9 , ipcw.method = "hajek" )' )
    r('POi_4 <- eventglm::pseudo_coxph(Surv(time_r, event_r) ~ 1, cutoff[4], cause = 1,data = data_surv_pso, type = "cuminc", formula.censoring = ~ cov_0+cov_1+cov_2+cov_3+cov_4+cov_5+cov_6+cov_7+cov_8+cov_9 , ipcw.method = "hajek" )' )
    #r('POi_5 <- eventglm::pseudo_coxph(Surv(time_r, event_r) ~ 1, cutoff[5], cause = 1,data = data_surv_pso, type = "cuminc", formula.censoring = ~ cov_0+cov_1+cov_2+cov_3+cov_4+cov_5+cov_6+cov_7+cov_8+cov_9 , ipcw.method = "hajek" )' )
    POi_1 = r('matrix(POi_1)')
    POi_2 = r('matrix(POi_2)')
    POi_3 = r('matrix(POi_3)')
    POi_4 = r('matrix(POi_4)')
    #POi_5 = r('matrix(POi_5)')
    ###################################################
    train_clindata_all = train_clindata_all.assign(risk_pso1 = np.array(POi_1,dtype=np.float64) ,
        risk_pso2 = np.array(POi_2,dtype=np.float64),
        risk_pso3 = np.array(POi_3,dtype=np.float64),
        risk_pso4 = np.array(POi_4,dtype=np.float64)       
        )
    #long_df = pd.melt(train_clindata_all, id_vars=['ID'],value_vars=['risk_pso1','risk_pso2','risk_pso3','risk_pso4','risk_pso5'] )
    long_df = pd.melt(train_clindata_all, id_vars=['ID'],value_vars=['risk_pso1','risk_pso2','risk_pso3','risk_pso4'] )   
    long_df.rename(columns={'variable': 'time_point','value': 'ps_risk'}, inplace=True)
    #mymap= {'risk_pso1': 'time1', 'risk_pso2': 'time2', 'risk_pso3': 'time3', 'risk_pso4': 'time4', 'risk_pso5': 'time5'  }
    mymap= {'risk_pso1': 'time1', 'risk_pso2': 'time2', 'risk_pso3': 'time3', 'risk_pso4': 'time4'  }
    long_df = long_df.applymap(lambda s : mymap.get(s) if s in mymap else s)
    train_val_clindata = pd.get_dummies(long_df, columns=['time_point'])
    #test_clindata_all = test_clindata_all.assign( time_point1=1,time_point2=2,time_point3=3,time_point4=4,time_point5=5 )
    test_clindata_all = test_clindata_all.assign( time_point1=1,time_point2=2,time_point3=3,time_point4=4 )
    #long_test_df = pd.melt(test_clindata_all, id_vars=['ID'],value_vars=['time_point1','time_point2','time_point3','time_point4','time_point5'] )
    long_test_df = pd.melt(test_clindata_all, id_vars=['ID'],value_vars=['time_point1','time_point2','time_point3','time_point4'] )
    long_test_df.rename(columns={'value': 'time_point'}, inplace=True)
    long_test_clindata_all = pd.merge(left=long_test_df, right=test_clindata_all, how='left',left_on='ID' ,right_on='ID')
    cols_test = long_test_clindata_all.columns.tolist()
    #long_test_clindata = long_test_clindata_all[ ['ID'] + ['time_point'] + ['time'] + ['event'] + ['event_1'] + ['event_2'] + ['event_3'] + ['event_4'] + ['event_5']]
    long_test_clindata = long_test_clindata_all[ ['ID'] + ['time_point'] + ['time'] + ['event'] + ['event_1'] + ['event_2'] + ['event_3'] + ['event_4'] ]
    long_test_clindata = pd.get_dummies(long_test_clindata, columns=['time_point'])
    covariates = df[['ID'] +  df.columns.tolist()[7:]]
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
    #event_5= np.where(true_times<= cutoff[4],1,0)    
    cens_perc = np.sum(censored_indicator)/num_samples
    cens_perc_train = np.sum(censored_indicator[:train_n])/train_n
    #df = np.concatenate((np.expand_dims(idx_sample,axis=1), np.expand_dims(times,axis=1),np.expand_dims(event,axis=1),
    #np.expand_dims(event_1,axis=1),np.expand_dims(event_2,axis=1),np.expand_dims(event_3,axis=1),np.expand_dims(event_4,axis=1),np.expand_dims(event_5,axis=1),clinical_data),axis=1)
    df = np.concatenate((np.expand_dims(idx_sample,axis=1), np.expand_dims(times,axis=1),np.expand_dims(event,axis=1),
    np.expand_dims(event_1,axis=1),np.expand_dims(event_2,axis=1),np.expand_dims(event_3,axis=1),np.expand_dims(event_4,axis=1),clinical_data),axis=1)
    #df = pd.DataFrame(df,columns= ('ID','time','event','event_1','event_2','event_3','event_4','event_5','cov1','cov2','cov3','cov4','cov5','cov6','cov7','cov8','cov9','cov10')) # the ID is the image chosen
    df = pd.DataFrame(df,columns= ('ID','time','event','event_1','event_2','event_3','event_4','cov1','cov2','cov3','cov4','cov5','cov6','cov7','cov8','cov9','cov10')) # the ID is the image chosen
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
    # IPCW Pseudo Observations 
    for i in (np.arange(10)+7):
        exec(f'cov_{i} = robjects.FloatVector(train_clindata_all.iloc[:,i]) ')
        exec(f'robjects.globalenv["cov_{i}"] = cov_{i}')
    r('xnam <- paste("cov_", 0:9, sep="")')
    r('data_surv_pso <- data.frame("time_r"=time_r,"event_r"=event_r , "cov_0"=cov_7, "cov_1"=cov_8, "cov_2"=cov_9, "cov_3"=cov_10,"cov_4"=cov_11,"cov_5"=cov_12,"cov_6"=cov_13, "cov_7"=cov_14, "cov_8"=cov_15,"cov_9"=cov_16) ' )
    r('POi_1 <- eventglm::pseudo_coxph(Surv(time_r, event_r) ~ 1, cutoff[1], cause = 1,data = data_surv_pso, type = "cuminc", formula.censoring = ~ cov_0+cov_1+cov_2+cov_3+cov_4+cov_5+cov_6+cov_7+cov_8+cov_9 , ipcw.method = "hajek" )' )
    r('POi_2 <- eventglm::pseudo_coxph(Surv(time_r, event_r) ~ 1, cutoff[2], cause = 1,data = data_surv_pso, type = "cuminc", formula.censoring = ~ cov_0+cov_1+cov_2+cov_3+cov_4+cov_5+cov_6+cov_7+cov_8+cov_9 , ipcw.method = "hajek" )' )
    r('POi_3 <- eventglm::pseudo_coxph(Surv(time_r, event_r) ~ 1, cutoff[3], cause = 1,data = data_surv_pso, type = "cuminc", formula.censoring = ~ cov_0+cov_1+cov_2+cov_3+cov_4+cov_5+cov_6+cov_7+cov_8+cov_9 , ipcw.method = "hajek" )' )
    r('POi_4 <- eventglm::pseudo_coxph(Surv(time_r, event_r) ~ 1, cutoff[4], cause = 1,data = data_surv_pso, type = "cuminc", formula.censoring = ~ cov_0+cov_1+cov_2+cov_3+cov_4+cov_5+cov_6+cov_7+cov_8+cov_9 , ipcw.method = "hajek" )' )
    #r('POi_5 <- eventglm::pseudo_coxph(Surv(time_r, event_r) ~ 1, cutoff[5], cause = 1,data = data_surv_pso, type = "cuminc", formula.censoring = ~ cov_0+cov_1+cov_2+cov_3+cov_4+cov_5+cov_6+cov_7+cov_8+cov_9 , ipcw.method = "hajek" )' )
    POi_1 = r('matrix(POi_1)')
    POi_2 = r('matrix(POi_2)')
    POi_3 = r('matrix(POi_3)')
    POi_4 = r('matrix(POi_4)')
    #POi_5 = r('matrix(POi_5)')
    ###################################################
    train_clindata_all = train_clindata_all.assign(risk_pso1 = np.array(POi_1,dtype=np.float64) ,
        risk_pso2 = np.array(POi_2,dtype=np.float64),
        risk_pso3 = np.array(POi_3,dtype=np.float64),
        risk_pso4 = np.array(POi_4,dtype=np.float64)       
        )
    #long_df = pd.melt(train_clindata_all, id_vars=['ID'],value_vars=['risk_pso1','risk_pso2','risk_pso3','risk_pso4','risk_pso5'] )
    long_df = pd.melt(train_clindata_all, id_vars=['ID'],value_vars=['risk_pso1','risk_pso2','risk_pso3','risk_pso4'] )   
    long_df.rename(columns={'variable': 'time_point','value': 'ps_risk'}, inplace=True)
    #mymap= {'risk_pso1': 'time1', 'risk_pso2': 'time2', 'risk_pso3': 'time3', 'risk_pso4': 'time4', 'risk_pso5': 'time5'  }
    mymap= {'risk_pso1': 'time1', 'risk_pso2': 'time2', 'risk_pso3': 'time3', 'risk_pso4': 'time4'  }
    long_df = long_df.applymap(lambda s : mymap.get(s) if s in mymap else s)
    train_val_clindata = pd.get_dummies(long_df, columns=['time_point'])
    #test_clindata_all = test_clindata_all.assign( time_point1=1,time_point2=2,time_point3=3,time_point4=4,time_point5=5 )
    test_clindata_all = test_clindata_all.assign( time_point1=1,time_point2=2,time_point3=3,time_point4=4 )
    #long_test_df = pd.melt(test_clindata_all, id_vars=['ID'],value_vars=['time_point1','time_point2','time_point3','time_point4','time_point5'] )
    long_test_df = pd.melt(test_clindata_all, id_vars=['ID'],value_vars=['time_point1','time_point2','time_point3','time_point4'] )
    long_test_df.rename(columns={'value': 'time_point'}, inplace=True)
    long_test_clindata_all = pd.merge(left=long_test_df, right=test_clindata_all, how='left',left_on='ID' ,right_on='ID')
    cols_test = long_test_clindata_all.columns.tolist()
    #long_test_clindata = long_test_clindata_all[ ['ID'] + ['time_point'] + ['time'] + ['event'] + ['event_1'] + ['event_2'] + ['event_3'] + ['event_4'] + ['event_5']]
    long_test_clindata = long_test_clindata_all[ ['ID'] + ['time_point'] + ['time'] + ['event'] + ['event_1'] + ['event_2'] + ['event_3'] + ['event_4'] ]
    long_test_clindata = pd.get_dummies(long_test_clindata, columns=['time_point'])
    covariates = df[['ID'] +  df.columns.tolist()[7:]]
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
    #event_5= np.where(true_times<= cutoff[4],1,0)    
    cens_perc = np.sum(censored_indicator)/num_samples
    cens_perc_train = np.sum(censored_indicator[:train_n])/train_n
    #df = np.concatenate((np.expand_dims(idx_sample,axis=1), np.expand_dims(times,axis=1),np.expand_dims(event,axis=1),
    #np.expand_dims(event_1,axis=1),np.expand_dims(event_2,axis=1),np.expand_dims(event_3,axis=1),np.expand_dims(event_4,axis=1),np.expand_dims(event_5,axis=1),clinical_data),axis=1)
    df = np.concatenate((np.expand_dims(idx_sample,axis=1), np.expand_dims(times,axis=1),np.expand_dims(event,axis=1),
    np.expand_dims(event_1,axis=1),np.expand_dims(event_2,axis=1),np.expand_dims(event_3,axis=1),np.expand_dims(event_4,axis=1),clinical_data),axis=1)
    #df = pd.DataFrame(df,columns= ('ID','time','event','event_1','event_2','event_3','event_4','event_5','cov1','cov2','cov3','cov4','cov5','cov6','cov7','cov8','cov9','cov10')) # the ID is the image chosen
    df = pd.DataFrame(df,columns= ('ID','time','event','event_1','event_2','event_3','event_4','cov1','cov2','cov3','cov4','cov5','cov6','cov7','cov8','cov9','cov10')) # the ID is the image chosen
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
    # IPCW Pseudo Observations 
    for i in (np.arange(10)+7):
        exec(f'cov_{i} = robjects.FloatVector(train_clindata_all.iloc[:,i]) ')
        exec(f'robjects.globalenv["cov_{i}"] = cov_{i}')
    r('xnam <- paste("cov_", 0:9, sep="")')
    r('data_surv_pso <- data.frame("time_r"=time_r,"event_r"=event_r , "cov_0"=cov_7, "cov_1"=cov_8, "cov_2"=cov_9, "cov_3"=cov_10,"cov_4"=cov_11,"cov_5"=cov_12,"cov_6"=cov_13, "cov_7"=cov_14, "cov_8"=cov_15,"cov_9"=cov_16) ' )
    r('POi_1 <- eventglm::pseudo_coxph(Surv(time_r, event_r) ~ 1, cutoff[1], cause = 1,data = data_surv_pso, type = "cuminc", formula.censoring = ~ cov_0+cov_1+cov_2+cov_3+cov_4+cov_5+cov_6+cov_7+cov_8+cov_9 , ipcw.method = "hajek" )' )
    r('POi_2 <- eventglm::pseudo_coxph(Surv(time_r, event_r) ~ 1, cutoff[2], cause = 1,data = data_surv_pso, type = "cuminc", formula.censoring = ~ cov_0+cov_1+cov_2+cov_3+cov_4+cov_5+cov_6+cov_7+cov_8+cov_9 , ipcw.method = "hajek" )' )
    r('POi_3 <- eventglm::pseudo_coxph(Surv(time_r, event_r) ~ 1, cutoff[3], cause = 1,data = data_surv_pso, type = "cuminc", formula.censoring = ~ cov_0+cov_1+cov_2+cov_3+cov_4+cov_5+cov_6+cov_7+cov_8+cov_9 , ipcw.method = "hajek" )' )
    r('POi_4 <- eventglm::pseudo_coxph(Surv(time_r, event_r) ~ 1, cutoff[4], cause = 1,data = data_surv_pso, type = "cuminc", formula.censoring = ~ cov_0+cov_1+cov_2+cov_3+cov_4+cov_5+cov_6+cov_7+cov_8+cov_9 , ipcw.method = "hajek" )' )
    #r('POi_5 <- eventglm::pseudo_coxph(Surv(time_r, event_r) ~ 1, cutoff[5], cause = 1,data = data_surv_pso, type = "cuminc", formula.censoring = ~ cov_0+cov_1+cov_2+cov_3+cov_4+cov_5+cov_6+cov_7+cov_8+cov_9 , ipcw.method = "hajek" )' )
    POi_1 = r('matrix(POi_1)')
    POi_2 = r('matrix(POi_2)')
    POi_3 = r('matrix(POi_3)')
    POi_4 = r('matrix(POi_4)')
    #POi_5 = r('matrix(POi_5)')
    ###################################################
    train_clindata_all = train_clindata_all.assign(risk_pso1 = np.array(POi_1,dtype=np.float64) ,
        risk_pso2 = np.array(POi_2,dtype=np.float64),
        risk_pso3 = np.array(POi_3,dtype=np.float64),
        risk_pso4 = np.array(POi_4,dtype=np.float64)       
        )
    #long_df = pd.melt(train_clindata_all, id_vars=['ID'],value_vars=['risk_pso1','risk_pso2','risk_pso3','risk_pso4','risk_pso5'] )
    long_df = pd.melt(train_clindata_all, id_vars=['ID'],value_vars=['risk_pso1','risk_pso2','risk_pso3','risk_pso4'] )   
    long_df.rename(columns={'variable': 'time_point','value': 'ps_risk'}, inplace=True)
    #mymap= {'risk_pso1': 'time1', 'risk_pso2': 'time2', 'risk_pso3': 'time3', 'risk_pso4': 'time4', 'risk_pso5': 'time5'  }
    mymap= {'risk_pso1': 'time1', 'risk_pso2': 'time2', 'risk_pso3': 'time3', 'risk_pso4': 'time4'  }
    long_df = long_df.applymap(lambda s : mymap.get(s) if s in mymap else s)
    train_val_clindata = pd.get_dummies(long_df, columns=['time_point'])
    #test_clindata_all = test_clindata_all.assign( time_point1=1,time_point2=2,time_point3=3,time_point4=4,time_point5=5 )
    test_clindata_all = test_clindata_all.assign( time_point1=1,time_point2=2,time_point3=3,time_point4=4 )
    #long_test_df = pd.melt(test_clindata_all, id_vars=['ID'],value_vars=['time_point1','time_point2','time_point3','time_point4','time_point5'] )
    long_test_df = pd.melt(test_clindata_all, id_vars=['ID'],value_vars=['time_point1','time_point2','time_point3','time_point4'] )
    long_test_df.rename(columns={'value': 'time_point'}, inplace=True)
    long_test_clindata_all = pd.merge(left=long_test_df, right=test_clindata_all, how='left',left_on='ID' ,right_on='ID')
    cols_test = long_test_clindata_all.columns.tolist()
    #long_test_clindata = long_test_clindata_all[ ['ID'] + ['time_point'] + ['time'] + ['event'] + ['event_1'] + ['event_2'] + ['event_3'] + ['event_4'] + ['event_5']]
    long_test_clindata = long_test_clindata_all[ ['ID'] + ['time_point'] + ['time'] + ['event'] + ['event_1'] + ['event_2'] + ['event_3'] + ['event_4'] ]
    long_test_clindata = pd.get_dummies(long_test_clindata, columns=['time_point'])
    covariates = df[['ID'] +  df.columns.tolist()[7:]]
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
    #event_5= np.where(true_times<= cutoff[4],1,0)    
    cens_perc = np.sum(censored_indicator)/num_samples
    cens_perc_train = np.sum(censored_indicator[:train_n])/train_n
    #df = np.concatenate((np.expand_dims(idx_sample,axis=1), np.expand_dims(times,axis=1),np.expand_dims(event,axis=1),
    #np.expand_dims(event_1,axis=1),np.expand_dims(event_2,axis=1),np.expand_dims(event_3,axis=1),np.expand_dims(event_4,axis=1),np.expand_dims(event_5,axis=1),clinical_data),axis=1)
    df = np.concatenate((np.expand_dims(idx_sample,axis=1), np.expand_dims(times,axis=1),np.expand_dims(event,axis=1),
    np.expand_dims(event_1,axis=1),np.expand_dims(event_2,axis=1),np.expand_dims(event_3,axis=1),np.expand_dims(event_4,axis=1),clinical_data),axis=1)
    #df = pd.DataFrame(df,columns= ('ID','time','event','event_1','event_2','event_3','event_4','event_5','cov1','cov2','cov3','cov4','cov5','cov6','cov7','cov8','cov9','cov10')) # the ID is the image chosen
    df = pd.DataFrame(df,columns= ('ID','time','event','event_1','event_2','event_3','event_4','cov1','cov2','cov3','cov4','cov5','cov6','cov7','cov8','cov9','cov10')) # the ID is the image chosen
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
    # IPCW Pseudo Observations 
    for i in (np.arange(10)+7):
        exec(f'cov_{i} = robjects.FloatVector(train_clindata_all.iloc[:,i]) ')
        exec(f'robjects.globalenv["cov_{i}"] = cov_{i}')
    r('xnam <- paste("cov_", 0:9, sep="")')
    r('data_surv_pso <- data.frame("time_r"=time_r,"event_r"=event_r , "cov_0"=cov_7, "cov_1"=cov_8, "cov_2"=cov_9, "cov_3"=cov_10,"cov_4"=cov_11,"cov_5"=cov_12,"cov_6"=cov_13, "cov_7"=cov_14, "cov_8"=cov_15,"cov_9"=cov_16) ' )
    r('POi_1 <- eventglm::pseudo_coxph(Surv(time_r, event_r) ~ 1, cutoff[1], cause = 1,data = data_surv_pso, type = "cuminc", formula.censoring = ~ cov_0+cov_1+cov_2+cov_3+cov_4+cov_5+cov_6+cov_7+cov_8+cov_9 , ipcw.method = "hajek" )' )
    r('POi_2 <- eventglm::pseudo_coxph(Surv(time_r, event_r) ~ 1, cutoff[2], cause = 1,data = data_surv_pso, type = "cuminc", formula.censoring = ~ cov_0+cov_1+cov_2+cov_3+cov_4+cov_5+cov_6+cov_7+cov_8+cov_9 , ipcw.method = "hajek" )' )
    r('POi_3 <- eventglm::pseudo_coxph(Surv(time_r, event_r) ~ 1, cutoff[3], cause = 1,data = data_surv_pso, type = "cuminc", formula.censoring = ~ cov_0+cov_1+cov_2+cov_3+cov_4+cov_5+cov_6+cov_7+cov_8+cov_9 , ipcw.method = "hajek" )' )
    r('POi_4 <- eventglm::pseudo_coxph(Surv(time_r, event_r) ~ 1, cutoff[4], cause = 1,data = data_surv_pso, type = "cuminc", formula.censoring = ~ cov_0+cov_1+cov_2+cov_3+cov_4+cov_5+cov_6+cov_7+cov_8+cov_9 , ipcw.method = "hajek" )' )
    #r('POi_5 <- eventglm::pseudo_coxph(Surv(time_r, event_r) ~ 1, cutoff[5], cause = 1,data = data_surv_pso, type = "cuminc", formula.censoring = ~ cov_0+cov_1+cov_2+cov_3+cov_4+cov_5+cov_6+cov_7+cov_8+cov_9 , ipcw.method = "hajek" )' )
    POi_1 = r('matrix(POi_1)')
    POi_2 = r('matrix(POi_2)')
    POi_3 = r('matrix(POi_3)')
    POi_4 = r('matrix(POi_4)')
    #POi_5 = r('matrix(POi_5)')
    ###################################################
    train_clindata_all = train_clindata_all.assign(risk_pso1 = np.array(POi_1,dtype=np.float64) ,
        risk_pso2 = np.array(POi_2,dtype=np.float64),
        risk_pso3 = np.array(POi_3,dtype=np.float64),
        risk_pso4 = np.array(POi_4,dtype=np.float64)       
        )
    #long_df = pd.melt(train_clindata_all, id_vars=['ID'],value_vars=['risk_pso1','risk_pso2','risk_pso3','risk_pso4','risk_pso5'] )
    long_df = pd.melt(train_clindata_all, id_vars=['ID'],value_vars=['risk_pso1','risk_pso2','risk_pso3','risk_pso4'] )   
    long_df.rename(columns={'variable': 'time_point','value': 'ps_risk'}, inplace=True)
    #mymap= {'risk_pso1': 'time1', 'risk_pso2': 'time2', 'risk_pso3': 'time3', 'risk_pso4': 'time4', 'risk_pso5': 'time5'  }
    mymap= {'risk_pso1': 'time1', 'risk_pso2': 'time2', 'risk_pso3': 'time3', 'risk_pso4': 'time4'  }
    long_df = long_df.applymap(lambda s : mymap.get(s) if s in mymap else s)
    train_val_clindata = pd.get_dummies(long_df, columns=['time_point'])
    #test_clindata_all = test_clindata_all.assign( time_point1=1,time_point2=2,time_point3=3,time_point4=4,time_point5=5 )
    test_clindata_all = test_clindata_all.assign( time_point1=1,time_point2=2,time_point3=3,time_point4=4 )
    #long_test_df = pd.melt(test_clindata_all, id_vars=['ID'],value_vars=['time_point1','time_point2','time_point3','time_point4','time_point5'] )
    long_test_df = pd.melt(test_clindata_all, id_vars=['ID'],value_vars=['time_point1','time_point2','time_point3','time_point4'] )
    long_test_df.rename(columns={'value': 'time_point'}, inplace=True)
    long_test_clindata_all = pd.merge(left=long_test_df, right=test_clindata_all, how='left',left_on='ID' ,right_on='ID')
    cols_test = long_test_clindata_all.columns.tolist()
    #long_test_clindata = long_test_clindata_all[ ['ID'] + ['time_point'] + ['time'] + ['event'] + ['event_1'] + ['event_2'] + ['event_3'] + ['event_4'] + ['event_5']]
    long_test_clindata = long_test_clindata_all[ ['ID'] + ['time_point'] + ['time'] + ['event'] + ['event_1'] + ['event_2'] + ['event_3'] + ['event_4'] ]
    long_test_clindata = pd.get_dummies(long_test_clindata, columns=['time_point'])
    covariates = df[['ID'] +  df.columns.tolist()[7:]]
    clindata = {'train_val':train_val_clindata , 'test':long_test_clindata, 'covariates': covariates,'time_train': train_clindata_all['time'], 'event_train': train_clindata_all['event'], 'slide_id_test': test_clindata_all['ID'], 'cutoff': cutoff , 'cens': cens_perc, 'cens_train': cens_perc_train}
    return clindata 


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

train_transformer = transforms.Compose([   
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=(-90,90)),    
    transforms.ToTensor()])




class PSODataset_train(Dataset):
    
    def __init__(self, clinical_data, transform): 
        self.images = torchvision.datasets.CIFAR10(root='C:/Users/pabgon/project_2/dataset_cifar', train=False,download=False, transform=transform)
        self.clinical_data = clinical_data
        self.id_images = clinical_data['ID'] 
    def __len__(self):
        return len(self.id_images)
    def __getitem__(self, idx):
        ID = int(self.id_images.iloc[idx])
        image = self.images.data[ID]
        image = np.transpose(image,(2,0,1))
        po = np.array(self.clinical_data.iloc[idx,1] ).astype(np.float32)
        time_point = np.array(self.clinical_data.iloc[idx,2:] ).astype(np.float32)
                
        return image, po, time_point,  ID


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
        time_point = np.array(self.clinical_data.iloc[idx,7:] ).astype(np.float32)
        time = np.array(self.clinical_data.iloc[idx,1] ).astype(np.float32)
        event = np.array(self.clinical_data.iloc[idx,2] ).astype(np.float32)
        event_1 = np.array(self.clinical_data.iloc[idx,3] ).astype(np.float32)
        event_2 = np.array(self.clinical_data.iloc[idx,4] ).astype(np.float32)
        event_3 = np.array(self.clinical_data.iloc[idx,5] ).astype(np.float32)
        event_4 = np.array(self.clinical_data.iloc[idx,6] ).astype(np.float32)
        #event_5 = np.array(self.clinical_data.iloc[idx,7] ).astype(np.float32)        
                
        return image, time, event, event_1, event_2, event_3, event_4, time_point,  ID
        

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
        self.projective = nn.Linear(1 + 14,1)
        self.nonlinearity = nn.LeakyReLU(inplace=True)
        
    def forward(self, image, clin_covariates ):
        x = self.vismodel(image)
        x = torch.flatten(x, 1)
        x = torch.cat((x, clin_covariates), dim=1)
        x = self.projective(x)
        x = self.nonlinearity(x)        
        return x

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
    

feature_extract = True 

def train_eval_resnet_sim(cifardataset,clindata,lr,niter):
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
    optimizer = optim.Adam(params_to_update,lr=lr)      
    batch_size = 2**8 # 2**8 2**9 or 2**7
    #split thge data
    idx_split = np.arange(len(clindata['train_val']))
    random.seed(5)
    random.shuffle(idx_split)
    split = int(.9*len(idx_split))
    train_idx = idx_split[:split]
    val_idx = idx_split[split:]
    train_clindata = clindata['train_val'].iloc[train_idx,]
    val_clindata = clindata['train_val'].iloc[val_idx,]
    ds_train = PSODataset_train(train_clindata,train_transformer)
    ds_val = PSODataset_train(val_clindata,train_transformer)
    trainloader = torch.utils.data.DataLoader(ds_train, batch_size=batch_size,shuffle=True)
    valloader = torch.utils.data.DataLoader(ds_val, batch_size=batch_size,shuffle=False)
    train_losses = []
    val_losses = []
    AUC1 = []
    AUC2 = []
    AUC3 = []
    AUC4 = []
    AUC5 = []
    for epoch in range(niter):  
        running_loss = 0.0
        epoch_steps = 0
        val_running_loss = 0.0
        val_epoch_steps = 0
        net.train()
        for i, data in enumerate(trainloader, 0):
            input_batch, po_batch, time_point_batch,  slide_id_batch = data          
            input_batch = input_batch.to(device)
            covariates_batch = np.concatenate([np.array(clindata['covariates'][clindata['covariates']['ID']== int(file)].iloc[:,1:]).astype(np.float32) for file in slide_id_batch])            
            covariates_batch = torch.from_numpy(covariates_batch).to(device)
            time_point_batch = time_point_batch.to(device)
            po_batch = po_batch.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            output_batch = net(input_batch.float(),torch.cat((time_point_batch, covariates_batch), 1))
            loss = criterion(output_batch,torch.unsqueeze(po_batch,axis=1)  )
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            #if i % 2000 == 1999:  # print every 2000 mini-batches
            #    print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,running_loss / epoch_steps))
            #    running_loss = 0.0
        train_loss_epoch = running_loss / epoch_steps  
        train_losses.append(train_loss_epoch)
        # Validation loss
        net.eval()
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                input_batch, po_batch, time_point_batch,  slide_id_batch = data          
                input_batch = input_batch.to(device)
                covariates_batch = np.concatenate([np.array(clindata['covariates'][clindata['covariates']['ID']== int(file)].iloc[:,1:]).astype(np.float32) for file in slide_id_batch])            
                covariates_batch = torch.from_numpy(covariates_batch).to(device)
                time_point_batch = time_point_batch.to(device)
                #po_batch = po_batch.to(device)
                output_batch = net(input_batch.float(),torch.cat((time_point_batch, covariates_batch), 1))
                output_batch = output_batch.data.cpu()
                val_loss = criterion(output_batch,torch.unsqueeze(po_batch,axis=1)  )
                val_running_loss += val_loss.item()
                val_epoch_steps += 1 
        val_loss_epoch = val_running_loss / val_epoch_steps  
        val_losses.append(val_loss_epoch)
        # Test loss       
        time = []
        event = []
        event_1 = []
        event_2 = []
        event_3 = []
        event_4 = []
        event_5 = []
        time_point = []
        output =  np.array([])
        slide_id =  np.array([])
        pred1_ind = []
        pred2_ind = []
        pred3_ind = []
        pred4_ind = []
        pred5_ind = []
        batch_size = 2**8 # 2**8 2**9 or 2**7        
        ds_test = Dataset_test(cifardataset.data,clindata['test'])
        testloader = torch.utils.data.DataLoader(ds_test, batch_size=batch_size,shuffle=False) 
        with torch.no_grad():
            for i,(input_batch, time_batch, event_batch, event_batch_1, event_batch_2, event_batch_3, event_batch_4, time_point_batch ,  slide_id_batch) in enumerate(testloader,0):           
                input_batch = input_batch.to(device)
                covariates_batch = np.concatenate([np.array(clindata['covariates'][clindata['covariates']['ID']== int(file)].iloc[:,1:]).astype(np.float32) for file in slide_id_batch])            
                covariates_batch = torch.from_numpy(covariates_batch).to(device)
                time_point.append(time_point_batch)
                time_point_batch = time_point_batch.to(device)                              
                output_batch = net(input_batch.float(), torch.cat((time_point_batch, covariates_batch), 1))            
                output_batch = output_batch.data.cpu()           
                output = np.concatenate([output, np.squeeze(output_batch.detach().numpy(),axis=1)])
                slide_id = np.concatenate([slide_id, np.array(slide_id_batch)])           
                time.append(time_batch.numpy())
                event.append(event_batch.numpy())
                event_1.append(event_batch_1)
                event_2.append(event_batch_2)
                event_3.append(event_batch_3)
                event_4.append(event_batch_4)
                #event_5.append(event_batch_5)
        time_point = np.concatenate(time_point)
        time = np.concatenate(time)
        event = np.concatenate(event)
        event1 = np.concatenate(event_1)
        event2 = np.concatenate(event_2)
        event3 = np.concatenate(event_3)
        event4 = np.concatenate(event_4)
        #event5 = np.concatenate(event_5)
        indx_1=(time_point[:,0]==1)
        indx_2=(time_point[:,1]==1)
        indx_3=(time_point[:,2]==1)
        indx_4=(time_point[:,3]==1)
        #indx_5=(time_point[:,4]==1)
        event1 = event1[indx_1]
        event2 = event2[indx_2]
        event3 = event3[indx_3]
        event4 = event4[indx_4]
        #event5 = event5[indx_5]    
        pred1_ind = output[indx_1]    
        pred2_ind = output[indx_2] 
        pred3_ind = output[indx_3]
        pred4_ind = output[indx_4]        
        #pred5_ind = output[indx_5]        
        auc1 = roc_auc(pred1_ind,event1)
        auc2 = roc_auc(pred2_ind,event2)
        auc3 = roc_auc(pred3_ind,event3)
        auc4 = roc_auc(pred4_ind,event4)
        #auc5 = roc_auc(pred5_ind,event5)
        AUC1.append(auc1)
        AUC2.append(auc2)
        AUC3.append(auc3)
        AUC4.append(auc4)
        #AUC5.append(auc5)
        #auc = {'cutoff_1':auc1,'cutoff_2': auc2, 'cutoff_3': auc3, 'cutoff_4':auc4, 'cutoff_5':auc5}
        
    return train_losses, val_losses, AUC1, AUC2, AUC3, AUC4


def main():
    # GLOBAL PARAMETERS
    num_sim = 100
    sample_size = 5000
    dataset = torchvision.datasets.CIFAR10(root='C:/Users/pabgon/project_2/dataset_cifar', train=False,download=False, transform=transform)
    #dataset = torchvision.datasets.CIFAR10(root='C:/Users/pabgon/project_2/dataset_cifar', train=False,download=False, transform=transform)
    ##################
    # CASE 1 #
    model_dir = 'case1_N5000'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    print(model_dir)
    AUC1_matrix = np.zeros((25,num_sim))
    AUC2_matrix = np.zeros((25,num_sim))
    AUC3_matrix = np.zeros((25,num_sim))
    AUC4_matrix = np.zeros((25,num_sim))
    AUC5_matrix = np.zeros((25,num_sim))
    train_loss_matrix = np.zeros((25,num_sim))
    val_loss_matrix = np.zeros((25,num_sim))
    for i in range(num_sim):
        clindata = sim_event_times_case1(dataset, sample_size)
        train_loss_j, val_loss_j, auc1_j, auc2_j, auc3_j, auc4_j = train_eval_resnet_sim(dataset, clindata, 0.01, 25) # this lr=0.01 works well
        AUC1_matrix[:,i] = auc1_j
        AUC2_matrix[:,i] = auc2_j
        AUC3_matrix[:,i] = auc3_j
        AUC4_matrix[:,i] = auc4_j
        #AUC5_matrix[:,i] = auc5_j
        train_loss_matrix[:,i] = train_loss_j
        val_loss_matrix[:,i] = val_loss_j
        if i % 25 == 0:
            print(i)
            
    #auc1    
    auc1_json_path = os.path.join(model_dir,"auc1_ipcwpo_so")
    np.save(auc1_json_path,AUC1_matrix)   
    #auc2
    auc2_json_path = os.path.join(model_dir,"auc2_ipcwpo_so")
    np.save(auc2_json_path,AUC2_matrix)
    #auc3
    auc3_json_path = os.path.join(model_dir,"auc3_ipcwpo_so")
    np.save(auc3_json_path,AUC3_matrix)
    #auc4
    auc4_json_path = os.path.join(model_dir,"auc4_ipcwpo_so")
    np.save(auc4_json_path,AUC4_matrix)
    #auc5
    #auc5_json_path = os.path.join(model_dir,"auc5_ipcwpo_so")
    #np.save(auc5_json_path,AUC5_matrix)
    #training loss
    train_loss_path = os.path.join(model_dir,"train_loss_ipcwpo_so")
    np.save(train_loss_path,train_loss_matrix)
    #val loss
    val_loss_path = os.path.join(model_dir,"val_loss_ipcwpo_so")
    np.save(val_loss_path,val_loss_matrix)
        
    ###########
    # CASE 2 # 
    model_dir =  'case2_N5000'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    print(model_dir)
    AUC1_matrix = np.zeros((25,num_sim))
    AUC2_matrix = np.zeros((25,num_sim))
    AUC3_matrix = np.zeros((25,num_sim))
    AUC4_matrix = np.zeros((25,num_sim))
    AUC5_matrix = np.zeros((25,num_sim))
    train_loss_matrix = np.zeros((25,num_sim))
    val_loss_matrix = np.zeros((25,num_sim))
    for i in range(num_sim):
        clindata = sim_event_times_case2(dataset, sample_size)
        train_loss_j, val_loss_j, auc1_j, auc2_j, auc3_j, auc4_j = train_eval_resnet_sim(dataset, clindata, 0.01, 25) 
        AUC1_matrix[:,i] = auc1_j
        AUC2_matrix[:,i] = auc2_j
        AUC3_matrix[:,i] = auc3_j
        AUC4_matrix[:,i] = auc4_j
        #AUC5_matrix[:,i] = auc5_j
        train_loss_matrix[:,i] = train_loss_j
        val_loss_matrix[:,i] = val_loss_j
        if i % 25 == 0:
            print(i)
            
    #auc1    
    auc1_json_path = os.path.join(model_dir,"auc1_ipcwpo_so")
    np.save(auc1_json_path,AUC1_matrix)   
    #auc2
    auc2_json_path = os.path.join(model_dir,"auc2_ipcwpo_so")
    np.save(auc2_json_path,AUC2_matrix)
    #auc3
    auc3_json_path = os.path.join(model_dir,"auc3_ipcwpo_so")
    np.save(auc3_json_path,AUC3_matrix)
    #auc4
    auc4_json_path = os.path.join(model_dir,"auc4_ipcwpo_so")
    np.save(auc4_json_path,AUC4_matrix)
    #auc5
    #auc5_json_path = os.path.join(model_dir,"auc5_ipcwpo_so")
    #np.save(auc5_json_path,AUC5_matrix)
    #training loss
    train_loss_path = os.path.join(model_dir,"train_loss_ipcwpo_so")
    np.save(train_loss_path,train_loss_matrix)
    #val loss
    val_loss_path = os.path.join(model_dir,"val_loss_ipcwpo_so")
    np.save(val_loss_path,val_loss_matrix)
    ################################################################
    # CASE 3 #
    model_dir =  'case3_N5000'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    print(model_dir)
    AUC1_matrix = np.zeros((25,num_sim))
    AUC2_matrix = np.zeros((25,num_sim))
    AUC3_matrix = np.zeros((25,num_sim))
    AUC4_matrix = np.zeros((25,num_sim))
    AUC5_matrix = np.zeros((25,num_sim))
    train_loss_matrix = np.zeros((25,num_sim))
    val_loss_matrix = np.zeros((25,num_sim))
    for i in range(num_sim):
        clindata = sim_event_times_case3(dataset, sample_size)
        train_loss_j, val_loss_j, auc1_j, auc2_j, auc3_j, auc4_j = train_eval_resnet_sim(dataset, clindata, 0.01, 25) 
        AUC1_matrix[:,i] = auc1_j
        AUC2_matrix[:,i] = auc2_j
        AUC3_matrix[:,i] = auc3_j
        AUC4_matrix[:,i] = auc4_j
        #AUC5_matrix[:,i] = auc5_j
        train_loss_matrix[:,i] = train_loss_j
        val_loss_matrix[:,i] = val_loss_j
        if i % 25 == 0:
            print(i)
            
    #auc1    
    auc1_json_path = os.path.join(model_dir,"auc1_ipcwpo_so")
    np.save(auc1_json_path,AUC1_matrix)   
    #auc2
    auc2_json_path = os.path.join(model_dir,"auc2_ipcwpo_so")
    np.save(auc2_json_path,AUC2_matrix)
    #auc3
    auc3_json_path = os.path.join(model_dir,"auc3_ipcwpo_so")
    np.save(auc3_json_path,AUC3_matrix)
    #auc4
    auc4_json_path = os.path.join(model_dir,"auc4_ipcwpo_so")
    np.save(auc4_json_path,AUC4_matrix)
    #auc5
    #auc5_json_path = os.path.join(model_dir,"auc5_ipcwpo_so")
    #np.save(auc5_json_path,AUC5_matrix)
    #training loss
    train_loss_path = os.path.join(model_dir,"train_loss_ipcwpo_so")
    np.save(train_loss_path,train_loss_matrix)
    #val loss
    val_loss_path = os.path.join(model_dir,"val_loss_ipcwpo_so")
    np.save(val_loss_path,val_loss_matrix)
    ################################################################
    # CASE 4 #
    model_dir =  'case4_N5000'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    print(model_dir)
    AUC1_matrix = np.zeros((25,num_sim))
    AUC2_matrix = np.zeros((25,num_sim))
    AUC3_matrix = np.zeros((25,num_sim))
    AUC4_matrix = np.zeros((25,num_sim))
    AUC5_matrix = np.zeros((25,num_sim))
    train_loss_matrix = np.zeros((25,num_sim))
    val_loss_matrix = np.zeros((25,num_sim))
    for i in range(num_sim):
        clindata = sim_event_times_case4(dataset, sample_size)
        train_loss_j, val_loss_j, auc1_j, auc2_j, auc3_j, auc4_j= train_eval_resnet_sim(dataset, clindata, 0.01, 25) 
        AUC1_matrix[:,i] = auc1_j
        AUC2_matrix[:,i] = auc2_j
        AUC3_matrix[:,i] = auc3_j
        AUC4_matrix[:,i] = auc4_j
        #AUC5_matrix[:,i] = auc5_j
        train_loss_matrix[:,i] = train_loss_j
        val_loss_matrix[:,i] = val_loss_j
        if i % 25 == 0:
            print(i)
            
    #auc1    
    auc1_json_path = os.path.join(model_dir,"auc1_ipcwpo_so")
    np.save(auc1_json_path,AUC1_matrix)   
    #auc2
    auc2_json_path = os.path.join(model_dir,"auc2_ipcwpo_so")
    np.save(auc2_json_path,AUC2_matrix)
    #auc3
    auc3_json_path = os.path.join(model_dir,"auc3_ipcwpo_so")
    np.save(auc3_json_path,AUC3_matrix)
    #auc4
    auc4_json_path = os.path.join(model_dir,"auc4_ipcwpo_so")
    np.save(auc4_json_path,AUC4_matrix)
    #auc5
    #auc5_json_path = os.path.join(model_dir,"auc5_ipcwpo_so")
    #np.save(auc5_json_path,AUC5_matrix)
    #training loss
    train_loss_path = os.path.join(model_dir,"train_loss_ipcwpo_so")
    np.save(train_loss_path,train_loss_matrix)
    #val loss
    val_loss_path = os.path.join(model_dir,"val_loss_ipcwpo_so")
    np.save(val_loss_path,val_loss_matrix)
    
    ################################################################
    # CASE 5 #
    model_dir =  'case5_N5000'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    print(model_dir)
    AUC1_matrix = np.zeros((25,num_sim))
    AUC2_matrix = np.zeros((25,num_sim))
    AUC3_matrix = np.zeros((25,num_sim))
    AUC4_matrix = np.zeros((25,num_sim))
    AUC5_matrix = np.zeros((25,num_sim))
    train_loss_matrix = np.zeros((25,num_sim))
    val_loss_matrix = np.zeros((25,num_sim))
    for i in range(num_sim):
        clindata = sim_event_times_case5(dataset, sample_size)
        train_loss_j, val_loss_j, auc1_j, auc2_j, auc3_j, auc4_j = train_eval_resnet_sim(dataset, clindata, 0.01, 25) 
        AUC1_matrix[:,i] = auc1_j
        AUC2_matrix[:,i] = auc2_j
        AUC3_matrix[:,i] = auc3_j
        AUC4_matrix[:,i] = auc4_j
        #AUC5_matrix[:,i] = auc5_j
        train_loss_matrix[:,i] = train_loss_j
        val_loss_matrix[:,i] = val_loss_j
        if i % 25 == 0:
            print(i)
            
    #auc1    
    auc1_json_path = os.path.join(model_dir,"auc1_ipcwpo_so")
    np.save(auc1_json_path,AUC1_matrix)   
    #auc2
    auc2_json_path = os.path.join(model_dir,"auc2_ipcwpo_so")
    np.save(auc2_json_path,AUC2_matrix)
    #auc3
    auc3_json_path = os.path.join(model_dir,"auc3_ipcwpo_so")
    np.save(auc3_json_path,AUC3_matrix)
    #auc4
    auc4_json_path = os.path.join(model_dir,"auc4_ipcwpo_so")
    np.save(auc4_json_path,AUC4_matrix)
    #auc5
    #auc5_json_path = os.path.join(model_dir,"auc5_ipcwpo_so")
    #np.save(auc5_json_path,AUC5_matrix)
    #training loss
    train_loss_path = os.path.join(model_dir,"train_loss_ipcwpo_so")
    np.save(train_loss_path,train_loss_matrix)
    #val loss
    val_loss_path = os.path.join(model_dir,"val_loss_ipcwpo_so")
    np.save(val_loss_path,val_loss_matrix)
    
    #################################################################
    # CASE 6 #
    model_dir =  'case6_N5000'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    print(model_dir)
    AUC1_matrix = np.zeros((25,num_sim))
    AUC2_matrix = np.zeros((25,num_sim))
    AUC3_matrix = np.zeros((25,num_sim))
    AUC4_matrix = np.zeros((25,num_sim))
    AUC5_matrix = np.zeros((25,num_sim))
    train_loss_matrix = np.zeros((25,num_sim))
    val_loss_matrix = np.zeros((25,num_sim))
    for i in range(num_sim):
        clindata = sim_event_times_case6(dataset, sample_size)
        train_loss_j, val_loss_j, auc1_j, auc2_j, auc3_j, auc4_j = train_eval_resnet_sim(dataset, clindata, 0.01, 25) 
        AUC1_matrix[:,i] = auc1_j
        AUC2_matrix[:,i] = auc2_j
        AUC3_matrix[:,i] = auc3_j
        AUC4_matrix[:,i] = auc4_j
        #AUC5_matrix[:,i] = auc5_j
        train_loss_matrix[:,i] = train_loss_j
        val_loss_matrix[:,i] = val_loss_j
        if i % 25 == 0:
            print(i)
            
    #auc1    
    auc1_json_path = os.path.join(model_dir,"auc1_ipcwpo_so")
    np.save(auc1_json_path,AUC1_matrix)   
    #auc2
    auc2_json_path = os.path.join(model_dir,"auc2_ipcwpo_so")
    np.save(auc2_json_path,AUC2_matrix)
    #auc3
    auc3_json_path = os.path.join(model_dir,"auc3_ipcwpo_so")
    np.save(auc3_json_path,AUC3_matrix)
    #auc4
    auc4_json_path = os.path.join(model_dir,"auc4_ipcwpo_so")
    np.save(auc4_json_path,AUC4_matrix)
    #auc5
    #auc5_json_path = os.path.join(model_dir,"auc5_ipcwpo_so")
    #np.save(auc5_json_path,AUC5_matrix)
    #training loss
    train_loss_path = os.path.join(model_dir,"train_loss_ipcwpo_so")
    np.save(train_loss_path,train_loss_matrix)
    #val loss
    val_loss_path = os.path.join(model_dir,"val_loss_ipcwpo_so")
    np.save(val_loss_path,val_loss_matrix)
    

    
if __name__ == '__main__':
    main()