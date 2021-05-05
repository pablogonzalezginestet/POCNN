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

    
def get_clinical_data_cox(df, cutoff):
    '''
    Arg : (cleaned)clinical data  

    Returns : a dictionary that contains the clinical data for train, val and test

    '''
    data_converted = pd.get_dummies(df,columns= ['race' ,'ethnicity' ,'pathologic_stage' ,'molecular_subtype'],dtype=float)
    train_clindata_all = data_converted[data_converted['Test']=='False']
    test_clindata = data_converted[data_converted['Test']=='True'] 
    train_clindata_all = train_clindata_all.drop('Test',axis=1)
    test_clindata = test_clindata.drop('Test',axis=1)   
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
    # Validation
    order_time_val = np.argsort(val_clindata['time_to_event'])
    val_clindata = val_clindata.iloc[order_time_val,:]
    # Test 
    test_clindata = test_clindata[ ['ID_slide'] +  ['time_to_event'] + ['event'] ]
    # Covariates
    cols_cov = data_converted.columns.tolist()[4:]
    covariates = data_converted[['ID_slide'] + cols_cov]   
    clindata = {'train': train_clindata ,'val': val_clindata, 'test':test_clindata, 'covariates': covariates, 'time_train': train_clindata_all[['time_to_event']], 'event_train': train_clindata_all[['event']], 'slide_id_test': test_clindata['ID_slide'] }
    return clindata