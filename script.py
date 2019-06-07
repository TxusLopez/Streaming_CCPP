import sys
from sklearn.naive_bayes import GaussianNB
from scipy.io import loadmat
from sklearn import preprocessing
from sklearn.metrics import cohen_kappa_score,confusion_matrix,classification_report
from sklearn.ensemble import ExtraTreesClassifier
from skmultiflow.trees import HoeffdingTree,HAT
from skmultiflow.lazy.knn import KNN
from skmultiflow.bayes import NaiveBayes
from sklearn.linear_model import SGDRegressor
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import RandomizedSearchCV
from skmultiflow.trees.regression_hoeffding_tree import RegressionHoeffdingTree
from skmultiflow.trees.regression_hoeffding_adaptive_tree import RegressionHAT
from sklearn.metrics import explained_variance_score,mean_absolute_error,mean_squared_error,mean_squared_log_error,median_absolute_error,r2_score
from sklearn.linear_model import PassiveAggressiveRegressor
from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.drift_detection.eddm import EDDM
from pandas.plotting import scatter_matrix
from scipy import stats
from sklearn.neural_network import MLPRegressor
from copy import deepcopy
from timeit import default_timer as timer
from skmultiflow.data.file_stream import FileStream
from skmultiflow.evaluation.evaluate_prequential_ENERGIA_v2 import EvaluatePrequential_ENERGIA_v2
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from skgarden import MondrianForestRegressor,MondrianTreeRegressor,DecisionTreeQuantileRegressor,ExtraTreeQuantileRegressor,ExtraTreesQuantileRegressor,RandomForestQuantileRegressor,ExtraTreesRegressor,RandomForestRegressor

import matplotlib.pyplot as plt
import pandas as pd
import h5py
import numpy as np
import seaborn as sns
import pickle
import scipy.io as sio
import warnings
#sns.set()
import matplotlib.style as style

style.use('seaborn-paper')

warnings.filterwarnings("ignore",category=DeprecationWarning)

#==============================================================================
# CLASSES
#==============================================================================


#==============================================================================
# FUNCTIONS
#==============================================================================

def fxn():
    warnings.warn("deprecated", DeprecationWarning) 

def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out

def savingResults(output_pickle,RUNS_MSE,RUNS_MAE,RUNS_RMSE,RUNS_R2,feature_selection,RUNS_hyperparam_regressors,RUN_real_predicciones,RUN_real_values,preparatory_size):
    
    if feature_selection:
        extra_name='_feat_'+str(preparatory_size)+'.pkl'

        output = open(output_pickle+'RUNS_MAE'+extra_name, 'wb')
        pickle.dump(RUNS_MAE, output)
        output.close()
#        sio.savemat(output_pickle+'RUNS_MAE_feat.mat', {'RUNS_MAE_feat':RUNS_MAE})

        output = open(output_pickle+'RUNS_MSE'+extra_name, 'wb')
        pickle.dump(RUNS_MSE, output)
        output.close()
#        sio.savemat(output_pickle+'RUNS_MSE_feat.mat', {'RUNS_MSE_feat':RUNS_MSE})

        output = open(output_pickle+'RUNS_RMSE'+extra_name, 'wb')
        pickle.dump(RUNS_RMSE, output)
        output.close()
#        sio.savemat(output_pickle+'RUNS_RMSE_feat.mat', {'RUNS_RMSE_feat':RUNS_RMSE})

        output = open(output_pickle+'RUNS_R2'+extra_name, 'wb')
        pickle.dump(RUNS_R2, output)
        output.close()
#        sio.savemat(output_pickle+'RUNS_R2_feat.mat', {'RUNS_R2_feat':RUNS_R2})
    
        #RUNS_hyperparam_regressors
        output = open(output_pickle+'RUNS_hyperparam_regressors'+extra_name, 'wb')
        pickle.dump(RUNS_hyperparam_regressors, output)
        output.close()

        #RUN_real_predicciones
        output = open(output_pickle+'RUN_real_predicciones'+extra_name, 'wb')
        pickle.dump(RUN_real_predicciones, output)
        output.close()

        #RUN_real_values
        output = open(output_pickle+'RUN_real_values'+extra_name, 'wb')
        pickle.dump(RUN_real_values, output)
        output.close()
    
    else:
        extra_name='_'+str(preparatory_size)+'.pkl'
        
        output = open(output_pickle+'RUNS_MAE'+extra_name, 'wb')
        pickle.dump(RUNS_MAE, output)
        output.close()
#        sio.savemat(output_pickle+'RUNS_MAE.mat', {'RUNS_MAE':RUNS_MAE})

        output = open(output_pickle+'RUNS_MSE'+extra_name, 'wb')
        pickle.dump(RUNS_MSE, output)
        output.close()
#        sio.savemat(output_pickle+'RUNS_MSE.mat', {'RUNS_MSE':RUNS_MSE})

        output = open(output_pickle+'RUNS_RMSE'+extra_name, 'wb')
        pickle.dump(RUNS_RMSE, output)
        output.close()
#        sio.savemat(output_pickle+'RUNS_RMSE.mat', {'RUNS_RMSE':RUNS_RMSE})

        output = open(output_pickle+'RUNS_R2'+extra_name, 'wb')
        pickle.dump(RUNS_R2, output)
        output.close()
#        sio.savemat(output_pickle+'RUNS_R2.mat', {'RUNS_R2':RUNS_R2})

        #RUNS_hyperparam_regressors
        output = open(output_pickle+'RUNS_hyperparam_regressors'+extra_name, 'wb')
        pickle.dump(RUNS_hyperparam_regressors, output)
        output.close()

        #RUN_real_predicciones
        output = open(output_pickle+'RUN_real_predicciones'+extra_name, 'wb')
        pickle.dump(RUN_real_predicciones, output)
        output.close()

        #RUN_real_values
        output = open(output_pickle+'RUN_real_values'+extra_name, 'wb')
        pickle.dump(RUN_real_values, output)
        output.close()


def savingTime(output_pickle,RUNS_PAR_total_time,RUNS_SGDR_total_time,RUNS_MLPR_total_time,RUNS_RHT_total_time,RUNS_RHAT_total_time,RUNS_MFR_total_time,RUNS_MTR_total_time,feature_selection,preparatory_size):
    
    if feature_selection:
        extra_name='_time_feat_'+str(preparatory_size)+'.pkl'

        #RUNS_PAR_total_time
        output = open(output_pickle+'RUNS_PAR'+extra_name, 'wb')
        pickle.dump(RUNS_PAR_total_time, output)
        output.close()
#        sio.savemat(output_pickle+'RUNS_PAR_total_time_feat.mat', {'RUNS_PAR_total_time_feat':RUNS_PAR_total_time})
    
        #RUNS_SGDR_total_time
        output = open(output_pickle+'RUNS_SGDR'+extra_name, 'wb')
        pickle.dump(RUNS_SGDR_total_time, output)
        output.close()
#        sio.savemat(output_pickle+'RUNS_SGDR_total_time_feat.mat', {'RUNS_SGDR_total_time_feat':RUNS_SGDR_total_time})
        
        #RUNS_MLPR_total_time
        output = open(output_pickle+'RUNS_MLPR'+extra_name, 'wb')
        pickle.dump(RUNS_MLPR_total_time, output)
        output.close()
#        sio.savemat(output_pickle+'RUNS_MLPR_total_time_feat.mat', {'RUNS_MLPR_total_time_feat':RUNS_MLPR_total_time})
    
        #RUNS_RHT_total_time
        output = open(output_pickle+'RUNS_RHT'+extra_name, 'wb')
        pickle.dump(RUNS_RHT_total_time, output)
        output.close()
#        sio.savemat(output_pickle+'RUNS_RHT_total_time_feat.mat', {'RUNS_RHT_total_time_feat':RUNS_RHT_total_time})
    
        #RUNS_RHAT_total_time
        output = open(output_pickle+'RUNS_RHAT'+extra_name, 'wb')
        pickle.dump(RUNS_RHAT_total_time, output)
        output.close()
#        sio.savemat(output_pickle+'RUNS_RHAT_total_time_feat.mat', {'RUNS_RHAT_total_time_feat':RUNS_RHAT_total_time})

        #RUNS_MFR_total_time
        output = open(output_pickle+'RUNS_MFR'+extra_name, 'wb')
        pickle.dump(RUNS_MFR_total_time, output)
        output.close()
#        sio.savemat(output_pickle+'RUNS_MFR_total_time_feat.mat', {'RUNS_MFR_total_time_feat':RUNS_MFR_total_time})

        #RUNS_MTR_total_time
        output = open(output_pickle+'RUNS_MTR'+extra_name, 'wb')
        pickle.dump(RUNS_MTR_total_time, output)
        output.close()
#        sio.savemat(output_pickle+'RUNS_MTR_total_time_feat.mat', {'RUNS_MTR_total_time_feat':RUNS_MTR_total_time})

    else:
        extra_name='_time_'+str(preparatory_size)+'.pkl'

        #RUNS_PAR_total_time
        output = open(output_pickle+'RUNS_PAR'+extra_name, 'wb')
        pickle.dump(RUNS_PAR_total_time, output)
        output.close()
#        sio.savemat(output_pickle+'RUNS_PAR_total_time.mat', {'RUNS_PAR_total_time':RUNS_PAR_total_time})
    
        #RUNS_SGDR_total_time
        output = open(output_pickle+'RUNS_SGDR'+extra_name, 'wb')
        pickle.dump(RUNS_SGDR_total_time, output)
        output.close()
#        sio.savemat(output_pickle+'RUNS_SGDR_total_time.mat', {'RUNS_SGDR_total_time':RUNS_SGDR_total_time})
        
        #RUNS_MLPR_total_time
        output = open(output_pickle+'RUNS_MLPR'+extra_name, 'wb')
        pickle.dump(RUNS_MLPR_total_time, output)
        output.close()
#        sio.savemat(output_pickle+'RUNS_MLPR_total_time.mat', {'RUNS_MLPR_total_time':RUNS_MLPR_total_time})
    
        #RUNS_RHT_total_time
        output = open(output_pickle+'RUNS_RHT'+extra_name, 'wb')
        pickle.dump(RUNS_RHT_total_time, output)
        output.close()
#        sio.savemat(output_pickle+'RUNS_RHT_total_time.mat', {'RUNS_RHT_total_time':RUNS_RHT_total_time})
    
        #RUNS_RHAT_total_time
        output = open(output_pickle+'RUNS_RHAT'+extra_name, 'wb')
        pickle.dump(RUNS_RHAT_total_time, output)
        output.close()
#        sio.savemat(output_pickle+'RUNS_RHAT_total_time.mat', {'RUNS_RHAT_total_time':RUNS_RHAT_total_time})

        #RUNS_MFR_total_time
        output = open(output_pickle+'RUNS_MFR'+extra_name, 'wb')
        pickle.dump(RUNS_MFR_total_time, output)
        output.close()
#        sio.savemat(output_pickle+'RUNS_MFR_total_time.mat', {'RUNS_MFR_total_time':RUNS_MFR_total_time})

        #RUNS_MTR_total_time
        output = open(output_pickle+'RUNS_MTR'+extra_name, 'wb')
        pickle.dump(RUNS_MTR_total_time, output)
        output.close()
#        sio.savemat(output_pickle+'RUNS_MTR_total_time.mat', {'RUNS_MTR_total_time':RUNS_MTR_total_time})
 
def loadingResults(output_pickle,feature_selection,preparatory_size):
    
    if feature_selection:
        extra_name='_feat_'+str(preparatory_size)+'.pkl'

        #MSE
        fil = open(output_pickle+'RUNS_MSE'+extra_name,'rb')
        RUNS_MSE = pickle.load(fil)
        fil.close()
        
        #RMSE
        fil = open(output_pickle+'RUNS_RMSE'+extra_name,'rb')
        RUNS_RMSE = pickle.load(fil)
        fil.close()
        
        #MAE
        fil = open(output_pickle+'RUNS_MAE'+extra_name,'rb')
        RUNS_MAE = pickle.load(fil)
        fil.close()
        
        #R2
        fil = open(output_pickle+'RUNS_R2'+extra_name,'rb')
        RUNS_R2 = pickle.load(fil)
        fil.close()

        #RUNS_hyperparam_regressors
        fil = open(output_pickle+'RUNS_hyperparam_regressors'+extra_name,'rb')
        RUNS_hyperparam_regressors = pickle.load(fil)
        fil.close()
        
        #RUN_real_predicciones
        fil = open(output_pickle+'RUN_real_predicciones'+extra_name,'rb')
        RUN_real_predicciones = pickle.load(fil)
        fil.close()        

        #RUN_real_values
        fil = open(output_pickle+'RUN_real_values'+extra_name,'rb')
        RUN_real_values = pickle.load(fil)
        fil.close()        
        
    else:
        extra_name='_'+str(preparatory_size)+'.pkl'
        
        #MSE
        fil = open(output_pickle+'RUNS_MSE'+extra_name,'rb')
        RUNS_MSE = pickle.load(fil)
        fil.close()
        
        #RMSE
        fil = open(output_pickle+'RUNS_RMSE'+extra_name,'rb')
        RUNS_RMSE = pickle.load(fil)
        fil.close()
        
        #MAE
        fil = open(output_pickle+'RUNS_MAE'+extra_name,'rb')
        RUNS_MAE = pickle.load(fil)
        fil.close()
        
        #R2
        fil = open(output_pickle+'RUNS_R2'+extra_name,'rb')
        RUNS_R2 = pickle.load(fil)
        fil.close()
    
        #RUNS_hyperparam_regressors
        fil = open(output_pickle+'RUNS_hyperparam_regressors'+extra_name,'rb')
        RUNS_hyperparam_regressors = pickle.load(fil)
        fil.close()
        
        #RUN_real_predicciones
        fil = open(output_pickle+'RUN_real_predicciones'+extra_name,'rb')
        RUN_real_predicciones = pickle.load(fil)
        fil.close()        

        #RUN_real_values
        fil = open(output_pickle+'RUN_real_values'+extra_name,'rb')
        RUN_real_values = pickle.load(fil)
        fil.close()        
        
        
    return RUNS_MSE,RUNS_MAE,RUNS_RMSE,RUNS_R2,RUNS_hyperparam_regressors,RUN_real_predicciones,RUN_real_values

def loadingTime(output_pickle,feature_selection,preparatory_size):
    
    if feature_selection:
        extra_name='_time_feat_'+str(preparatory_size)+'.pkl'
        
        fil = open(output_pickle+'RUNS_PAR'+extra_name,'rb')
        RUNS_PAR_total_time = pickle.load(fil)
        fil.close()
        
        fil = open(output_pickle+'RUNS_SGDR'+extra_name,'rb')
        RUNS_SGDR_total_time = pickle.load(fil)
        fil.close()
    
        fil = open(output_pickle+'RUNS_MLPR'+extra_name,'rb')
        RUNS_MLPR_total_time = pickle.load(fil)
        fil.close()

        fil = open(output_pickle+'RUNS_RHT'+extra_name,'rb')
        RUNS_RHT_total_time = pickle.load(fil)
        fil.close()

        fil = open(output_pickle+'RUNS_RHAT'+extra_name,'rb')
        RUNS_RHAT_total_time = pickle.load(fil)
        fil.close()

        fil = open(output_pickle+'RUNS_MFR'+extra_name,'rb')
        RUNS_MFR_total_time = pickle.load(fil)
        fil.close()

        fil = open(output_pickle+'RUNS_MTR'+extra_name,'rb')
        RUNS_MTR_total_time = pickle.load(fil)
        fil.close()
        
    else:
        extra_name='_time_'+str(preparatory_size)+'.pkl'

        fil = open(output_pickle+'RUNS_PAR'+extra_name,'rb')
        RUNS_PAR_total_time = pickle.load(fil)
        fil.close()
        
        fil = open(output_pickle+'RUNS_SGDR'+extra_name,'rb')
        RUNS_SGDR_total_time = pickle.load(fil)
        fil.close()
    
        fil = open(output_pickle+'RUNS_MLPR'+extra_name,'rb')
        RUNS_MLPR_total_time = pickle.load(fil)
        fil.close()

        fil = open(output_pickle+'RUNS_RHT'+extra_name,'rb')
        RUNS_RHT_total_time = pickle.load(fil)
        fil.close()

        fil = open(output_pickle+'RUNS_RHAT'+extra_name,'rb')
        RUNS_RHAT_total_time = pickle.load(fil)
        fil.close()
        
        fil = open(output_pickle+'RUNS_MFR'+extra_name,'rb')
        RUNS_MFR_total_time = pickle.load(fil)
        fil.close()        
        
        fil = open(output_pickle+'RUNS_MTR'+extra_name,'rb')
        RUNS_MTR_total_time = pickle.load(fil)
        fil.close()
        
    return RUNS_PAR_total_time,RUNS_SGDR_total_time,RUNS_MLPR_total_time,RUNS_RHT_total_time,RUNS_RHAT_total_time,RUNS_MFR_total_time,RUNS_MTR_total_time
    
def plotting_metrics(regressors,means_R2,stds_R2,means_RMSE,stds_RMSE,means_MSE,stds_MSE,means_MAE,stds_MAE,test_samples_size,regressors_names,w):
    
    size_X=25
    size_Y=12
    colors=['b','g','r','y','m','k','c','brown','pink','gray'] 
    font_size=20
    
    #Plot R2 Evolution
#    sns.set(font_scale=2.0)

    fig=plt.figure(figsize=(size_X,size_Y))
#    plt.title('R2 evolution',size=font_size)
    plt.xlabel('Samples',size=font_size)
    plt.ylabel('R2',size=font_size)
    plt.ylim(0.4,1.0)
    plt.xlim(0,test_samples_size)
    plt.tick_params(labelsize=font_size)
    
    for r in range(len(regressors)):  
        reg_name=regressors[r].__class__.__name__  
        df_m=pd.DataFrame(means_R2[r])
        df_std=pd.DataFrame(stds_R2[r])
        plt.errorbar(range(df_m.rolling(window=w).mean().shape[0]),df_m.rolling(window=w).mean().values, yerr=df_std.rolling(window=w).mean().values,linewidth=1,errorevery=1000+((r+1)*10),label=reg_name)
  
#    plt.legend(loc='lower right', prop={'size': font_size})   
    plt.legend(prop={'size': font_size},loc='lower center',fancybox=True, shadow=True, ncol=3)#bbox_to_anchor=(0.5, -0.25)
    plt.show()
   
    #Plot RMSE Evolution
#    sns.set(font_scale=2.0)

    fig=plt.figure(figsize=(size_X,size_Y))
#    plt.title('RMSE evolution',size=font_size)
    plt.xlabel('Samples',size=font_size)
    plt.ylabel('RMSE',size=font_size)
    plt.ylim(0.0,0.2)
    plt.xlim(0,test_samples_size)
    plt.tick_params(labelsize=font_size)
    
    for r in range(len(regressors)):  
        reg_name=regressors[r].__class__.__name__  
        df_m=pd.DataFrame(means_RMSE[r])
        df_std=pd.DataFrame(stds_RMSE[r])
        plt.errorbar(range(df_m.rolling(window=w).mean().shape[0]),df_m.rolling(window=w).mean().values, yerr=df_std.rolling(window=w).mean().values,linewidth=1,errorevery=1000+((r+1)*10),label=reg_name)

#    plt.legend(loc='upper right', prop={'size': font_size})   
    plt.legend(prop={'size': font_size},loc='upper center',fancybox=True, shadow=True, ncol=3)#bbox_to_anchor=(0.5, -0.25)
    plt.show()
    
    
    #Plot MSE Evolution
#    sns.set(font_scale=2.0)

    fig=plt.figure(figsize=(size_X,size_Y))
#    plt.title('MSE evolution',size=font_size)
    plt.xlabel('Samples',size=font_size)
    plt.ylabel('MSE',size=font_size)
    plt.ylim(0.0,0.05)
    plt.xlim(0,test_samples_size)
    plt.tick_params(labelsize=font_size)
    
    for r in range(len(regressors)):  
        reg_name=regressors[r].__class__.__name__  
        df_m=pd.DataFrame(means_MSE[r])
        df_std=pd.DataFrame(stds_MSE[r])
        plt.errorbar(range(df_m.rolling(window=w).mean().shape[0]),df_m.rolling(window=w).mean().values, yerr=df_std.rolling(window=w).mean().values,linewidth=1,errorevery=1000+((r+1)*10),label=reg_name)

#    plt.legend(loc='upper right', prop={'size': font_size})   
    plt.legend(prop={'size': font_size},loc='upper center',fancybox=True, shadow=True, ncol=3)#bbox_to_anchor=(0.5, -0.25)
    plt.show()    


    #Plot MAE Evolution
#    sns.set(font_scale=2.0)

    fig=plt.figure(figsize=(size_X,size_Y))
#    plt.title('MAE evolution',size=font_size)
    plt.xlabel('Samples',size=font_size)
    plt.ylabel('MAE',size=font_size)
    plt.ylim(0.0,0.2)
    plt.xlim(0,test_samples_size)
    plt.tick_params(labelsize=font_size)
    
    for r in range(len(regressors)):  
        reg_name=regressors[r].__class__.__name__  
        df_m=pd.DataFrame(means_MAE[r])
        df_std=pd.DataFrame(stds_MAE[r])
        plt.errorbar(range(df_m.rolling(window=w).mean().shape[0]),df_m.rolling(window=w).mean().values, yerr=df_std.rolling(window=w).mean().values,linewidth=1,errorevery=1000+((r+1)*10),label=reg_name)

#    plt.legend(loc='upper right', prop={'size': font_size})   
    plt.legend(prop={'size': font_size},loc='upper center',fancybox=True, shadow=True, ncol=3)#bbox_to_anchor=(0.5, -0.25)
    plt.show()    

def feature_importance(df,coeff_threshold,target):
    
    corrs=df.corr()[target].abs()
    #keeping only columns that have correlation with target higher than threshold
    df=df.drop(corrs[corrs<coeff_threshold].index, axis=1)
    
    return df
    
def plotting_predictions(output_images,feat_sel,test_then_train_size,rp,rv,reg,samp_size,window,regressor_name):
    
    size_X=10
    size_Y=5
    colors=['b','g','r','y','m','c','pink'] 
    font_size=30

    #Plot R2 Evolution
#    sns.set(font_scale=2.0)
    
    if test_then_train_size==0.95:
        test_then_train_size=95
    elif test_then_train_size==0.8:
        test_then_train_size=80
    
    if feat_sel==True and test_then_train_size==95:

        fig=plt.figure(figsize=(size_X,size_Y))
    #    plt.title('Predictions vs Real values',size=font_size)
        plt.xlabel('Samples',size=font_size)
        plt.ylabel('PE',size=font_size)
        plt.ylim(450,460)
        plt.xlim(0,samp_size)
                
        plt.tick_params(labelsize=font_size)
        
        #Valores reales
        df_reals=pd.DataFrame(rv)
        df_pre=pd.DataFrame(rp).T
    
        plt.plot(df_reals.rolling(window=window).mean(),color='k',label='Real value',linestyle='-.')        
        plt.plot(df_pre.rolling(window=window).mean(),color=colors[reg],label=regressor_name)
    
#        plt.legend(prop={'size': font_size},loc='lower center',fancybox=True, shadow=True, ncol=8)#bbox_to_anchor=(0.5, -0.25)    
        plt.show()
        
        fig.savefig(output_images+str(regressor_name)+'_'+str(feat_sel)+'_'+str(test_then_train_size)+'.pdf', bbox_inches='tight')

    else:
        
        fig=plt.figure(figsize=(size_X,size_Y))
    #    plt.title('Predictions vs Real values',size=font_size)
        plt.xlabel('Samples',size=font_size)
#        plt.ylabel('PE',size=font_size)
        plt.ylim(450,460)
        plt.xlim(0,samp_size)
        
        plt.yticks([])    
        plt.ylabel('')    
        
        plt.tick_params(labelsize=font_size)
        
        #Valores reales
        df_reals=pd.DataFrame(rv)
        df_pre=pd.DataFrame(rp).T
    
        plt.plot(df_reals.rolling(window=window).mean(),color='k',label='Real value',linestyle='-.')        
        plt.plot(df_pre.rolling(window=window).mean(),color=colors[reg],label=regressor_name)
    
#        plt.legend(prop={'size': font_size},loc='lower center',fancybox=True, shadow=True, ncol=8)#bbox_to_anchor=(0.5, -0.25)    
        plt.show()
        
        fig.savefig(output_images+str(regressor_name)+'_'+str(feat_sel)+'_'+str(test_then_train_size)+'.pdf', bbox_inches='tight')
        
    
#==============================================================================
# MAIN
#==============================================================================

output_pickle='your_path'
output_images='your_path'
ruta='your_path'

datos='CCPP_data.csv'#CCPP_data.csv
df=pd.read_csv(ruta+datos,sep=',',header=0)

n_samples=df.shape[0]

#For HT and HAT regressors
stream = FileStream(ruta+datos)
stream.prepare_for_use() 

with warnings.catch_warnings():
    
    warnings.simplefilter("ignore")
    fxn()


    ###########################PROCESS
    runs=25
    execute=True
    hyperparameter_tuning=True
    feat_sel=True

    test_then_train_size=0.80#0.8,0.95
    preparatory_size=1-test_then_train_size
    
    mode=0
    if test_then_train_size==0.95:
        mode=95
    elif test_then_train_size==0.8:
        mode=80
    
    output_file='out_'+str(feat_sel)+'_'+str(mode)+'.csv'    
    
#    preparatory_samples_size=int(preparatory_size*n_samples)
#    test_samples_size=int(test_then_train_size*n_samples)
    
    coeff_threshold=0.65
    scoring='neg_mean_squared_error'#https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        
    df.columns=['Ambient Temperature','Vacuum','Ambient Pressure','Relative Humidity','class']
        
    if execute:

        RUNS_MSE,RUNS_RMSE,RUNS_MAE,RUNS_R2=[],[],[],[]
        RUNS_PAR_total_time,RUNS_SGDR_total_time,RUNS_MLPR_total_time,RUNS_RHT_total_time,RUNS_RHAT_total_time,RUNS_MFR_total_time,RUNS_MTR_total_time=[],[],[],[],[],[],[]
        RUNS_hyperparam_regressors=[]
        RUN_real_predicciones=[]
        RUN_real_values=[]
        
        for ru in range(runs):

            print ('-RUN='+str(ru))

            features = df[['Ambient Temperature','Vacuum','Ambient Pressure','Relative Humidity']]
            labels=df[['class']]#electrical energy output of the plant
        
            #STANDARDIZATION OF DATA     
            scaler_X = preprocessing.MinMaxScaler()
            features = scaler_X.fit_transform(features)
        
            features = pd.DataFrame(features)
        
            scaler_y = preprocessing.MinMaxScaler()
            labels = scaler_y.fit_transform(labels)
            labels = pd.DataFrame(labels)

            #REGRESSORS
            PAR=PassiveAggressiveRegressor()
            SGDR=SGDRegressor()
            MLPR=MLPRegressor()
            RHT=RegressionHoeffdingTree()
            RHAT=RegressionHAT()
            MFR=MondrianForestRegressor()
            MTR=MondrianTreeRegressor()
                        
            regressors=[PAR,SGDR,MLPR,RHT,RHAT,MFR,MTR]#7
                   
            regressors_names=[]            
            for r in range(len(regressors)):
                reg_name=regressors[r].__class__.__name__
                
                if reg_name=='PassiveAggressiveRegressor':
                    regressors_names.append('PAR')                    
                elif reg_name=='SGDRegressor':
                    regressors_names.append('SGDR')                
                elif reg_name=='MLPRegressor':
                    regressors_names.append('MLPR')
                elif reg_name=='RegressionHoeffdingTree':
                    regressors_names.append('RHT')
                elif reg_name=='RegressionHAT':
                    regressors_names.append('RHAT')
                elif reg_name=='MondrianForestRegressor':
                    regressors_names.append('MFR')
                elif reg_name=='MondrianTreeRegressor':
                    regressors_names.append('MTR')
                                    
            #TIME MEASURING            
            PAR_hyperparameter_time,SGDR_hyperparameter_time,MLPR_hyperparameter_time,RHT_hyperparameter_time,RHAT_hyperparameter_time,MFR_hyperparameter_time,MTR_hyperparameter_time=0,0,0,0,0,0,0,
            PAR_warming_time,SDGR_warming_time,MLPR_warming_time,RHT_warming_time,RHAT_warming_time,MFR_warming_time,MTR_warming_time=0,0,0,0,0,0,0
            PAR_TR_time,SGDR_TR_time,MLPR_TR_time,RHT_TR_time,RHAT_TR_time,MFR_TR_time,MTR_TR_time=0,0,0,0,0,0,0
            PAR_TS_time,SGDR_TS_time,MLPR_TS_time,RHT_TS_time,RHAT_TS_time,MFR_TS_time,MTR_TS_time=0,0,0,0,0,0,0
            PAR_total_time,SGDR_total_time,MLPR_total_time,RHT_total_time,RHAT_total_time,MFR_total_time,MTR_total_time=0,0,0,0,0,0,0
            
            #VARIABLES   
            predictions,real_predictions,r2,mae,mse,rmse=[],[],[],[],[],[]

            for reg in range(len(regressors)):
                predictions.append([])
                real_predictions.append([])                
                r2.append([])
                mae.append([])
                mse.append([])
                rmse.append([])

            ################### DATA SLICINIG
            #Shuffling
            shuff_X, shuff_y = shuffle(features, labels, random_state=ru)
            features=pd.DataFrame(shuff_X)
            features.columns=['Ambient Temperature','Vacuum','Ambient Pressure','Relative Humidity']
            labels=pd.DataFrame(shuff_y)
            
            #feature_sel/hyperparam/warming part and test-then-train part
            X_init, X_test_then_train, y_init, y_test_then_train = train_test_split(features, labels, test_size=test_then_train_size)
            
            X_test_then_train.columns=['Ambient Temperature','Vacuum','Ambient Pressure','Relative Humidity']                     
            X_init.columns=['Ambient Temperature','Vacuum','Ambient Pressure','Relative Humidity']                                 
            y_test_then_train.columns=['class']
            y_init.columns=['class']
            
            ################### FEATURE SELECTION
            if feat_sel:
                print ('FEATURE SELECTION ...')
                
                data_frame=pd.concat([X_init, y_init], axis=1)
                data_frame=feature_importance(data_frame,coeff_threshold,'class')
                
                X_test_then_train=X_test_then_train[data_frame.columns[:-1]]
                X_init=X_init[data_frame.columns[:-1]]
                
                features=features[data_frame.columns[:-1]]
                
                
            #DATA PREPARATION FOR SCIKIT-MULTIFLOW
            stream.X=features.values
            stream.y=labels.values                        
                        
            ################### HYPERPARAMETER TUNING
                                                
            if hyperparameter_tuning:
                print ('HYPERPARAMETER TUNING ...')
                
                for reg in range(len(regressors)):
                
                    reg_name=regressors[reg].__class__.__name__                                                
    
                    if reg_name=='PassiveAggressiveRegressor':
                        print (reg_name,' tuning ...')
    
                        PAR_timer=timer()            
                        PAR_grid = {'C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],'max_iter': [1]}                            
                        
                        grid_cv_PAR = RandomizedSearchCV(regressors[reg], PAR_grid, cv=10,scoring=scoring)
                        grid_cv_PAR.fit(X_init,y_init)
                        
#                        print('PAR ',scoring,'::{}'.format(grid_cv_PAR.best_score_))
#                        print('PAR Best Hyperparameters::\n{}'.format(grid_cv_PAR.best_params_))
                        
                        regressors[reg]=grid_cv_PAR.best_estimator_                         
                        PAR_hyperparameter_time=timer()-PAR_timer                

                    if reg_name=='SGDRegressor':
                        print (reg_name,' tuning ...')
        
                        SGDR_timer=timer()            
                        SGDR_grid = {
                            'alpha': 10.0 ** -np.arange(1, 7),
                            'loss': ['squared_loss', 'huber', 'epsilon_insensitive'],
                            'penalty': ['l2', 'l1', 'elasticnet'],
                            'learning_rate': ['constant', 'optimal', 'invscaling'],
                            'max_iter': [1]
                        }                            
                        
                        grid_cv_SGDR = RandomizedSearchCV(regressors[reg], SGDR_grid, cv=10,scoring=scoring)
                        grid_cv_SGDR.fit(X_init,y_init)
                        
#                        print("SGDR R-Squared::{}".format(grid_cv_SGDR.best_score_))
#                        print("SGDR Best Hyperparameters::\n{}".format(grid_cv_SGDR.best_params_))
                        
                        regressors[reg]=grid_cv_SGDR.best_estimator_
                        SGDR_hyperparameter_time=timer()-SGDR_timer

                    if reg_name=='MLPRegressor':
                        print (reg_name,' tuning ...')

                        MLPR_timer=timer()
                        MLPR_grid = {'hidden_layer_sizes': [(50, ), (100,), (500,), (50, 50), (100, 100)],
                                      'activation': ['identity', 'logistic', 'tanh', 'relu'],
                                      'solver': ['sgd','adam'],
                                      'learning_rate': ['constant','invscaling','adaptive'],
                                      'learning_rate_init': [0.0005,0.001,0.005],
                                      'alpha': 10.0 ** -np.arange(1, 10),
                                      'max_iter': [1],
                                      'batch_size': [1]
                                      }        
                
                        grid_cv_MLPR = RandomizedSearchCV(regressors[reg], MLPR_grid, cv=10,scoring=scoring)
                        grid_cv_MLPR.fit(X_init,y_init)
                        
#                        print("MLPR R-Squared::{}".format(grid_cv_MLPR.best_score_))
#                        print("MLPR Best Hyperparameters::\n{}".format(grid_cv_MLPR.best_params_))
                        
                        regressors[reg]=grid_cv_MLPR.best_estimator_                
                        MLPR_hyperparameter_time=timer()-MLPR_timer

                    if reg_name=='MondrianForestRegressor':
                        print (reg_name,' tuning ...')

                        MFR_timer=timer()
                        MFR_grid = {'n_estimators': [5,10,25,50,100],
                                      'max_depth': [None,10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
                                      'min_samples_split': [2, 5, 10]
                                      }        
                
                        grid_cv_MFR = RandomizedSearchCV(regressors[reg], MFR_grid, cv=10,scoring=scoring)
                        grid_cv_MFR.fit(X_init,y_init)
                        
#                        print("MFR R-Squared::{}".format(grid_cv_MFR.best_score_))
#                        print("MFR Best Hyperparameters::\n{}".format(grid_cv_MFR.best_params_))
                        
                        regressors[reg]=grid_cv_MFR.best_estimator_                
                        MFR_hyperparameter_time=timer()-MFR_timer

                    if reg_name=='MondrianTreeRegressor':
                        print (reg_name,' tuning ...')

                        MTR_timer=timer()
                        MTR_grid = {'max_depth': [None,10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
                                      'min_samples_split': [2, 5, 10]
                                      }        
                
                        grid_cv_MTR = RandomizedSearchCV(regressors[reg], MTR_grid, cv=10,scoring=scoring)
                        grid_cv_MTR.fit(X_init,y_init)
                        
#                        print("MTR R-Squared::{}".format(grid_cv_MTR.best_score_))
#                        print("MTR Best Hyperparameters::\n{}".format(grid_cv_MTR.best_params_))
                        
                        regressors[reg]=grid_cv_MTR.best_estimator_                
                        MTR_hyperparameter_time=timer()-MTR_timer
                        
                RUNS_hyperparam_regressors.append(regressors)                        
                                                                 
            ######################## SCIKIT-MULTIFLOW PROCESSING ########################
            print ('SCIKIT-MULTIFLOW PROCESS ...')

            predicciones=[]
            verdades=[]

            evaluator = EvaluatePrequential_ENERGIA_v2(output_file=output_pickle+output_file, show_plot=False,metrics=['mean_square_error','mean_absolute_error'],pretrain_size=X_init.shape[0],n_wait=1,predicciones=predicciones,verdades=verdades)
            _,predicciones,verdades=evaluator.evaluate(stream=stream,model=regressors,model_names=regressors_names)
            #Transformar variable para hacerla DataFrame
            new_preds=[]
            for x in range(len(predicciones)):
                p=np.array(predicciones[x]).ravel()
                new_preds.append(p)            
            
            df_predicciones=pd.DataFrame(new_preds)
            df_verdades=pd.DataFrame(verdades)
                        
            df_skmflow_metrics=pd.read_csv(output_pickle+output_file,sep=',',skiprows=[0,1,2,3,4,5],header=None)
            df_skmflow_metrics.columns=['id',
                                        'mean_mse_[PassiveAggressiveRegressor]','current_mse_[PassiveAggressiveRegressor]',
                                        'mean_mse_[SGDRegressor]','current_mse_[SGDRegressor]',
                                        'mean_mse_[MLPRegressor]','current_mse_[MLPRegressor]',
                                        'mean_mse_[RegressionHoeffdingTree]','current_mse_[RegressionHoeffdingTree]',
                                        'mean_mse_[RegressionHAT]','current_mse_[RegressionHAT]',
                                        'mean_mse_[MondrianForestRegressor]','current_mse_[MondrianForestRegressor]',
                                        'mean_mse_[MondrianTreeRegressor]','current_mse_[MondrianTreeRegressor]',
                                        'mean_mae_[PassiveAggressiveRegressor]','current_mae_[PassiveAggressiveRegressor]',
                                        'mean_mae_[SGDRegressor]','current_mae_[SGDRegressor]',
                                        'mean_mae_[MLPRegressor]','current_mae_[MLPRegressor]',
                                        'mean_mae_[RegressionHoeffdingTree]','current_mae_[RegressionHoeffdingTree]',
                                        'mean_mae_[RegressionHAT]','current_mae_[RegressionHAT]',
                                        'mean_mae_[MondrianForestRegressor]','current_mae_[MondrianForestRegressor]',
                                        'mean_mae_[MondrianTreeRegressor]','current_mae_[MondrianTreeRegressor]'
                                        ]
            df_skmflow_metrics=df_skmflow_metrics[1:]#Drop first row
            df_skmflow_metrics=df_skmflow_metrics.drop(['id'], axis=1)
            df_skmflow_metrics=df_skmflow_metrics.astype('float')
            
            #Se recogen las metricas                
            for r in range(len(regressors)):
                reg_name=regressors[r].__class__.__name__
                
                #R2
                r_squared_evolution=[]
                for ps in range(df_predicciones.shape[0]):
                    metric=r2_score(df_verdades.values[0:ps+1].ravel(),np.array(df_predicciones.iloc[:,r].values[0:ps+1]))
                    r_squared_evolution.append(metric)
                
                r2[r]=r_squared_evolution
                  
                if reg_name=='PassiveAggressiveRegressor':
                    mae[r]=(df_skmflow_metrics['current_mae_[PassiveAggressiveRegressor]'].values)
                    mse[r]=(df_skmflow_metrics['current_mse_[PassiveAggressiveRegressor]'].values)
                    rmse[r]=(np.sqrt(df_skmflow_metrics['current_mse_[PassiveAggressiveRegressor]'].values))
                          
                elif reg_name=='SGDRegressor':
                    mae[r]=(df_skmflow_metrics['current_mae_[SGDRegressor]'].values)
                    mse[r]=(df_skmflow_metrics['current_mse_[SGDRegressor]'].values)
                    rmse[r]=(np.sqrt(df_skmflow_metrics['current_mse_[SGDRegressor]'].values))
                    
                elif reg_name=='MLPRegressor':
                    mae[r]=(df_skmflow_metrics['current_mae_[MLPRegressor]'].values)
                    mse[r]=(df_skmflow_metrics['current_mse_[MLPRegressor]'].values)
                    rmse[r]=(np.sqrt(df_skmflow_metrics['current_mse_[MLPRegressor]'].values))                                        
                    
                elif reg_name=='RegressionHoeffdingTree':
                    mae[r]=(df_skmflow_metrics['current_mae_[RegressionHoeffdingTree]'].values)
                    mse[r]=(df_skmflow_metrics['current_mse_[RegressionHoeffdingTree]'].values)
                    rmse[r]=(np.sqrt(df_skmflow_metrics['current_mse_[RegressionHoeffdingTree]'].values))
                    
                elif reg_name=='RegressionHAT':
                    mae[r]=(df_skmflow_metrics['current_mae_[RegressionHAT]'].values)
                    mse[r]=(df_skmflow_metrics['current_mse_[RegressionHAT]'].values)
                    rmse[r]=(np.sqrt(df_skmflow_metrics['current_mse_[RegressionHAT]'].values))

                elif reg_name=='MondrianForestRegressor':
                    mae[r]=(df_skmflow_metrics['current_mae_[MondrianForestRegressor]'].values)
                    mse[r]=(df_skmflow_metrics['current_mse_[MondrianForestRegressor]'].values)
                    rmse[r]=(np.sqrt(df_skmflow_metrics['current_mse_[MondrianForestRegressor]'].values))
                 
                elif reg_name=='MondrianTreeRegressor':
                    mae[r]=(df_skmflow_metrics['current_mae_[MondrianTreeRegressor]'].values)
                    mse[r]=(df_skmflow_metrics['current_mse_[MondrianTreeRegressor]'].values)
                    rmse[r]=(np.sqrt(df_skmflow_metrics['current_mse_[MondrianTreeRegressor]'].values))

            #Se recogen las tiempos
            training_t_skmflow=0
            testing_t_skmflow=0
            total_t_skmflow=0

            for r in range(len(regressors)):                    
                reg_name=regressors[r].__class__.__name__

                if reg_name=='PassiveAggressiveRegressor':
                    PAR_TR_time=evaluator.running_time_measurements[0]._training_time
                    PAR_TS_time=evaluator.running_time_measurements[0]._testing_time
                    PAR_total_time=evaluator.running_time_measurements[0].get_current_total_running_time()
                                            
                if reg_name=='SGDRegressor':
                    SGDR_TR_time=evaluator.running_time_measurements[1]._training_time
                    SGDR_TS_time=evaluator.running_time_measurements[1]._testing_time
                    SGDR_total_time=evaluator.running_time_measurements[1].get_current_total_running_time()
                    
                if reg_name=='MLPRegressor':
                    MLPR_TR_time=evaluator.running_time_measurements[2]._training_time
                    MLPR_TS_time=evaluator.running_time_measurements[2]._testing_time
                    MLPR_total_time=evaluator.running_time_measurements[2].get_current_total_running_time()
                                        
                if reg_name=='RegressionHoeffdingTree':
                    RHT_TR_time=evaluator.running_time_measurements[3]._training_time
                    RHT_TS_time=evaluator.running_time_measurements[3]._testing_time
                    RHT_total_time=evaluator.running_time_measurements[3].get_current_total_running_time()
            
                if reg_name=='RegressionHAT':
                    RHAT_TR_time=evaluator.running_time_measurements[4]._training_time
                    RHAT_TS_time=evaluator.running_time_measurements[4]._testing_time
                    RHAT_total_time=evaluator.running_time_measurements[4].get_current_total_running_time()
            
                if reg_name=='MondrianForestRegressor':
                    MFR_TR_time=evaluator.running_time_measurements[5]._training_time
                    MFR_TS_time=evaluator.running_time_measurements[5]._testing_time
                    MFR_total_time=evaluator.running_time_measurements[5].get_current_total_running_time()
            
                if reg_name=='MondrianTreeRegressor':
                    MTR_TR_time=evaluator.running_time_measurements[6]._training_time
                    MTR_TS_time=evaluator.running_time_measurements[6]._testing_time
                    MTR_total_time=evaluator.running_time_measurements[6].get_current_total_running_time()

            RUNS_MSE.append(mse)                        
            RUNS_RMSE.append(rmse)                
            RUNS_MAE.append(mae)    
            RUNS_R2.append(r2)            
        
            RUNS_PAR_total_time.append(PAR_total_time)
            RUNS_SGDR_total_time.append(SGDR_total_time)
            RUNS_MLPR_total_time.append(MLPR_total_time)
            RUNS_RHT_total_time.append(RHT_total_time)
            RUNS_RHAT_total_time.append(RHAT_total_time)
            RUNS_MFR_total_time.append(MFR_total_time)
            RUNS_MTR_total_time.append(MTR_total_time)

            #Transformar el escalado antes de las predicciones antes de plotearlas
            real_predictions=[]
            for r in range(len(regressors)):  
                preds=scaler_y.inverse_transform([df_predicciones.iloc[:,r].values])
                real_predictions.append(preds)

            real_values=scaler_y.inverse_transform(df_verdades.values)
            
            RUN_real_predicciones.append(real_predictions)
            RUN_real_values.append(real_values)

                 
####################################### SAVING RESULTS
savingResults(output_pickle,RUNS_MSE,RUNS_MAE,RUNS_RMSE,RUNS_R2,feat_sel,RUNS_hyperparam_regressors,RUN_real_predicciones,RUN_real_values,preparatory_size)

####################################### SAVING TIME
savingTime(output_pickle,RUNS_PAR_total_time,RUNS_SGDR_total_time,RUNS_MLPR_total_time,RUNS_RHT_total_time,RUNS_RHAT_total_time,RUNS_MFR_total_time,RUNS_MTR_total_time,feat_sel,preparatory_size)

####################################### LOADING RESULTS
RUNS_MSE,RUNS_MAE,RUNS_RMSE,RUNS_R2,RUNS_hyperparam_regressors,RUN_real_predicciones,RUN_real_values=loadingResults(output_pickle,feat_sel,preparatory_size)

####################################### LOADING TIME
RUNS_PAR_total_time,RUNS_SGDR_total_time,RUNS_MLPR_total_time,RUNS_RHT_total_time,RUNS_RHAT_total_time,RUNS_MFR_total_time,RUNS_MTR_total_time=loadingTime(output_pickle,feat_sel,preparatory_size)

####################################### METRICS SUMMARY

mean_MSE=np.round(np.mean(np.array(RUNS_MSE),axis=0),3)
std_MSE=np.round(np.std(np.array(RUNS_MSE),axis=0),3)

mean_MAE=np.round(np.mean(np.array(RUNS_MAE),axis=0),3)
std_MAE=np.round(np.std(np.array(RUNS_MAE),axis=0),3)

mean_RMSE=np.round(np.mean(np.array(RUNS_RMSE),axis=0),3)
std_RMSE=np.round(np.std(np.array(RUNS_RMSE),axis=0),3)

mean_R2=np.round(np.mean(np.array(RUNS_R2),axis=0),3)
std_R2=np.round(np.std(np.array(RUNS_R2),axis=0),3)

for r in range(len(regressors)):  
    reg_name=regressors[r].__class__.__name__
    print ('------- ',reg_name,' -------')
            
    print ('MSE runs average: ', np.round(np.mean(mean_MSE[r][2:]),3), ' +- ',np.round(np.mean(std_MSE[r][2:]),3))
    print ('RMSE runs average: ', np.round(np.mean(mean_RMSE[r][2:]),3), ' +- ',np.round(np.mean(std_RMSE[r][2:]),3))
    print ('MAE runs average: ', np.round(np.mean(mean_MAE[r][2:]),3), ' +- ',np.round(np.mean(std_MAE[r][2:]),3))
    print ('R2 runs average: ', np.round(np.mean(mean_R2[r][2:]),3), ' +- ',np.round(np.mean(std_R2[r][2:]),3))


####################################### TIMES SUMMARY
print ('------------------------------------------------------------------------')
print ('PassiveAggressiveRegressor time: ',np.round(np.mean(np.array(RUNS_PAR_total_time),axis=0),3),' +- ',np.round(np.std(np.array(RUNS_PAR_total_time),axis=0),3))
print ('SGDRegressor time: ',np.round(np.mean(np.array(RUNS_SGDR_total_time),axis=0),3),' +- ',np.round(np.std(np.array(RUNS_SGDR_total_time),axis=0),3))
print ('MLPRegressor time: ',np.round(np.mean(np.array(RUNS_MLPR_total_time),axis=0),3),' +- ',np.round(np.std(np.array(RUNS_MLPR_total_time),axis=0),3))
print ('RegressionHoeffdingTree time: ',np.round(np.mean(np.array(RUNS_RHT_total_time),axis=0),3),' +- ',np.round(np.std(np.array(RUNS_RHT_total_time),axis=0),3))
print ('RegressionHAT time: ',np.round(np.mean(np.array(RUNS_RHAT_total_time),axis=0),3),' +- ',np.round(np.std(np.array(RUNS_RHAT_total_time),axis=0),3))
print ('MondrianForestRegressor time: ',np.round(np.mean(np.array(RUNS_MFR_total_time),axis=0),3),' +- ',np.round(np.std(np.array(RUNS_MFR_total_time),axis=0),3))
print ('MondrianTreeRegressor time: ',np.round(np.mean(np.array(RUNS_MTR_total_time),axis=0),3),' +- ',np.round(np.std(np.array(RUNS_MTR_total_time),axis=0),3))

####################################### PLOTS
#w=300
#plotting_metrics(regressors,mean_R2,std_R2,mean_RMSE,std_RMSE,mean_MSE,std_MSE,mean_MAE,std_MAE,X_test_then_train.shape[0],regressors_names,w)

#run_for_plot=0
w=750
run_plot=1
mean_real_predicciones=np.array(RUN_real_predicciones[run_plot])
mean_real_values=np.array(RUN_real_values[run_plot])

for r in range(len(regressors)):  
    plotting_predictions(output_images,feat_sel,test_then_train_size,mean_real_predicciones[r],mean_real_values,r,len(mean_real_values),w,regressors_names[r])    


    
            
