import argparse
import datetime
from util.create_dataset import create_dataset_multiple_cultivars
import torch

import os
import pickle
import glob
import pandas as pd
import gc
from pathlib import Path

from util.create_dataset import MyDataset #REMOVE
from torch.utils.data import DataLoader

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Running all experiments from here, preventing code duplication")
    parser.add_argument('--experiment', type=str, default="multiplicative_embedding", choices=['multiplicative_embedding', 'mtl', 'additive_embedding', 'concat_embedding', 'single', 'ferguson', 'pheno_embedding', 'basic_cultivar', 'encode_cultivar', 'encode_cultivar_2', 'concat_embedding_vectors'], help='type of the experiment')
    #arg for freeze/unfreeze, all/leaveoneout, afterL1, L2,etc, linear/non linear embedding, scratch/linear combination for finetune, task weighting
    parser.add_argument('--setting', type=str, default="all", choices=['all','leaveoneout','allfinetune','embed','oracle','rand_embeds','embed_lin','baseline_avg','baseline_wavg','baseline_all','baseline_each','test_pheno'], help='experiment setting')
    parser.add_argument('--variant', type=str, default='none',choices=['none','afterL1','afterL2','afterL3','afterL4'])
    parser.add_argument('--unfreeze', type=str, default='no', choices=['yes','no'], help="unfreeze weights during finetune")
    #todo
    parser.add_argument('--nonlinear', type=str, default='no', choices=['yes','no'],help='try non linear embedding/prediction head')
    #todo
    parser.add_argument('--scratch', type=str, default='no', choices=['yes','no'],help='try learning embedding from scratch')
    parser.add_argument('--weighting', type=str, default='none', choices=['none', 'inverse_freq', 'uncertainty'],
                        help="loss weighting strategy")
    parser.add_argument('--name', type=str, default=datetime.datetime.now(
    ).strftime("%d_%b_%Y_%H_%M_%S"), help='name of the experiment')
    parser.add_argument('--epochs', type=int, default=400,
                        help='No of epochs to run the model for')
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning Rate")
    parser.add_argument('--lr_emb', type=float, default=1e-4, help="Learning Rate for embedding")
    parser.add_argument('--no_seasons', type=int, default=-1, help="no of seasons to select for the Riesling Cultivar")
    parser.add_argument('--batch_size', type=int,
                        default=12, help="Batch size")
    parser.add_argument('--evalpath', type=int,
                        default=None, help="Evaluation Path")
    parser.add_argument('--data_path', type=str,
                        default='./data/valid/', help="csv Path")
    parser.add_argument('--pretrained_path', type=str, default=None,
                        help="pretrained model to load for finetuning")
    parser.add_argument('--embedding_path', type=str, default=None,
                        help="pretrained embedding model to load for cultivar prediction ground truth")
    parser.add_argument('--specific_cultivar', type=str, default=None,
                        help="specific cultivar to train for")
    parser.add_argument('--include_lte', action='store_true',
                        help="include lte loss in training") #If you don't include this flag, won't use LTE!!!
    parser.add_argument('--include_pheno', action='store_true',
                        help="include pheno loss in training") #If you don't include this flag, won't use pheno!!!
    parser.add_argument('--phenos', type=str, default=None,
                        help="comma seperated list of phenological events to predict, replace spaces with _")
    parser.add_argument('--train_cultivars', type=str, default=None,
                        help="comma seperated list of training cultivars, replace spaces with _, order matters, must include whenever model does not use default cultivars")
    parser.add_argument('--test_cultivars', type=str, default=None,
                        help="comma seperated list of test cultivars to predict, replace spaces with _")
    parser.add_argument('--train_embeds_LTE', action='store_true',
                        help="train embeds with LTE instead of pheno")
    parser.add_argument('--a_weight', type=str, default=None,
                        help="constant used in baseline weighted average") 
    parser.add_argument('--extra_embeds', action='store_true',
                        help="consider extra cultivars")    
    parser.add_argument('--exclude_source', action='store_true',
                        help="dont use original cultivar embeddings")                         
    args = parser.parse_args()
    valid_cultivars = ['Zinfandel',
                       #'Cabernet Franc',
                       'Concord',
                       'Malbec',
                       'Barbera',
                       'Semillon',
                       'Merlot',
                       #'Lemberger',
                       'Chenin Blanc',
                       'Riesling',
                       'Nebbiolo',
                        'Cabernet Sauvignon',
                       'Chardonnay',
                       'Viognier',
                       #'Gewurztraminer',
                       'Mourvedre',
                       'Pinot Gris',
                       'Grenache',
                       'Syrah',
                       'Sangiovese',
                       'Sauvignon Blanc']
    if args.train_cultivars != None:
        args.valid_cultivars = args.train_cultivars.replace('_',' ').split(',')
        valid_cultivars = args.valid_cultivars #both are used throughout the code, make sure both are the same
    else:
        args.valid_cultivars = valid_cultivars
    #args.valid_cultivars = valid_cultivars
    args.bb_day_diff = {cultivar:list() for cultivar in args.valid_cultivars}
    if args.phenos != None:
        args.phenos_list = args.phenos.replace('_',' ').split(',')
    else:
        args.phenos_list = []
        
    if args.test_cultivars != None:
        args.formatted_test_cultivars = args.test_cultivars.replace('_',' ').split(',')    
    else:
        args.formatted_test_cultivars = valid_cultivars
    args.cultivar_file_dict = {cultivar: pd.read_csv(
    glob.glob(args.data_path+'*'+cultivar+'*')[0]) for cultivar in valid_cultivars}
    args.features = [
        # 'DATE', # date of weather observation
        # 'AWN_STATION', # closest AWN station
        # 'SEASON',
        # 'SEASON_JDAY',
        # 'DORMANT_SEASON',
        # 'YEAR_JDAY',
        # 'PHENOLOGY',
        # 'PREDICTED_LTE50',
        # 'PREDICTED_Budbreak',
        # mean temperature is the calculation of (max_f+min_f)/2 and then converted to Celsius. # they use this one
        'MEAN_AT',
        'MIN_AT',  # a
        # 'AVG_AT', # average temp is AgWeather Network
        'MAX_AT',  # a
        'MIN_RH',  # a
        'AVG_RH',  # a
        'MAX_RH',  # a
        'MIN_DEWPT',  # a
        'AVG_DEWPT',  # a
        'MAX_DEWPT',  # a
        'P_INCHES',  # precipitation # a
        'WS_MPH',  # wind speed. if no sensor then value will be na # a
        'MAX_WS_MPH',  # a
        # 'WD_DEGREE', # wind direction, if no sensor then value will be na
        # 'LW_UNITY', # leaf wetness sensor
        # 'SR_WM2', # solar radiation
        # 'MIN_ST2',
        # 'ST2',
        # 'MAX_ST2',
        # 'MIN_ST8',
        # 'ST8', # soil temperature
        # 'MAX_ST8',
        # 'SM8_PCNT', # soil moisture import matplotlib.pyplot as plt@ 8-inch depth # too many missing values for merlot
        # 'SWP8_KPA', # stem water potential @ 8-inch depth # too many missing values for merlot
        # 'MSLP_HPA', # barrometric pressure
        # 'ETO', # evaporation of soil water lost to atmosphere
        # 'ETR' # ???
    ]
    args.ferguson_features = ['PREDICTED_LTE10',
                              'PREDICTED_LTE50', 'PREDICTED_LTE90']
    args.label = ['LTE10', 'LTE50', 'LTE90']
    args.device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    from experiments.unified_api import run_experiment
    from experiments.unified_api import run_embed_train 
    from experiments.unified_api import run_embed_eval
    from experiments.unified_api import run_cultivar_train
    from experiments.unified_api import run_cultivar_eval
    from experiments.unified_api import run_state_vectors_eval
    #from experiments.unified_api import run_eval as run_experiment
    print(args.experiment, "experiment")
    if args.experiment=='ferguson':
        pass
    else:
        if args.variant=='none':
            exec("from nn.models import "+args.experiment+"_net as nn_model")
            exec("from nn.models import "+args.experiment+"_net_finetune as nn_model_finetune")
        else:
            exec("from nn.models import "+args.experiment+"_net_"+args.variant+" as nn_model")
            exec("from nn.models import "+args.experiment+"_net_finetune_"+args.variant+" as nn_model_finetune")


    overall_loss = dict()
    #single model training
    if args.experiment in ['single']:
        
        if args.setting in ['baseline_all']:
            args.nn_model = nn_model
            for left_out in valid_cultivars:
                gc.collect()
                loss_dicts = dict()
                other_cultivars = list(set(valid_cultivars) - set([left_out]))
                args.current_cultivar = left_out
                args.no_of_cultivars = len(other_cultivars)
                # get dataset by selecting features
                args.cultivar_list = list(other_cultivars)
                for trial in range(3):
                    args.trial = 'trial_'+str(trial)
                    #train model on everything else
                    args.valid_cultivars = other_cultivars
                    args.dataset = create_dataset_multiple_cultivars(args)
                    _ = run_experiment(args)
                    
                    #eval model on new season
                    args.valid_cultivars = list(left_out)
                    args.dataset = create_dataset_multiple_cultivars(args)
                    loss_dicts[args.trial] = run_eval(args)
                overall_loss[left_out] = loss_dicts
        else:
            valid_cultivars = valid_cultivars if args.specific_cultivar is None else list([
                                                                                          args.specific_cultivar])
            print("valid_cultivars", valid_cultivars)
            args.nn_model = nn_model
            for left_out in valid_cultivars:
                gc.collect()
                loss_dicts = dict()
                other_cultivars = list([left_out])
                args.current_cultivar = left_out
                args.no_of_cultivars = len(other_cultivars)
                # get dataset by selecting features
                args.cultivar_list = list(other_cultivars)
                for trial in range(3):
                    args.trial = 'trial_'+str(trial)
                    args.dataset = create_dataset_multiple_cultivars(args)
                    loss_dicts[args.trial] = run_experiment(args)
                overall_loss[left_out] = loss_dicts
            
    #ferguson model evaluation
    elif args.experiment in ['ferguson']:
        from util.data_processing import evaluate_ferguson
        for left_out in valid_cultivars:
            gc.collect()
            loss_dicts = dict()
            other_cultivars = list([left_out])
            args.current_cultivar = left_out
            args.no_of_cultivars = len(other_cultivars)
            # get dataset by selecting features
            args.cultivar_list = list(other_cultivars)
            for trial in range(3): #CHNAGE
                args.trial = 'trial_'+str(trial)
                args.dataset = create_dataset_multiple_cultivars(args)
                #change back:
                #train_dataset = MyDataset(args.dataset['train'])
                #trainLoader = DataLoader(train_dataset, batch_size=1, shuffle=True)
                #print(left_out)
                #for i, (x, y, cultivar_id, freq) in enumerate(trainLoader):
                #    print(i)
                loss_dicts[args.trial] = evaluate_ferguson(args)
            overall_loss[left_out] = loss_dicts
    elif args.experiment in ['basic_cultivar', 'encode_cultivar','encode_cultivar_2']:
        args.nn_model = nn_model
        args.nn_model_finetune = nn_model_finetune
        for left_out_idx, left_out in enumerate(valid_cultivars):#['Zinfandel']):#valid_cultivars):
            gc.collect()
            loss_dicts = dict()
            finetune_loss_dicts = dict()
            
            other_cultivars = list(set(valid_cultivars) - set([left_out]))
            args.current_cultivar = left_out
            #args.no_of_cultivars = len(other_cultivars)
            # get dataset by selecting features
            #args.cultivar_list = list(other_cultivars)
            
            # similar for all experiments
            for trial in range(3):
                #train model on all but one
                args.no_of_cultivars = len(other_cultivars)
                args.cultivar_list = list(other_cultivars)
            
                args.trial = 'trial_'+str(trial)
                args.dataset = create_dataset_multiple_cultivars(args)
                
                #loss_dicts[args.trial+'_train'] = run_cultivar_train(args, left_out_idx=left_out_idx)
                
                #test on held out cultivar
                args.cultivar_list = list([left_out])
                
                args.pretrained_path = os.path.join(
                    './models', args.name, args.current_cultivar, args.trial, args.experiment+'_setting_'+args.setting+'_variant_'+args.variant+'_weighting_'+args.weighting+'_unfreeze_'+args.unfreeze+'_nonlinear_'+args.nonlinear+'_scratch_'+args.scratch+".pt")
                
                args.dataset = create_dataset_multiple_cultivars(args)
                
                loss_dicts[args.trial+'_test'] = run_cultivar_eval(args, left_out_idx=left_out_idx, finetune=False)
             
            overall_loss[left_out] = loss_dicts
        
        
        
    elif args.setting in ['baseline_avg','baseline_wavg']:
        #from nn.models import pheno_embedding_net_finetune_emb as nn_model_finetune
        #from nn.models import pheno_embedding_net as nn_model
        #from nn.models import pheno_embedding_net_finetune as nn_model_finetune
            
        loss_dicts = dict()
        finetune_loss_dicts = dict()
        args.cultivar_list = list(valid_cultivars)
        #other_cultivars = list(set(valid_cultivars) - set([left_out]))
        args.current_cultivar = 'all'
        #args.batch_size = 1
        args.no_of_cultivars = len(valid_cultivars)-1
        args.nn_model = nn_model
        args.nn_model_finetune = nn_model_finetune
       
        
        # get dataset by selecting features
        for idx, left_out in enumerate(valid_cultivars):
            args.cultivar_list = list([left_out])
            args.trial = 'trial_'+str(0)
            args.dataset = create_dataset_multiple_cultivars(args)
            args.current_cultivar = left_out
            args.pretrained_path = os.path.join(
                #'./models', 'embed_oracle', args.current_cultivar, args.trial, 'pheno_embedding_setting_oracle_variant_'+args.variant+'_weighting_'+args.weighting+'_unfreeze_'+args.unfreeze+'_nonlinear_'+args.nonlinear+'_scratch_'+args.scratch+".pt")
                #'./models', 'experiment051723', args.current_cultivar, 'trial_0', args.experiment+'_setting_embed_variant_'+args.variant+'_weighting_'+args.weighting+'_unfreeze_'+args.unfreeze+'_nonlinear_'+args.nonlinear+'_scratch_'+args.scratch+".pt")
                './models', 'emb_models063023', args.current_cultivar, 'trial_0', args.experiment+'_setting_leaveoneout_variant_'+args.variant+'_weighting_'+args.weighting+'_unfreeze_'+args.unfreeze+'_nonlinear_'+args.nonlinear+'_scratch_'+args.scratch+".pt")
                #'./models', 'mtl_models063023', args.current_cultivar, 'trial_0', args.experiment+'_setting_leaveoneout_variant_'+args.variant+'_weighting_'+args.weighting+'_unfreeze_'+args.unfreeze+'_nonlinear_'+args.nonlinear+'_scratch_'+args.scratch+".pt")
            print(args.pretrained_path)                    
                # similar for all experiments
            #args.current_cultivar = left_out #change for tensorboard
            loss_dicts[left_out] = run_embed_eval(args, finetune=False)
        
        overall_loss = loss_dicts
        
    elif args.setting in ['baseline_each']:
        loss_dicts = dict()
        finetune_loss_dicts = dict()
        args.cultivar_list = list(valid_cultivars)
        args.current_cultivar = 'all'
        args.no_of_cultivars = len(valid_cultivars)-1
        args.nn_model = nn_model
        args.nn_model_finetune = nn_model_finetune
       
        # get dataset by selecting features
        for idx, left_out in enumerate(valid_cultivars):
            print("Held out: ",left_out)
            #args.cultivar_list = list([left_out])
            args.trial = 'trial_'+str(0)
            args.dataset = create_dataset_multiple_cultivars(args)
            args.current_cultivar = left_out
            args.pretrained_path = os.path.join(
                #'./models', 'embed_oracle', args.current_cultivar, args.trial, 'pheno_embedding_setting_oracle_variant_'+args.variant+'_weighting_'+args.weighting+'_unfreeze_'+args.unfreeze+'_nonlinear_'+args.nonlinear+'_scratch_'+args.scratch+".pt")
                #'./models', 'experiment051723', args.current_cultivar, 'trial_0', args.experiment+'_setting_embed_variant_'+args.variant+'_weighting_'+args.weighting+'_unfreeze_'+args.unfreeze+'_nonlinear_'+args.nonlinear+'_scratch_'+args.scratch+".pt")
                './models', 'emb_models063023', args.current_cultivar, 'trial_0', args.experiment+'_setting_leaveoneout_variant_'+args.variant+'_weighting_'+args.weighting+'_unfreeze_'+args.unfreeze+'_nonlinear_'+args.nonlinear+'_scratch_'+args.scratch+".pt")
                #'./models', 'mtl_models063023', args.current_cultivar, 'trial_0', args.experiment+'_setting_leaveoneout_variant_'+args.variant+'_weighting_'+args.weighting+'_unfreeze_'+args.unfreeze+'_nonlinear_'+args.nonlinear+'_scratch_'+args.scratch+".pt")
            print(args.pretrained_path)                    
                # similar for all experiments
            #args.current_cultivar = left_out #change for tensorboard
            loss_dicts[left_out] = run_embed_eval(args, finetune=False)
        
        overall_loss = loss_dicts
    elif args.experiment in ['concat_embedding_vectors']:
        #train
        overall_loss = dict()
        loss_dicts = dict()
        finetune_loss_dicts = dict()
        
        if args.pretrained_path == None:
            args.current_cultivar = 'all'
            args.no_of_cultivars = len(valid_cultivars)
            # get dataset by selecting features
            args.cultivar_list = list(valid_cultivars)
            args.nn_model = nn_model
            # similar for all experiments
            for trial in range(3):
                args.trial = 'trial_'+str(trial)
                args.dataset = create_dataset_multiple_cultivars(args)
                args.pretrained_path = os.path.join(
                    './models', args.name, 'all', args.trial, args.experiment+'_setting_'+args.setting+'_variant_'+args.variant+'_weighting_'+args.weighting+'_unfreeze_'+args.unfreeze+'_nonlinear_'+args.nonlinear+'_scratch_'+args.scratch+".pt")
                loss_dicts[args.trial] = run_experiment(args)
            overall_loss[args.experiment] = loss_dicts
        
        #get all vectors
        
        overall_rnn_vectors = dict()
        overall_penul_vectors = dict()
        overall_LTE_dict = dict()
        args.nn_model = nn_model
        for left_out in args.formatted_test_cultivars:#enumerate(valid_cultivars):
            c_id = args.valid_cultivars.index(left_out)
            gc.collect()
            loss_dicts = dict()
            rnn_vectors = dict()
            penul_vectors = dict()
            LTE_dict = dict()
            other_cultivars = list([left_out])
            args.current_cultivar = left_out
            args.no_of_cultivars = len(valid_cultivars)
            # get dataset by selecting features
            args.cultivar_list = list([left_out])
            for trial in range(3):
                print(left_out, " Trial ", trial)
                args.trial = 'trial_'+str(trial)
                args.dataset = create_dataset_multiple_cultivars(args)
                if args.pretrained_path == None:
                    args.pretrained_path = os.path.join(
                    './models', args.name, 'all', args.trial, args.experiment+'_setting_'+args.setting+'_variant_'+args.variant+'_weighting_'+args.weighting+'_unfreeze_'+args.unfreeze+'_nonlinear_'+args.nonlinear+'_scratch_'+args.scratch+".pt")
                rnn_vectors[args.trial], penul_vectors[args.trial], LTE_dict[args.trial] = run_state_vectors_eval(args, c_id)
            overall_rnn_vectors[left_out] = rnn_vectors
            overall_penul_vectors[left_out] = penul_vectors
            overall_LTE_dict[left_out] = LTE_dict
            
            #with open(os.path.join('./models', args.name, args.experiment+'_setting_'+args.setting+'_variant_'+args.variant+'_weighting_'+args.weighting+'_unfreeze_'+args.unfreeze+'_nonlinear_'+args.nonlinear+'_scratch_'+args.scratch+"_"+left_out+"_rnn_vectors.pkl"), 'wb') as f:
            #    pickle.dump(rnn_vectors, f)
            #with open(os.path.join('./models', args.name, args.experiment+'_setting_'+args.setting+'_variant_'+args.variant+'_weighting_'+args.weighting+'_unfreeze_'+args.unfreeze+'_nonlinear_'+args.nonlinear+'_scratch_'+args.scratch+"_"+left_out+"_penul_vectors.pkl"), 'wb') as f:
            #    pickle.dump(penul_vectors, f)
        
            
        
    else:
        #leave one out setting
        if args.setting in ['leaveoneout']:
            args.nn_model = nn_model
            args.nn_model_finetune = nn_model_finetune
            for left_out in valid_cultivars:
                gc.collect()
                loss_dicts = dict()
                finetune_loss_dicts = dict()
                other_cultivars = list(set(valid_cultivars) - set([left_out]))
                args.current_cultivar = left_out
                args.no_of_cultivars = len(other_cultivars)
                # get dataset by selecting features
                args.cultivar_list = list(other_cultivars)
                # similar for all experiments
                for trial in range(3):
                    args.trial = 'trial_'+str(trial)
                    args.dataset = create_dataset_multiple_cultivars(args)
                    loss_dicts[args.trial] = run_experiment(args)
                if (False): #don't want to finetune
                    # get dataset by selecting features
                    args.cultivar_list = list([left_out])
                    for trial in range(3):
                        args.trial = 'trial_'+str(trial)
                        args.dataset = create_dataset_multiple_cultivars(args)
                        args.pretrained_path = os.path.join(
                            './models', args.name, args.current_cultivar, args.trial, args.experiment+'_setting_'+args.setting+'_variant_'+args.variant+'_weighting_'+args.weighting+'_unfreeze_'+args.unfreeze+'_nonlinear_'+args.nonlinear+'_scratch_'+args.scratch+".pt")
                        # similar for all experiments
                        finetune_loss_dicts[args.trial] = run_experiment(args, finetune=True)
                    for trial in range(3):
                        args.trial = 'trial_'+str(trial)
                        loss_dicts[args.trial].update(finetune_loss_dicts[args.trial])
                overall_loss[left_out] = loss_dicts
        #embed train setting
        elif args.setting in ['embed']:
            from nn.models import pheno_embedding_net_finetune_emb as nn_model_finetune
            use_pretrained = args.pretrained_path
            for left_out in valid_cultivars:
                loss_dicts = dict()
                other_cultivars = list(set(valid_cultivars) - set([left_out]))
                args.current_cultivar = left_out
                args.no_of_cultivars = len(other_cultivars)
                args.nn_model = nn_model
                args.nn_model_finetune = nn_model_finetune
                # get dataset by selecting features
                args.cultivar_list = list(other_cultivars)
               
                if(use_pretrained is None):
                    # similar for all experiments
                    args.trial = 'trial_'+str(0)
                    args.dataset = create_dataset_multiple_cultivars(args)
                    loss_dicts['base_model'], embeds = run_experiment(args)
                
                # get dataset by selecting features
                args.cultivar_list = list([left_out]) 
                for i in range(3):
                    args.trial = 'trial_'+str(i)
                    args.dataset = create_dataset_multiple_cultivars(args)
                    args.pretrained_path = os.path.join(
                        './models', 'emb_models063023', args.current_cultivar, 'trial_0', args.experiment+'_setting_leaveoneout_variant_'+args.variant+'_weighting_'+args.weighting+'_unfreeze_'+args.unfreeze+'_nonlinear_'+args.nonlinear+'_scratch_'+args.scratch+".pt")
                        # similar for all experiments
                    
                    loss_dicts[args.trial] = run_embed_train(args)
                
                overall_loss[left_out] = loss_dicts
        #embed linear train setting
        elif args.setting in ['embed_lin']:
            from nn.models import pheno_embedding_net_finetune as nn_model_finetune
            use_pretrained = args.pretrained_path
            for left_out in valid_cultivars:
                loss_dicts = dict()
                other_cultivars = list(set(valid_cultivars) - set([left_out]))
                args.current_cultivar = left_out
                args.no_of_cultivars = len(other_cultivars)
                args.nn_model = nn_model
                args.nn_model_finetune = nn_model_finetune
                # get dataset by selecting features
                args.cultivar_list = list(other_cultivars)
               
                if(use_pretrained is None):
                    # similar for all experiments
                    args.trial = 'trial_'+str(0)
                    args.dataset = create_dataset_multiple_cultivars(args)
                    loss_dicts['base_model'], embeds = run_experiment(args)
                
                # get dataset by selecting features
                args.cultivar_list = list([left_out]) 
                for i in range(3):
                    args.trial = 'trial_'+str(i)
                    args.dataset = create_dataset_multiple_cultivars(args)
                    args.pretrained_path = os.path.join(
                        './models', 'emb_models063023', args.current_cultivar, 'trial_0', args.experiment+'_setting_leaveoneout_variant_'+args.variant+'_weighting_'+args.weighting+'_unfreeze_'+args.unfreeze+'_nonlinear_'+args.nonlinear+'_scratch_'+args.scratch+".pt")
                        # similar for all experiments
                    
                    loss_dicts[args.trial] = run_embed_train(args)
                
                overall_loss[left_out] = loss_dicts
        #oracle test
        elif args.setting in ['oracle']:
            from nn.models import pheno_embedding_net_finetune_emb as nn_model_finetune
            
            loss_dicts = dict()
            finetune_loss_dicts = dict()
            args.cultivar_list = list(valid_cultivars)
            #other_cultivars = list(set(valid_cultivars) - set([left_out]))
            args.current_cultivar = 'all'
            args.no_of_cultivars = len(valid_cultivars)
            args.nn_model = nn_model
            args.nn_model_finetune = nn_model_finetune
            # get dataset by selecting features
            #args.cultivar_list = list(other_cultivars)
           
            # similar for all experiments
            args.trial = 'trial_'+str(0)
            args.dataset = create_dataset_multiple_cultivars(args)
            loss_dicts[args.trial], embeds = run_experiment(args)
            print(loss_dicts[args.trial])
            
            # get dataset by selecting features
            for idx, left_out in enumerate(valid_cultivars):
                print("Training embed for", left_out)
                args.cultivar_list = list([left_out])
                args.trial = 'trial_'+str(0)
                args.dataset = create_dataset_multiple_cultivars(args)
                args.current_cultivar = 'all'
                args.pretrained_path = os.path.join(
                    './models', args.name, args.current_cultivar, args.trial, args.experiment+'_setting_'+args.setting+'_variant_'+args.variant+'_weighting_'+args.weighting+'_unfreeze_'+args.unfreeze+'_nonlinear_'+args.nonlinear+'_scratch_'+args.scratch+".pt")
                    # similar for all experiments
                args.current_cultivar = left_out #change for tensorboard
                loss_dicts[left_out] = run_embed_train(args, embedding = embeds[idx])
            
            overall_loss = loss_dicts
            
        #random embeds test
        elif args.setting in ['rand_embeds']:
            from nn.models import pheno_embedding_net_finetune_emb as nn_model_finetune
            
            loss_dicts = dict()
            finetune_loss_dicts = dict()
            args.cultivar_list = list(valid_cultivars)
            #other_cultivars = list(set(valid_cultivars) - set([left_out]))
            args.current_cultivar = 'all'
            args.no_of_cultivars = len(valid_cultivars)
            args.nn_model = nn_model
            args.nn_model_finetune = nn_model_finetune
           
            
            # get dataset by selecting features
            for idx, left_out in enumerate(valid_cultivars):
                args.cultivar_list = list([left_out])
                args.trial = 'trial_'+str(0)
                args.dataset = create_dataset_multiple_cultivars(args)
                args.current_cultivar = 'all'
                args.pretrained_path = os.path.join(
                    './models', 'embed_oracle', args.current_cultivar, args.trial, args.experiment+'_setting_oracle_variant_'+args.variant+'_weighting_'+args.weighting+'_unfreeze_'+args.unfreeze+'_nonlinear_'+args.nonlinear+'_scratch_'+args.scratch+".pt")
                    # similar for all experiments
                args.current_cultivar = left_out #change for tensorboard
                loss_dicts[left_out] = run_embed_eval(args)
            
            overall_loss = loss_dicts
        #all setting
        elif args.setting in ['all']:
            loss_dicts = dict()
            finetune_loss_dicts = dict()
            args.current_cultivar = 'all'
            args.no_of_cultivars = len(valid_cultivars)
            # get dataset by selecting features
            args.cultivar_list = list(valid_cultivars)
            args.nn_model = nn_model
            # similar for all experiments
            for trial in range(3):
                args.trial = 'trial_'+str(trial)
                args.dataset = create_dataset_multiple_cultivars(args)
                loss_dicts[args.trial] = run_experiment(args)
            overall_loss[args.experiment] = loss_dicts
        elif args.setting in ['allfinetune']:
            loss_dicts = dict()
            finetune_loss_dicts = dict()
            args.current_cultivar = 'all'
            args.no_of_cultivars = len(valid_cultivars)
            # get dataset by selecting features
            args.cultivar_list = list(valid_cultivars)
            args.nn_model = nn_model
            args.nn_model_finetune = nn_model_finetune
            # # similar for all experiments
            for trial in range(3):
                args.trial = 'trial_'+str(trial)
                args.dataset = create_dataset_multiple_cultivars(args)
                loss_dicts[args.trial] = run_experiment(args)
            # similar for all experiments
            print("before finetune", loss_dicts)
            for left_out in valid_cultivars:
                gc.collect()
                args.cultivar_list = list([left_out])
                for trial in range(3):
                    args.trial = 'trial_'+str(trial)
                    args.dataset = create_dataset_multiple_cultivars(args)
                    args.pretrained_path = os.path.join(
                        './models', args.name, args.current_cultivar, args.trial, args.experiment+'_setting_'+args.setting+'_variant_'+args.variant+'_weighting_'+args.weighting+'_unfreeze_'+args.unfreeze+'_nonlinear_'+args.nonlinear+'_scratch_'+args.scratch+".pt")
                    # similar for all experiments
                    finetune_loss_dicts[args.trial] = run_experiment(args, finetune=True)
                for trial in range(3):
                    args.trial = 'trial_'+str(trial)
                    loss_dicts[args.trial].update(finetune_loss_dicts[args.trial])
            print("after finetune", loss_dicts)
            overall_loss[args.experiment] = loss_dicts
            
        elif args.setting in ['test_pheno']:
            args.setting = 'baseline_each'
            loss_dicts = dict()
            args.cultivar_list = ['Zinfandel']
            args.current_cultivar = 'Zinfandel'
            args.no_of_cultivars = 17
            args.nn_model = nn_model
            args.nn_model_finetune = nn_model_finetune
           
            # get dataset by selecting features
            args.trial = 'trial_'+str(2)
            args.dataset = create_dataset_multiple_cultivars(args)
            
            args.pretrained_path = os.path.join(
                #'./models', 'embed_oracle', args.current_cultivar, args.trial, 'pheno_embedding_setting_oracle_variant_'+args.variant+'_weighting_'+args.weighting+'_unfreeze_'+args.unfreeze+'_nonlinear_'+args.nonlinear+'_scratch_'+args.scratch+".pt")
                #'./models', 'experiment051723', args.current_cultivar, 'trial_0', args.experiment+'_setting_embed_variant_'+args.variant+'_weighting_'+args.weighting+'_unfreeze_'+args.unfreeze+'_nonlinear_'+args.nonlinear+'_scratch_'+args.scratch+".pt")
                #'./models', 'emb_models063023', args.current_cultivar, 'trial_0', args.experiment+'_setting_leaveoneout_variant_'+args.variant+'_weighting_'+args.weighting+'_unfreeze_'+args.unfreeze+'_nonlinear_'+args.nonlinear+'_scratch_'+args.scratch+".pt")
                './models', 'mtl_models063023', args.current_cultivar, 'trial_0', args.experiment+'_setting_leaveoneout_variant_'+args.variant+'_weighting_'+args.weighting+'_unfreeze_'+args.unfreeze+'_nonlinear_'+args.nonlinear+'_scratch_'+args.scratch+".pt")
            print(args.pretrained_path)                    
            loss_dicts['leaveoneout'] = run_embed_eval(args, finetune=False)
            
            args.no_of_cultivars = 1
            args.pretrained_path = os.path.join(
                #'./models', 'embed_oracle', args.current_cultivar, args.trial, 'pheno_embedding_setting_oracle_variant_'+args.variant+'_weighting_'+args.weighting+'_unfreeze_'+args.unfreeze+'_nonlinear_'+args.nonlinear+'_scratch_'+args.scratch+".pt")
                #'./models', 'experiment051723', args.current_cultivar, 'trial_0', args.experiment+'_setting_embed_variant_'+args.variant+'_weighting_'+args.weighting+'_unfreeze_'+args.unfreeze+'_nonlinear_'+args.nonlinear+'_scratch_'+args.scratch+".pt")
                #'./models', 'emb_models063023', args.current_cultivar, 'trial_0', args.experiment+'_setting_leaveoneout_variant_'+args.variant+'_weighting_'+args.weighting+'_unfreeze_'+args.unfreeze+'_nonlinear_'+args.nonlinear+'_scratch_'+args.scratch+".pt")
                #'./models', 'mtl_models063023', args.current_cultivar, 'trial_0', args.experiment+'_setting_leaveoneout_variant_'+args.variant+'_weighting_'+args.weighting+'_unfreeze_'+args.unfreeze+'_nonlinear_'+args.nonlinear+'_scratch_'+args.scratch+".pt")
                './models','mtl_all_pheno','all','trial_2','mtl_setting_all_variant_none_weighting_none_unfreeze_no_nonlinear_no_scratch_no.pt')
            print(args.pretrained_path)                    
            loss_dicts['all'] = run_embed_eval(args, finetune=False)
            
            overall_loss = loss_dicts
        
    print(overall_loss)
    
    import numpy as np
    #all_diffs = np.array([y for x in list(args.bb_day_diff.values()) for y in x])
    #print("outliers,",all_diffs[all_diffs>=50])
    #print("mean excluding outliers",args.experiment,np.mean(all_diffs[all_diffs<50]))
    modelSavePath = "./models/"
    Path(os.path.join(modelSavePath, args.name)).mkdir(parents=True, exist_ok=True)
    with open(os.path.join('./models', args.name, args.experiment+'_setting_'+args.setting+'_variant_'+args.variant+'_weighting_'+args.weighting+'_unfreeze_'+args.unfreeze+'_nonlinear_'+args.nonlinear+'_scratch_'+args.scratch+"_losses.pkl"), 'wb') as f:
        pickle.dump(overall_loss, f)
        
    if args.experiment in ['concat_embedding_vectors']:
        with open(os.path.join('./models', args.name, "rnn_vectors.pkl"), 'wb') as f:
            pickle.dump(overall_rnn_vectors, f)
        with open(os.path.join('./models', args.name, "penul_vectors.pkl"), 'wb') as f:
            pickle.dump(overall_penul_vectors, f)
        with open(os.path.join('./models', args.name, "LTE_pred.pkl"), 'wb') as f:
            pickle.dump(overall_LTE_dict, f)
