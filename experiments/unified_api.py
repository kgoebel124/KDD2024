import torch
from torch.autograd import Variable
from util.create_dataset import MyDataset, get_not_nan
from util.create_dataset import create_dataset_multiple_cultivars
from util.data_processing import evaluate
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.nn as nn
import os
from pathlib import Path
import pickle
#import ipdb as pdb

def run_experiment(args, finetune=False, embedding = None):
    dataset = args.dataset
    feature_len = dataset['train']['x'].shape[-1]
    no_of_cultivars = args.no_of_cultivars
    if finetune:
        if args.experiment in ['pheno_embedding','mtl']:
            model = args.nn_model_finetune(feature_len, no_of_cultivars, len(args.phenos_list), nonlinear = args.nonlinear)
        else:
            model = args.nn_model_finetune(feature_len, no_of_cultivars, nonlinear = args.nonlinear)
    else:
        if args.experiment in ['pheno_embedding','mtl']:
            model = args.nn_model(feature_len, no_of_cultivars, len(args.phenos_list), nonlinear = args.nonlinear)
        else:
            model = args.nn_model(feature_len, no_of_cultivars, nonlinear = args.nonlinear)
    if args.unfreeze=='yes':
        for param in model.parameters():
            param.requires_grad = True
    if finetune:
        model.load_state_dict(torch.load(args.pretrained_path), strict=False)
    model.to(args.device)
    trainable_params = sum([np.prod(p.size()) for p in filter(
        lambda p: p.requires_grad, model.parameters())])
    print("Trainable Parameters:", trainable_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss(reduction='none')
    criterion.to(args.device)
    criterion2 = nn.BCELoss(reduction='none')
    criterion2.to(args.device)
    if finetune:
        log_dir = os.path.join('./tensorboard/',args.name, args.experiment+'_setting_'+args.setting+'_variant_'+args.variant+'_weighting_'+args.weighting+'_unfreeze_'+args.unfreeze+'_nonlinear_'+args.nonlinear+'_scratch_'+args.scratch+"_finetune", args.trial, args.current_cultivar)
    else:
        log_dir = os.path.join('./tensorboard/',args.name, args.experiment+'_setting_'+args.setting+'_variant_'+args.variant+'_weighting_'+args.weighting+'_unfreeze_'+args.unfreeze+'_nonlinear_'+args.nonlinear+'_scratch_'+args.scratch, args.trial, args.current_cultivar)
    writer = SummaryWriter(log_dir)
    train_dataset = MyDataset(dataset['train'])
    trainLoader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = MyDataset(dataset['test'])
    valLoader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    embeds = dict()
    for epoch in range(args.epochs):
        # Training Loop
        model.train()
        total_loss = 0
        count = 0
        total_loss_pheno = 0
        for i, (x, y, cultivar_id, freq) in enumerate(trainLoader):
            x_torch = x.to(args.device)
            y_torch = y.to(args.device)
            freq = freq.to(args.device)
            cultivar_id_torch = cultivar_id.to(args.device)
            count += 1
            if args.experiment == 'pheno_embedding':
                out_lt_10, out_lt_50, out_lt_90, out_ph, _, embed = model(x_torch, cultivar_label=cultivar_id_torch)
                if epoch == args.epochs - 1: #save embeds at the end
                    for b in range(cultivar_id.shape[0]):
                        embeds[int(cultivar_id[b,0])] = embed[b,0,:]
            elif args.experiment == 'concat_embedding_vectors':
                out_lt_10, out_lt_50, out_lt_90, out_ph, _, _, _ = model(x_torch, cultivar_label=cultivar_id_torch)
            else:
                out_lt_10, out_lt_50, out_lt_90, out_ph, _ = model(x_torch, cultivar_label=cultivar_id_torch)
            optimizer.zero_grad()       # zero the parameter gradients
            #replace nan in gt with 0s, replace corresponding values in pred with 0s
            nan_locs_lt_10 = y_torch[:, :, 0].isnan()
            nan_locs_lt_50 = y_torch[:, :, 1].isnan()
            nan_locs_lt_90 = y_torch[:, :, 2].isnan()
            #nan_locs_bb = y_torch[:, :, 3].isnan()
            out_lt_10[:,:,0][nan_locs_lt_10] = 0
            out_lt_50[:,:,0][nan_locs_lt_50] = 0
            out_lt_90[:,:,0][nan_locs_lt_90] = 0
            #out_ph[:,:,0][nan_locs_bb] = 0
            #assuming lte values are present together
            y_torch = torch.nan_to_num(y_torch)
            loss_lt_10 = criterion(out_lt_10[:,:,0], y_torch[:, :, 0])[~nan_locs_lt_10]  # LT10 GT
            loss_lt_50 = criterion(out_lt_50[:,:,0], y_torch[:, :, 1])[~nan_locs_lt_50]
            loss_lt_90 = criterion(out_lt_90[:,:,0], y_torch[:, :, 2])[~nan_locs_lt_90]
            freq_lt_10 = freq[~nan_locs_lt_10]
            freq_lt_50 = freq[~nan_locs_lt_50]
            freq_lt_90 = freq[~nan_locs_lt_90]
            loss_pheno = 0
            if args.include_pheno:
                for j in range(out_ph.shape[0]):
                    nan_locs_pheno = y_torch[:, :, j+3].isnan()
                    loss_pheno += (criterion2(out_ph[j,:,:,0], y_torch[:,:,j+3])[~nan_locs_pheno]).mean()

            # if args.weighting=='inverse_freq':
            #     loss = torch.mul(loss_lt_10, freq_lt_10).mean() + torch.mul(loss_lt_50, freq_lt_50).mean() + torch.mul(loss_lt_90, freq_lt_90).mean()
            # elif args.weighting=='uncertainty':
            #     pass
            # else:
            loss = 0
            if args.include_lte:
                loss += loss_lt_10.mean() + loss_lt_50.mean() + loss_lt_90.mean()
            if args.include_pheno:
                loss += loss_pheno
            loss.backward()             # backward +
            optimizer.step()            # optimize
            total_loss += loss.item()
            if args.include_pheno:
                total_loss_pheno+= loss_pheno.mean().item()
        writer.add_scalar('Train_Loss', total_loss / count, epoch)
        writer.add_scalar('Train_Loss_Pheno', total_loss_pheno / count, epoch)
        # Validation Loop
        with torch.no_grad():
            model.eval()
            total_loss = 0
            total_loss_pheno = 0
            count = 0
            for i, (x, y, cultivar_id, freq) in enumerate(valLoader):
                x_torch = x.to(args.device)
                y_torch = y.to(args.device)
                cultivar_id_torch = cultivar_id.to(args.device)
                count += 1
                if args.experiment == 'pheno_embedding':
                    out_lt_10, out_lt_50, out_lt_90, out_ph, _, _ = model(x_torch, cultivar_label=cultivar_id_torch)
                elif args.experiment == 'concat_embedding_vectors':
                    out_lt_10, out_lt_50, out_lt_90, out_ph, _, _, _ = model(x_torch, cultivar_label=cultivar_id_torch)
                else:
                    out_lt_10, out_lt_50, out_lt_90, out_ph, _ = model(x_torch, cultivar_label=cultivar_id_torch)
                #replace nan in gt with 0s, replace corresponding values in pred with 0s
                nan_locs_lt_10 = y_torch[:, :, 0].isnan()
                nan_locs_lt_50 = y_torch[:, :, 1].isnan()
                nan_locs_lt_90 = y_torch[:, :, 2].isnan()
                out_lt_10[:,:,0][nan_locs_lt_10] = 0
                out_lt_50[:,:,0][nan_locs_lt_50] = 0
                out_lt_90[:,:,0][nan_locs_lt_90] = 0
                y_torch = torch.nan_to_num(y_torch)
                # getting non nan values is slow right now due to copying to cpu, write pytorch gpu version
                loss_lt_10 = criterion(out_lt_10[:,:,0], y_torch[:, :, 0])[~nan_locs_lt_10]  # LT10 GT
                loss_lt_50 = criterion(out_lt_50[:,:,0], y_torch[:, :, 1])[~nan_locs_lt_50]
                loss_lt_90 = criterion(out_lt_90[:,:,0], y_torch[:, :, 2])[~nan_locs_lt_90]
                loss_pheno = 0
                if args.include_pheno:
                    for j in range(out_ph.shape[0]):
                        nan_locs_pheno = y_torch[:, :, j+3].isnan()
                        loss_pheno += (criterion2(out_ph[j,:,:,0], y_torch[:,:,j+3])[~nan_locs_pheno]).mean()
                
                loss = 0
                if args.include_lte:
                    loss += loss_lt_10.mean() + loss_lt_50.mean() + loss_lt_90.mean()
                if args.include_pheno:
                    loss += loss_pheno
                total_loss += loss.mean().item()
                if args.include_pheno:
                    total_loss_pheno+= loss_pheno.mean().item()
            writer.add_scalar('Val_Loss', total_loss / count, epoch)
            writer.add_scalar('Val_Loss_Pheno', total_loss_pheno / count, epoch)
    loss_dict = dict()
    modelSavePath = "./models/"
    Path(os.path.join(modelSavePath, args.name, args.current_cultivar, args.trial)).mkdir(parents=True, exist_ok=True)
    if finetune:
        torch.save(model.state_dict(), os.path.join('./models', args.name, args.current_cultivar, args.trial, args.experiment+'_setting_'+args.setting+'_variant_'+args.variant+'_weighting_'+args.weighting+'_unfreeze_'+args.unfreeze+'_nonlinear_'+args.nonlinear+'_scratch_'+args.scratch+"_finetune.pt"))
    else:
        dir_path = os.path.join('./models', args.name, args.current_cultivar, args.trial)
        if (not os.path.exists(dir_path)):
            os.makedirs(dir_path)
        torch.save(model.state_dict(), os.path.join('./models', args.name, args.current_cultivar, args.trial, args.experiment+'_setting_'+args.setting+'_variant_'+args.variant+'_weighting_'+args.weighting+'_unfreeze_'+args.unfreeze+'_nonlinear_'+args.nonlinear+'_scratch_'+args.scratch+".pt"))
    # Validation Loop
    total_loss_lt_10, total_loss_lt_50, total_loss_lt_90, total_loss_pheno = 0, 0, 0, 0
    with torch.no_grad():
        model.eval()
        for i, ((x, y, cultivar_id, freq), cultivar) in enumerate(zip(valLoader,args.cultivar_list)):
            x_torch = x.to(args.device)
            y_torch = y.to(args.device)
            cultivar_id_torch = cultivar_id.to(args.device)
            if args.experiment == 'pheno_embedding':
                out_lt_10, out_lt_50, out_lt_90, out_ph, _, _ = model(x_torch, cultivar_label=cultivar_id_torch)
            elif args.experiment == 'concat_embedding_vectors':
                out_lt_10, out_lt_50, out_lt_90, out_ph, _, _, _ = model(x_torch, cultivar_label=cultivar_id_torch)
            else:
                out_lt_10, out_lt_50, out_lt_90, out_ph, _ = model(x_torch, cultivar_label=cultivar_id_torch)
            #replace nan in gt with 0s, replace corresponding values in pred with 0s
            nan_locs_lt_10 = y_torch[:, :, 0].isnan()
            nan_locs_lt_50 = y_torch[:, :, 1].isnan()
            nan_locs_lt_90 = y_torch[:, :, 2].isnan()
            out_lt_10[:,:,0][nan_locs_lt_10] = 0
            out_lt_50[:,:,0][nan_locs_lt_50] = 0
            out_lt_90[:,:,0][nan_locs_lt_90] = 0
            y_torch = torch.nan_to_num(y_torch)
            loss_lt_10 = criterion(out_lt_10[:,:,0], y_torch[:, :, 0])[~nan_locs_lt_10].mean().item()  # LT10 GT
            loss_lt_50 = criterion(out_lt_50[:,:,0], y_torch[:, :, 1])[~nan_locs_lt_50].mean().item()
            loss_lt_90 = criterion(out_lt_90[:,:,0], y_torch[:, :, 2])[~nan_locs_lt_90].mean().item()
            loss_pheno = 0
            if args.include_pheno:
                for j in range(out_ph.shape[0]):
                    nan_locs_pheno = y_torch[:, :, j+3].isnan()
                    loss_pheno += (criterion2(out_ph[j,:,:,0], y_torch[:,:,j+3])[~nan_locs_pheno]).mean().item()
            total_loss_lt_10+=loss_lt_10
            total_loss_lt_50+=loss_lt_50
            total_loss_lt_90+=loss_lt_90
            total_loss_pheno+=loss_pheno
            loss_dict[cultivar] = list([np.sqrt(loss_lt_10), np.sqrt(loss_lt_50), np.sqrt(loss_lt_90), loss_pheno])
    loss_dict['overall'] = list([np.sqrt(total_loss_lt_10), np.sqrt(total_loss_lt_50), np.sqrt(total_loss_lt_90),total_loss_pheno])
    if args.experiment == 'pheno_embedding':
        return loss_dict, embeds
    else:
        return loss_dict

def run_eval(args, finetune=False):
    import matplotlib.pyplot as plt
    dataset = args.dataset
    feature_len = dataset['train']['x'].shape[-1]
    no_of_cultivars = args.no_of_cultivars
    model = args.nn_model(feature_len, no_of_cultivars, nonlinear = args.nonlinear)
    model.load_state_dict(torch.load(os.path.join('./models', args.name, args.current_cultivar, args.trial, args.experiment+'_setting_'+args.setting+'_variant_'+args.variant+'_weighting_'+args.weighting+'_unfreeze_'+args.unfreeze+'_nonlinear_'+args.nonlinear+'_scratch_'+args.scratch+".pt")), strict=False)
    
    model.to(args.device)
    trainable_params = sum([np.prod(p.size()) for p in filter(
        lambda p: p.requires_grad, model.parameters())])
    print("Trainable Parameters:", trainable_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss(reduction='none')
    criterion.to(args.device)
    criterion2 = nn.BCELoss(reduction='none')
    criterion2.to(args.device)
    train_dataset = MyDataset(dataset['train'])
    trainLoader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = MyDataset(dataset['test'])
    valLoader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    total_loss_lt_10, total_loss_lt_50, total_loss_lt_90, total_loss_bb = 0, 0, 0, 0
    loss_dict = dict()
    with torch.no_grad():
        model.eval()
        for i, ((x, y, cultivar_id, freq), cultivar) in enumerate(zip(valLoader,args.cultivar_list)):
            x_torch = x.to(args.device)
            y_torch = y.to(args.device)
            cultivar_id_torch = cultivar_id.to(args.device)
            out_lt_10, out_lt_50, out_lt_90, out_ph, _ = model(x_torch, cultivar_label=cultivar_id_torch)
            #replace nan in gt with 0s, replace corresponding values in pred with 0s
            nan_locs_lt_10 = y_torch[:, :, 0].isnan()
            nan_locs_lt_50 = y_torch[:, :, 1].isnan()
            nan_locs_lt_90 = y_torch[:, :, 2].isnan()
            out_lt_10[:,:,0][nan_locs_lt_10] = 0
            out_lt_50[:,:,0][nan_locs_lt_50] = 0
            out_lt_90[:,:,0][nan_locs_lt_90] = 0
            y_torch = torch.nan_to_num(y_torch)
            loss_lt_10 = criterion(out_lt_10[:,:,0], y_torch[:, :, 0])[~nan_locs_lt_10].mean().item()  # LT10 GT
            loss_lt_50 = criterion(out_lt_50[:,:,0], y_torch[:, :, 1])[~nan_locs_lt_50].mean().item()
            loss_lt_90 = criterion(out_lt_90[:,:,0], y_torch[:, :, 2])[~nan_locs_lt_90].mean().item()
            loss_bb = criterion2(out_ph[:,:,0], y_torch[:,:,3]).mean().item()
            #pdb.set_trace()
            total_loss_lt_10+=loss_lt_10
            total_loss_lt_50+=loss_lt_50
            total_loss_lt_90+=loss_lt_90
            total_loss_bb+=loss_bb
            loss_dict[cultivar] = list([np.sqrt(loss_lt_10), np.sqrt(loss_lt_50), np.sqrt(loss_lt_90), loss_bb])
            no_days = out_ph.shape[1]
            # #print(cultivar+' '+args.trial+' '+args.experiment)
            # #print("difference in budbreak occurance for season 1",abs(torch.where(out_ph[0,:,0]>0.5)[0][0].item()-torch.where(y[0,:,3]==1)[0][0].item()))
            Path(os.path.join('./plots', 'bb', args.experiment, cultivar, args.trial)).mkdir(parents=True, exist_ok=True)
            try:
                args.bb_day_diff[cultivar].append(abs(torch.where(out_ph[0,:,0]>0.5)[0][0].item()-torch.where(y[0,:,3]==1)[0][0].item()))
            except Exception as e:
                args.bb_day_diff[cultivar].append(100)
            # #print("difference in budbreak occurance for season 2",abs(torch.where(out_ph[1,:,0]>0.5)[0][0].item()-torch.where(y[1,:,3]==1)[0][0].item()))
            try:
                args.bb_day_diff[cultivar].append(abs(torch.where(out_ph[1,:,0]>0.5)[0][0].item()-torch.where(y[1,:,3]==1)[0][0].item()))
            except Exception as e:
                args.bb_day_diff[cultivar].append(100)
            plt.close('all')
            plt.figure(figsize=(16, 9))
            plt.scatter(np.arange(no_days),out_ph[0,:,0].cpu().numpy(), marker = 'o')
            plt.scatter(np.arange(no_days),y_torch[0,:,3].cpu().numpy(), marker = 'v')
            plt.xlabel('Day')
            plt.ylabel('Budbreak')
            plt.title('Budbreak for '+cultivar+' and experiment '+args.experiment+' CE Loss '+str(loss_bb)[0:5]+' difference in days '+str(args.bb_day_diff[cultivar][0]))
            plt.savefig(os.path.join('./plots', 'bb', args.experiment, cultivar, args.trial,'plot.png'))
    loss_dict['overall'] = list([np.sqrt(total_loss_lt_10), np.sqrt(total_loss_lt_50), np.sqrt(total_loss_lt_90),total_loss_bb])
    return loss_dict
    
    
def run_embed_train(args, finetune=True, embedding = None):
    dataset = args.dataset
    feature_len = dataset['train']['x'].shape[-1]
    
    no_of_cultivars = args.no_of_cultivars
    
    if embedding == None: 
        if args.setting == 'embed_lin':
            embedding=torch.rand(no_of_cultivars, requires_grad=True, device=args.device) #TODO: best starting values?
        else:
            embedding=torch.rand(feature_len, requires_grad=True, device=args.device) #TODO: best starting values?
    else:
        #embedding = torch.clone(embedding)
        embedding = embedding.detach().clone()
        embedding.to(args.device).requires_grad_(True)
    print(embedding)
    
    
    if finetune:
        model = args.nn_model_finetune(feature_len, no_of_cultivars, len(args.phenos_list), nonlinear = args.nonlinear)
    else:
        model = args.nn_model(feature_len, no_of_cultivars, len(args.phenos_list), nonlinear = args.nonlinear)
    if args.unfreeze=='yes':
        for param in model.parameters():
            param.requires_grad = True
    if finetune:
        model.load_state_dict(torch.load(args.pretrained_path), strict=False)
    model.to(args.device)
    
    optimizer = torch.optim.Adam([embedding], lr=args.lr_emb)
    criterion = nn.MSELoss(reduction='none')
    criterion.to(args.device)
    criterion2 = nn.BCELoss(reduction='none')
    criterion2.to(args.device)
    if finetune:
        log_dir = os.path.join('./tensorboard/',args.name, args.experiment+'_setting_'+args.setting+'_variant_'+args.variant+'_weighting_'+args.weighting+'_unfreeze_'+args.unfreeze+'_nonlinear_'+args.nonlinear+'_scratch_'+args.scratch+"_finetune", args.trial, args.current_cultivar)
    else:
        log_dir = os.path.join('./tensorboard/',args.name, args.experiment+'_setting_'+args.setting+'_variant_'+args.variant+'_weighting_'+args.weighting+'_unfreeze_'+args.unfreeze+'_nonlinear_'+args.nonlinear+'_scratch_'+args.scratch, args.trial, args.current_cultivar)
    writer = SummaryWriter(log_dir)
    
    train_dataset = MyDataset(dataset['train'])
    trainLoader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = MyDataset(dataset['test'])
    valLoader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    
    for epoch in range(args.epochs):
        # Training Loop
        model.train()
        total_loss = 0
        count = 0
        total_loss_pheno = 0
        for i, (x, y, cultivar_id, freq) in enumerate(trainLoader):
            x_torch = x.to(args.device)
            y_torch = y.to(args.device)
            freq = freq.to(args.device)
            cultivar_id_torch = cultivar_id.to(args.device)
            count += 1
            if args.setting == 'embed_lin':
                out_lt_10, out_lt_50, out_lt_90, out_ph, _ = model(x_torch, cultivar_label=embedding)
            else:
                out_lt_10, out_lt_50, out_lt_90, out_ph, _ = model(x_torch, embedding, cultivar_label=cultivar_id_torch)
            optimizer.zero_grad()       # zero the parameter gradients
            
            #replace nan in gt with 0s, replace corresponding values in pred with 0s
            nan_locs_lt_10 = y_torch[:, :, 0].isnan()
            nan_locs_lt_50 = y_torch[:, :, 1].isnan()
            nan_locs_lt_90 = y_torch[:, :, 2].isnan()
            #nan_locs_bb = y_torch[:, :, 3].isnan()
            out_lt_10[:,:,0][nan_locs_lt_10] = 0
            out_lt_50[:,:,0][nan_locs_lt_50] = 0
            out_lt_90[:,:,0][nan_locs_lt_90] = 0
            #out_ph[:,:,0][nan_locs_bb] = 0
            #assuming lte values are present together
            y_torch = torch.nan_to_num(y_torch)
            loss_lt_10 = criterion(out_lt_10[:,:,0], y_torch[:, :, 0])[~nan_locs_lt_10]  # LT10 GT
            loss_lt_50 = criterion(out_lt_50[:,:,0], y_torch[:, :, 1])[~nan_locs_lt_50]
            loss_lt_90 = criterion(out_lt_90[:,:,0], y_torch[:, :, 2])[~nan_locs_lt_90]
            freq_lt_10 = freq[~nan_locs_lt_10]
            freq_lt_50 = freq[~nan_locs_lt_50]
            freq_lt_90 = freq[~nan_locs_lt_90]

            loss_LTE = loss = loss_lt_10.mean() + loss_lt_50.mean() + loss_lt_90.mean()           
            loss_pheno = 0
            for j in range(out_ph.shape[0]):
                nan_locs_pheno = y_torch[:, :, j+3].isnan()
                loss_pheno += (criterion2(out_ph[j,:,:,0], y_torch[:,:,j+3])[~nan_locs_pheno]).mean()
            
            if args.train_embeds_LTE:
                loss = loss_LTE
            else:
                loss = loss_pheno
            loss.backward()             # backward +
            optimizer.step()            # optimize
            total_loss += loss.item()
            total_loss_pheno+= loss_pheno.mean().item()
        writer.add_scalar('Train_Loss', total_loss / count, epoch)
        writer.add_scalar('Train_Loss_Pheno', total_loss_pheno / count, epoch)
        
        # Validation Loop
        with torch.no_grad():
            model.eval()
            total_loss = 0
            total_loss_pheno = 0
            count = 0
            for i, (x, y, cultivar_id, freq) in enumerate(valLoader):
                x_torch = x.to(args.device)
                y_torch = y.to(args.device)
                cultivar_id_torch = cultivar_id.to(args.device)
                count += 1
                if args.setting == 'embed_lin':
                    out_lt_10, out_lt_50, out_lt_90, out_ph, _ = model(x_torch, cultivar_label=embedding)
                else:
                    out_lt_10, out_lt_50, out_lt_90, out_ph, _ = model(x_torch, embedding, cultivar_label=cultivar_id_torch)
                #replace nan in gt with 0s, replace corresponding values in pred with 0s
                nan_locs_lt_10 = y_torch[:, :, 0].isnan()
                nan_locs_lt_50 = y_torch[:, :, 1].isnan()
                nan_locs_lt_90 = y_torch[:, :, 2].isnan()
                out_lt_10[:,:,0][nan_locs_lt_10] = 0
                out_lt_50[:,:,0][nan_locs_lt_50] = 0
                out_lt_90[:,:,0][nan_locs_lt_90] = 0
                y_torch = torch.nan_to_num(y_torch)
                # getting non nan values is slow right now due to copying to cpu, write pytorch gpu version
                loss_lt_10 = criterion(out_lt_10[:,:,0], y_torch[:, :, 0])[~nan_locs_lt_10]  # LT10 GT
                loss_lt_50 = criterion(out_lt_50[:,:,0], y_torch[:, :, 1])[~nan_locs_lt_50]
                loss_lt_90 = criterion(out_lt_90[:,:,0], y_torch[:, :, 2])[~nan_locs_lt_90]
                loss_pheno = 0
                for j in range(out_ph.shape[0]):
                    nan_locs_pheno = y_torch[:, :, j+3].isnan()
                    loss_pheno += (criterion2(out_ph[j,:,:,0], y_torch[:,:,j+3])[~nan_locs_pheno]).mean()
                
                loss = loss_lt_10.mean() + loss_lt_50.mean() + loss_lt_90.mean()
                total_loss += loss.mean().item()
                total_loss_pheno+= loss_pheno.mean().item()
            writer.add_scalar('Val_Loss_LTE', total_loss / count, epoch)
            writer.add_scalar('Val_Loss_Pheno', total_loss_pheno / count, epoch)
    loss_dict = dict()
    modelSavePath = "./models/"
    Path(os.path.join(modelSavePath, args.name, args.current_cultivar, args.trial)).mkdir(parents=True, exist_ok=True)
    if finetune:
        torch.save(model.state_dict(), os.path.join('./models', args.name, args.current_cultivar, args.trial, args.experiment+'_setting_'+args.setting+'_variant_'+args.variant+'_weighting_'+args.weighting+'_unfreeze_'+args.unfreeze+'_nonlinear_'+args.nonlinear+'_scratch_'+args.scratch+"_finetune.pt"))
    else:
        dir_path = os.path.join('./models', args.name, args.current_cultivar, args.trial)
        if (not os.path.exists(dir_path)):
            os.makedirs(dir_path)
        torch.save(model.state_dict(), os.path.join('./models', args.name, args.current_cultivar, args.trial, args.experiment+'_setting_'+args.setting+'_variant_'+args.variant+'_weighting_'+args.weighting+'_unfreeze_'+args.unfreeze+'_nonlinear_'+args.nonlinear+'_scratch_'+args.scratch+".pt"))
    
    # Validation Loop
    total_loss_lt_10, total_loss_lt_50, total_loss_lt_90, total_loss_pheno = 0, 0, 0, 0
    with torch.no_grad():
        model.eval()
        for i, ((x, y, cultivar_id, freq), cultivar) in enumerate(zip(valLoader,args.cultivar_list)):
            x_torch = x.to(args.device)
            y_torch = y.to(args.device)
            cultivar_id_torch = cultivar_id.to(args.device)
            if args.setting == 'embed_lin':
                out_lt_10, out_lt_50, out_lt_90, out_ph, _ = model(x_torch, cultivar_label=embedding)
            else:
                out_lt_10, out_lt_50, out_lt_90, out_ph, _ = model(x_torch, embedding, cultivar_label=cultivar_id_torch)
            #replace nan in gt with 0s, replace corresponding values in pred with 0s
            nan_locs_lt_10 = y_torch[:, :, 0].isnan()
            nan_locs_lt_50 = y_torch[:, :, 1].isnan()
            nan_locs_lt_90 = y_torch[:, :, 2].isnan()
            out_lt_10[:,:,0][nan_locs_lt_10] = 0
            out_lt_50[:,:,0][nan_locs_lt_50] = 0
            out_lt_90[:,:,0][nan_locs_lt_90] = 0
            y_torch = torch.nan_to_num(y_torch)
            loss_lt_10 = criterion(out_lt_10[:,:,0], y_torch[:, :, 0])[~nan_locs_lt_10].mean().item()  # LT10 GT
            loss_lt_50 = criterion(out_lt_50[:,:,0], y_torch[:, :, 1])[~nan_locs_lt_50].mean().item()
            loss_lt_90 = criterion(out_lt_90[:,:,0], y_torch[:, :, 2])[~nan_locs_lt_90].mean().item()
            loss_pheno = 0
            for j in range(out_ph.shape[0]):
                nan_locs_pheno = y_torch[:, :, j+3].isnan()
                loss_pheno += (criterion2(out_ph[j,:,:,0], y_torch[:,:,j+3])[~nan_locs_pheno]).mean().item()
            total_loss_lt_10+=loss_lt_10
            total_loss_lt_50+=loss_lt_50
            total_loss_lt_90+=loss_lt_90
            total_loss_pheno+=loss_pheno
            loss_dict[cultivar] = list([np.sqrt(loss_lt_10), np.sqrt(loss_lt_50), np.sqrt(loss_lt_90), loss_pheno])
            
            batch, time, _ = out_lt_50.shape
            for j in range(batch):
                plot_title = 'LTE50' + str(i) + '-' + str(j)
                for k in range(time):
                    
                    writer.add_scalar(plot_title+'_pred', out_lt_50[j,k,0].item(), k)
                    if not np.isnan(y_torch[j,k,1].item()):
                        writer.add_scalar(plot_title+'_actual', y_torch[j,k,1].item(), k)
    loss_dict['overall'] = list([np.sqrt(total_loss_lt_10), np.sqrt(total_loss_lt_50), np.sqrt(total_loss_lt_90),total_loss_pheno])
    
    print(embedding)
    return loss_dict
    
def run_embed_eval(args, finetune=True):
    dataset = args.dataset
    feature_len = dataset['train']['x'].shape[-1]
    loss_dict = dict()
    
    no_of_cultivars = args.no_of_cultivars
    if finetune:
        model = args.nn_model_finetune(feature_len, no_of_cultivars, len(args.phenos_list), nonlinear = args.nonlinear)
    else:
        model = args.nn_model(feature_len, no_of_cultivars, len(args.phenos_list), nonlinear = args.nonlinear)
    if args.unfreeze=='yes':
        for param in model.parameters():
            param.requires_grad = True
    model.load_state_dict(torch.load(args.pretrained_path, map_location=args.device), strict=False)
    if(args.extra_embeds):#add extra cultivars to the model first
        with open('./models/extra_embeds/uniform_100.pkl','rb') as f:
        #with open('./models/extra_embeds/convex_comb_0_19.pkl','rb') as f:
            new_weights = pickle.load(f)
            
        #with open('./models/extra_embeds/uniform_68'+str(args.a_weight)+'.pkl','rb') as f:
        #    new_weights = pickle.load(f)
        added_cultivars, _ = new_weights[args.current_cultivar].shape
        if(args.exclude_source):
            model.embedding = nn.Embedding.from_pretrained(new_weights[args.current_cultivar])
            no_of_cultivars = added_cultivars
        else:
            concat_weights = torch.cat((model.embedding.weight, new_weights[args.current_cultivar]))
            model.embedding = nn.Embedding.from_pretrained(concat_weights)
            no_of_cultivars += added_cultivars
    model.to(args.device)
    
    criterion = nn.MSELoss(reduction='none')
    criterion.to(args.device)
    criterion2 = nn.BCELoss(reduction='none')
    criterion2.to(args.device)
    if finetune:
        log_dir = os.path.join('./tensorboard/',args.name, args.experiment+'_setting_'+args.setting+'_variant_'+args.variant+'_weighting_'+args.weighting+'_unfreeze_'+args.unfreeze+'_nonlinear_'+args.nonlinear+'_scratch_'+args.scratch+"_finetune", args.trial, args.current_cultivar)
    else:
        log_dir = os.path.join('./tensorboard/',args.name, args.experiment+'_setting_'+args.setting+'_variant_'+args.variant+'_weighting_'+args.weighting+'_unfreeze_'+args.unfreeze+'_nonlinear_'+args.nonlinear+'_scratch_'+args.scratch, args.trial, args.current_cultivar)
    writer = SummaryWriter(log_dir)
    
    for trial in ['trial_0','trial_1','trial_2']:
        args.trial = trial
        
        '''
        no_of_cultivars = args.no_of_cultivars
        
        if finetune:
            model = args.nn_model_finetune(feature_len, no_of_cultivars, len(args.phenos_list), nonlinear = args.nonlinear)
        else:
            model = args.nn_model(feature_len, no_of_cultivars, len(args.phenos_list), nonlinear = args.nonlinear)
        if args.unfreeze=='yes':
            for param in model.parameters():
                param.requires_grad = True
        
        args.pretrained_path = os.path.join(
            './models', 'emb_models063023', args.current_cultivar, args.trial, args.experiment+'_setting_leaveoneout_variant_'+args.variant+'_weighting_'+args.weighting+'_unfreeze_'+args.unfreeze+'_nonlinear_'+args.nonlinear+'_scratch_'+args.scratch+".pt")
            
        
        model.load_state_dict(torch.load(args.pretrained_path, map_location=args.device), strict=False)
        if(args.extra_embeds):#add extra cultivars to the model first
            #with open('./models/extra_embeds/uniform_68.pkl','rb') as f:
            with open('./models/extra_embeds/convex_comb_'+trial+'_19.pkl','rb') as f:
                new_weights = pickle.load(f)
                
            #with open('./models/extra_embeds/uniform_68'+str(args.a_weight)+'.pkl','rb') as f:
            #    new_weights = pickle.load(f)
            added_cultivars, _ = new_weights[args.current_cultivar].shape
            if(args.exclude_source):
                model.embedding = nn.Embedding.from_pretrained(new_weights[args.current_cultivar])
                no_of_cultivars = added_cultivars
                concat_weights = new_weights[args.current_cultivar]
            else:
                concat_weights = torch.cat((model.embedding.weight, new_weights[args.current_cultivar]))
                model.embedding = nn.Embedding.from_pretrained(concat_weights)
                no_of_cultivars += added_cultivars
        model.to(args.device)
        '''
        
        args.dataset = create_dataset_multiple_cultivars(args)
        dataset=args.dataset
        train_dataset = MyDataset(dataset['train'])
        trainLoader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_dataset = MyDataset(dataset['test'])
        valLoader = DataLoader(val_dataset, batch_size=2, shuffle=False)
        if args.setting == "baseline_each":
            loss_dict[trial] = {}

        # Validation Loop
        total_loss_lt_10, total_loss_lt_50, total_loss_lt_90, total_loss_pheno = 0, 0, 0, 0
        with torch.no_grad():
            model.eval()
            for i, ((x, y, cultivar_id, freq), cultivar) in enumerate(zip(valLoader,args.cultivar_list)):
                print(cultivar+str(i))
                #print(x.shape)
                x_torch = x.to(args.device)
                y_torch = y.to(args.device)
                cultivar_id_torch = cultivar_id.to(args.device)
                #replace nan in gt with 0s, replace corresponding values in pred with 0s
                nan_locs_lt_10 = y_torch[:, :, 0].isnan()
                nan_locs_lt_50 = y_torch[:, :, 1].isnan()
                nan_locs_lt_90 = y_torch[:, :, 2].isnan()
                y_torch = torch.nan_to_num(y_torch)
                
                if args.setting == 'rand_embeds':
                    for _ in range(5):
                        embedding=torch.rand(feature_len, requires_grad=True, device=args.device)*2 - 1
                        out_lt_10, out_lt_50, out_lt_90, out_ph, _ = model(x_torch, embedding, cultivar_label=cultivar_id_torch)
                        out_lt_10[:,:,0][nan_locs_lt_10] = 0
                        out_lt_50[:,:,0][nan_locs_lt_50] = 0
                        out_lt_90[:,:,0][nan_locs_lt_90] = 0
                        loss_lt_10 = criterion(out_lt_10[:,:,0], y_torch[:, :, 0])[~nan_locs_lt_10].mean().item()  # LT10 GT
                        loss_lt_50 = criterion(out_lt_50[:,:,0], y_torch[:, :, 1])[~nan_locs_lt_50].mean().item()
                        loss_lt_90 = criterion(out_lt_90[:,:,0], y_torch[:, :, 2])[~nan_locs_lt_90].mean().item()
                        loss_pheno = 0
                        for j in range(out_ph.shape[0]):
                            nan_locs_pheno = y_torch[:, :, j+3].isnan()
                            loss_pheno += (criterion2(out_ph[j,:,:,0], y_torch[:,:,j+3])[~nan_locs_pheno]).mean().item()
                        total_loss_lt_10+=loss_lt_10
                        total_loss_lt_50+=loss_lt_50
                        total_loss_lt_90+=loss_lt_90
                        total_loss_pheno+=loss_pheno
                        loss_dict[cultivar] = list([np.sqrt(loss_lt_10), np.sqrt(loss_lt_50), np.sqrt(loss_lt_90), loss_pheno])
                        
                        loss_dict[embedding.to("cpu")] = list([np.sqrt(loss_lt_50), loss_pheno])
                        
                elif args.setting == 'baseline_avg':
                    for c_id in range(no_of_cultivars):
                        cultivar_id_torch = (torch.ones((x.shape[0], x.shape[1]))*c_id).type(torch.LongTensor).to(args.device)
                        if args.experiment == 'pheno_embedding':
                            out_lt_10, out_lt_50, out_lt_90, out_ph, _, _ = model(x_torch, cultivar_label=cultivar_id_torch)
                        else:
                            out_lt_10, out_lt_50, out_lt_90, out_ph, _ = model(x_torch, cultivar_label=cultivar_id_torch)
                        out_lt_10[:,:,0][nan_locs_lt_10] = 0
                        out_lt_50[:,:,0][nan_locs_lt_50] = 0
                        out_lt_90[:,:,0][nan_locs_lt_90] = 0
                        
                        if c_id == 0:
                            avg_out_lt_10 = out_lt_10
                            avg_out_lt_50 = out_lt_50
                            avg_out_lt_90 = out_lt_90
                            avg_out_ph = out_ph
                        else:
                            avg_out_lt_10 += out_lt_10
                            avg_out_lt_50 += out_lt_50
                            avg_out_lt_90 += out_lt_90
                            avg_out_ph += out_ph
                    
                    avg_out_lt_10 /= no_of_cultivars
                    avg_out_lt_50 /= no_of_cultivars
                    avg_out_lt_90 /= no_of_cultivars
                    avg_out_ph /= no_of_cultivars                    
                    loss_lt_10 = criterion(avg_out_lt_10[:,:,0], y_torch[:, :, 0])[~nan_locs_lt_10].mean().item()  # LT10 GT
                    loss_lt_50 = criterion(avg_out_lt_50[:,:,0], y_torch[:, :, 1])[~nan_locs_lt_50].mean().item()
                    loss_lt_90 = criterion(avg_out_lt_90[:,:,0], y_torch[:, :, 2])[~nan_locs_lt_90].mean().item()
                    loss_pheno = []
                    for j in range(out_ph.shape[0]):
                        nan_locs_pheno = y_torch[:, :, j+3].isnan()
                        loss_pheno.append((criterion2(avg_out_ph[j,:,:,0], y_torch[:,:,j+3])[~nan_locs_pheno]).mean().item())

                    return_list = list([np.sqrt(loss_lt_10), np.sqrt(loss_lt_50), np.sqrt(loss_lt_90)])
                    return_list.extend(loss_pheno)
                    loss_dict[trial] = return_list
                elif args.setting == 'baseline_wavg':
                    if(args.extra_embeds):
                        with open(os.path.join('./models', 'base_wavg',"weights_only_extra_68_" + args.a_weight + ".pkl"), 'rb') as f:
                            all_weights = pickle.load(f)
                    else:
                        with open(os.path.join('./models', 'base_wavg',"weights_" + args.a_weight + ".pkl"), 'rb') as f:
                            all_weights = pickle.load(f)
                    trial_loss_dict = {}
                    #for ph_w in range(3,7):    
                    #    weights = all_weights[args.current_cultivar][ph_w]
                    weights = all_weights[args.current_cultivar]
                    for c_id in range(no_of_cultivars):
                        cultivar_id_torch = (torch.ones((x.shape[0], x.shape[1]))*c_id).type(torch.LongTensor).to(args.device)
                        if args.experiment == 'pheno_embedding':
                            out_lt_10, out_lt_50, out_lt_90, out_ph, _, _ = model(x_torch, cultivar_label=cultivar_id_torch)
                        else:
                            out_lt_10, out_lt_50, out_lt_90, out_ph, _ = model(x_torch, cultivar_label=cultivar_id_torch)
                        out_lt_10[:,:,0][nan_locs_lt_10] = 0
                        out_lt_50[:,:,0][nan_locs_lt_50] = 0
                        out_lt_90[:,:,0][nan_locs_lt_90] = 0
                        '''
                        #print out to check weird Merlot values
                        for j in range(out_ph.shape[0]):
                            for b in range (2):
                                if out_ph[j,b,-1,0] < 0.8:
                                    nan_locs_pheno = y_torch[:, :, j+3].isnan()
                                    print(trial, c_id, b, j, out_ph[j,b,-1,0], (criterion2(out_ph[j,:,:,0], y_torch[:,:,j+3])[~nan_locs_pheno]).mean().item())
                        '''
                        if c_id == 0:
                            avg_out_lt_10 = weights[c_id]*out_lt_10
                            avg_out_lt_50 = weights[c_id]*out_lt_50
                            avg_out_lt_90 = weights[c_id]*out_lt_90
                            avg_out_ph = weights[c_id]*out_ph
                        else:
                            avg_out_lt_10 += weights[c_id]*out_lt_10
                            avg_out_lt_50 += weights[c_id]*out_lt_50
                            avg_out_lt_90 += weights[c_id]*out_lt_90
                            avg_out_ph += weights[c_id]*out_ph
                    
                                       
                    loss_lt_10 = criterion(avg_out_lt_10[:,:,0], y_torch[:, :, 0])[~nan_locs_lt_10].mean().item()  # LT10 GT
                    loss_lt_50 = criterion(avg_out_lt_50[:,:,0], y_torch[:, :, 1])[~nan_locs_lt_50].mean().item()
                    loss_lt_90 = criterion(avg_out_lt_90[:,:,0], y_torch[:, :, 2])[~nan_locs_lt_90].mean().item()
                    loss_pheno = []
                    for j in range(out_ph.shape[0]):
                        nan_locs_pheno = y_torch[:, :, j+3].isnan()
                        loss_pheno.append((criterion2(avg_out_ph[j,:,:,0], y_torch[:,:,j+3])[~nan_locs_pheno]).mean().item())

                    return_list = list([np.sqrt(loss_lt_10), np.sqrt(loss_lt_50), np.sqrt(loss_lt_90)])
                    return_list.extend(loss_pheno)
                    #trial_loss_dict[ph_w] = return_list
                    loss_dict[trial] = return_list#trial_loss_dict
                else:
                    loss_dict[trial][cultivar] = {}
                    for c_id in range(no_of_cultivars):
                        cultivar_id_torch = (torch.ones((x.shape[0], x.shape[1]))*c_id).type(torch.LongTensor).to(args.device)
                        if args.experiment == 'pheno_embedding':
                            out_lt_10, out_lt_50, out_lt_90, out_ph, _, _ = model(x_torch, cultivar_label=cultivar_id_torch)
                        else:
                            out_lt_10, out_lt_50, out_lt_90, out_ph, _ = model(x_torch, cultivar_label=cultivar_id_torch)
                        
                        loss_dict[trial][cultivar][c_id] = {}
                        
                        loss_dict[trial][cultivar][c_id]["lte50_preds"] = out_lt_50.numpy(force=True)
                        out_lt_10[:,:,0][nan_locs_lt_10] = 0
                        #out_lt_50[:,:,0][nan_locs_lt_50] = 0
                        out_lt_90[:,:,0][nan_locs_lt_90] = 0
                        
                        if args.setting == 'baseline_each':
                            for b in range(2):
                                num_phenos = out_ph.shape[0]
                                pheno_list = np.zeros(3*num_phenos+1)
                                for j in range(num_phenos):
                                    if len(torch.where(out_ph[j,b,:,0]>0.5)[0] != 0):
                                        pheno_list[j] = torch.where(out_ph[j,b,:,0]>0.5)[0][0].item()
                                    else:
                                        pheno_list[j] = -1
                                    pheno_list[j+num_phenos] = torch.where(y[b,:,j+3]==1)[0][0].item()
                                    nan_locs_pheno = y_torch[b, :, j+3].isnan()
                                    pheno_list[2*num_phenos+j] = (criterion2(out_ph[j,b,:,0], y_torch[b,:,j+3])[~nan_locs_pheno]).mean().item()
                                pheno_list[3*num_phenos] = np.sqrt(criterion(out_lt_50[b,:,0], y_torch[b, :, 1])[~nan_locs_lt_50[b]].mean().item())
                                loss_dict[trial][cultivar][c_id][b] = pheno_list
                            #loss_dict[trial][cultivar][c_id]["embed"] = concat_weights[c_id,:].numpy(force=True) 
                        
                        else:
                            for b in range(2):
                                
                                num_phenos = out_ph.shape[0]
                                pheno_list = np.zeros(2*num_phenos+2)
                                for j in range(num_phenos):
                                    if len(torch.where(out_ph[j,b,:,0]>0.5)[0] != 0):
                                        pheno_list[j] = torch.where(out_ph[j,b,:,0]>0.5)[0][0].item()
                                    else:
                                        pheno_list[j] = -1
                                    pheno_list[j+num_phenos] = torch.where(y[b,:,j+3]==1)[0][0].item()
                                    nan_locs_pheno = y_torch[b, :, j+3].isnan()
                                    pheno_list[2*num_phenos+1] += (criterion2(out_ph[j,b,:,0], y_torch[b,:,j+3])[~nan_locs_pheno]).mean().item()
                                pheno_list[2*num_phenos] = np.sqrt(criterion(out_lt_50[b,:,0], y_torch[b, :, 1])[~nan_locs_lt_50[b]].mean().item())
                                loss_dict[trial][cultivar][c_id][b] = pheno_list
                            loss_dict[trial][cultivar][c_id]["embed"] = concat_weights[c_id,:].numpy(force=True)    
                            
                            #out_lt_10[:,:,0][nan_locs_lt_10] = 0
                            #out_lt_50[:,:,0][nan_locs_lt_50] = 0
                            #out_lt_90[:,:,0][nan_locs_lt_90] = 0
                            
                            '''
                            loss = criterion(out_lt_50[:,:,0], y_torch[:, :, 1])
                            print(loss)
                            print(~nan_locs_lt_50)
                            print(loss[~nan_locs_lt_50])
                            print(loss[~nan_locs_lt_50].mean())
                            print(loss[~nan_locs_lt_50].mean().item())
                            '''
                            
                            loss_lt_10 = criterion(out_lt_10[:,:,0], y_torch[:, :, 0])[~nan_locs_lt_10].mean().item()  # LT10 GT
                            loss_lt_50 = criterion(out_lt_50[:,:,0], y_torch[:, :, 1])[~nan_locs_lt_50].mean().item()
                            loss_lt_90 = criterion(out_lt_90[:,:,0], y_torch[:, :, 2])[~nan_locs_lt_90].mean().item()
                            loss_pheno = []
                            for j in range(out_ph.shape[0]):
                                nan_locs_pheno = y_torch[:, :, j+3].isnan()
                                loss_pheno.append((criterion2(out_ph[j,:,:,0], y_torch[:,:,j+3])[~nan_locs_pheno]).mean().item())
                            #print(out_ph)
                            #print(loss_pheno)

                            return_list = list([np.sqrt(loss_lt_10), np.sqrt(loss_lt_50), np.sqrt(loss_lt_90)])
                            return_list.extend(loss_pheno)
                            loss_dict[trial][cultivar][c_id]["losses"] = return_list
                            loss_dict[trial][cultivar][c_id]["pheno_preds"] = out_ph.numpy(force=True)
                            #loss_dict[trial][c_id]["lte50_preds"] = out_lt_50.numpy(force=True)
                            #print(cultivar, return_list)
    
    return loss_dict


def run_cultivar_train(args, left_out_idx = None, finetune=False, embedding = None):
    dataset = args.dataset
    feature_len = dataset['train']['x'].shape[-1]
    no_of_cultivars = args.no_of_cultivars
    if finetune:
        model = args.nn_model_finetune(feature_len, no_of_cultivars, len(args.phenos_list), nonlinear = args.nonlinear)
    else:
        model = args.nn_model(feature_len, no_of_cultivars, len(args.phenos_list), nonlinear = args.nonlinear)
    if args.unfreeze=='yes':
        for param in model.parameters():
            param.requires_grad = True
    if finetune:
        model.load_state_dict(torch.load(args.pretrained_path), strict=False)
    model.to(args.device)
    trainable_params = sum([np.prod(p.size()) for p in filter(
        lambda p: p.requires_grad, model.parameters())])
    print("Trainable Parameters:", trainable_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss(reduction='none')
    criterion.to(args.device)
    #criterion2 = nn.BCELoss(reduction='none')
    #criterion2.to(args.device)
    if finetune:
        log_dir = os.path.join('./tensorboard/',args.name, args.experiment+'_setting_'+args.setting+'_variant_'+args.variant+'_weighting_'+args.weighting+'_unfreeze_'+args.unfreeze+'_nonlinear_'+args.nonlinear+'_scratch_'+args.scratch+"_finetune", args.trial, args.current_cultivar)
    else:
        log_dir = os.path.join('./tensorboard/',args.name, args.experiment+'_setting_'+args.setting+'_variant_'+args.variant+'_weighting_'+args.weighting+'_unfreeze_'+args.unfreeze+'_nonlinear_'+args.nonlinear+'_scratch_'+args.scratch, args.trial, args.current_cultivar)
    writer = SummaryWriter(log_dir)
    train_dataset = MyDataset(dataset['train'])
    trainLoader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = MyDataset(dataset['test'])
    valLoader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    

    from nn.models import pheno_embedding_net as emb_net
    if left_out_idx is not None:
        all_model = emb_net(feature_len, no_of_cultivars+1, len(args.phenos_list))
    else:
        all_model = emb_net(feature_len, no_of_cultivars, len(args.phenos_list))
    #emb_path = ".\models\embed_allcultivar_072423\\both_model.pt" #TODO: get the right file working here!!!
    #all_model.load_state_dict(torch.load(emb_path, map_location=torch.device('cpu')), strict=False)
    all_model.load_state_dict(torch.load(args.embedding_path, map_location=args.device), strict=False)
    embeds = all_model.embedding.weight
    embeds_torch = embeds.to(args.device)
 
    for epoch in range(args.epochs):
        # Training Loop
        model.train()
        total_loss = 0
        count = 0
        
        for i, (x, y, cultivar_id, freq) in enumerate(trainLoader):
            x_torch = x.to(args.device)
            y_torch = y.to(args.device)
            #adjust id if holding a cultivar out
            batch_size = x.shape[0]
            #print(batch_size)
            if left_out_idx is not None:
                for b in range(batch_size):
                    if cultivar_id[b,0] >= left_out_idx:
                        cultivar_id[b,0] += 1
            cultivar_id_torch = cultivar_id.to(args.device)
            count += 1
            no_phenos = y_torch.shape[2]-3 #subtract lte from count
            out_embed, _ = model(x_torch, y_torch[:,:,3:3+no_phenos], cultivar_label=cultivar_id_torch)
            optimizer.zero_grad()       # zero the parameter gradients
            
            #print(out_embed.shape) #batch size, day, embed dim
            #print(out_embed)
            #print(cultivar_id) #batch size, day 
            
            #could do loss on all days or just the last one
            #print(weights[0,:].repeat(args.batch_size, 1))
            if args.experiment in ['basic_cultivar']:
                out_embed = out_embed[:,-1,:]
            loss_embed = criterion(out_embed, embeds_torch[cultivar_id_torch[:,0],:])
            
            loss = loss_embed.mean()
            loss.backward()             # backward +
            optimizer.step()            # optimize
            total_loss += loss.item()
            
        writer.add_scalar('Train_Loss', total_loss / count, epoch)
        # Validation Loop
        with torch.no_grad():
            model.eval()
            total_loss = 0
            count = 0
            
            for i, (x, y, cultivar_id, freq) in enumerate(valLoader):
                x_torch = x.to(args.device)
                y_torch = y.to(args.device)
                #adjust id if holding a cultivar out
                batch_size = x.shape[0]
                #print(batch_size)
                if left_out_idx is not None:
                    for b in range(batch_size):
                        if cultivar_id[b,0] >= left_out_idx:
                            cultivar_id[b,0] += 1
                cultivar_id_torch = cultivar_id.to(args.device)
                count += 1
                no_phenos = y_torch.shape[2]-3 #subtract lte from count
                out_embed, _ = model(x_torch, y_torch[:,:,3:3+no_phenos], cultivar_label=cultivar_id_torch)
                #could do loss on all days or just the last one
                if args.experiment in ['basic_cultivar']:
                    out_embed = out_embed[:,-1,:]
                loss_embed = criterion(out_embed, embeds_torch[cultivar_id_torch[:,0],:])
                
                loss = loss_embed.mean()
                total_loss += loss.item()
            writer.add_scalar('Val_Loss', total_loss / count, epoch)
    loss_dict = dict()
    modelSavePath = "./models/"
    Path(os.path.join(modelSavePath, args.name, args.current_cultivar, args.trial)).mkdir(parents=True, exist_ok=True)
    if finetune:
        torch.save(model.state_dict(), os.path.join('./models', args.name, args.current_cultivar, args.trial, args.experiment+'_setting_'+args.setting+'_variant_'+args.variant+'_weighting_'+args.weighting+'_unfreeze_'+args.unfreeze+'_nonlinear_'+args.nonlinear+'_scratch_'+args.scratch+"_finetune.pt"))
    else:
        dir_path = os.path.join('./models', args.name, args.current_cultivar, args.trial)
        if (not os.path.exists(dir_path)):
            os.makedirs(dir_path)
        torch.save(model.state_dict(), os.path.join('./models', args.name, args.current_cultivar, args.trial, args.experiment+'_setting_'+args.setting+'_variant_'+args.variant+'_weighting_'+args.weighting+'_unfreeze_'+args.unfreeze+'_nonlinear_'+args.nonlinear+'_scratch_'+args.scratch+".pt"))
    # Validation Loop
    total_loss = 0
    with torch.no_grad():
        model.eval()
        for i, ((x, y, cultivar_id, freq), cultivar) in enumerate(zip(valLoader,args.cultivar_list)):
            x_torch = x.to(args.device)
            y_torch = y.to(args.device)
            #adjust id if holding a cultivar out
            batch_size = x.shape[0]
            #print(batch_size)
            if left_out_idx is not None:
                for b in range(batch_size):
                    if cultivar_id[b,0] >= left_out_idx:
                        cultivar_id[b,0] += 1
            cultivar_id_torch = cultivar_id.to(args.device)
            
            no_phenos = y_torch.shape[2]-3 #subtract lte from count
            out_embed, _ = model(x_torch, y_torch[:,:,3:3+no_phenos], cultivar_label=cultivar_id_torch)
            #could do loss on all days or just the last one
            if args.experiment in ['basic_cultivar']:
                out_embed = out_embed[:,-1,:]
            loss_embed = criterion(out_embed, embeds_torch[cultivar_id_torch[:,0],:])
            
            loss = loss_embed.mean().item()
            total_loss += loss
            loss_dict[cultivar] = np.sqrt(loss)
    loss_dict['overall'] = np.sqrt(total_loss)
    
    return loss_dict

def run_cultivar_eval(args, left_out_idx = None, finetune=False):
    dataset = args.dataset
    feature_len = dataset['train']['x'].shape[-1]
    no_of_cultivars = args.no_of_cultivars
    if left_out_idx is not None:
        model = args.nn_model(feature_len, no_of_cultivars+1, len(args.phenos_list), nonlinear = args.nonlinear)
    else:
        model = args.nn_model(feature_len, no_of_cultivars, len(args.phenos_list), nonlinear = args.nonlinear)
    model.load_state_dict(torch.load(os.path.join('./models', args.name, args.current_cultivar, args.trial, args.experiment+'_setting_'+args.setting+'_variant_'+args.variant+'_weighting_'+args.weighting+'_unfreeze_'+args.unfreeze+'_nonlinear_'+args.nonlinear+'_scratch_'+args.scratch+".pt")), strict=False)
    
    model.to(args.device)
    trainable_params = sum([np.prod(p.size()) for p in filter(
        lambda p: p.requires_grad, model.parameters())])
    print("Trainable Parameters:", trainable_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss(reduction='none')
    criterion.to(args.device)
    train_dataset = MyDataset(dataset['train'])
    trainLoader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = MyDataset(dataset['test'])
    valLoader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    
    from nn.models import pheno_embedding_net as emb_net
    if left_out_idx is not None:
        all_model = emb_net(feature_len, no_of_cultivars+1, len(args.phenos_list))
    else:
        all_model = emb_net(feature_len, no_of_cultivars, len(args.phenos_list))
    #emb_path = ".\models\embed_allcultivar_072423\\both_model.pt"
    #all_model.load_state_dict(torch.load(emb_path, map_location=torch.device('cpu')), strict=False)
    all_model.load_state_dict(torch.load(args.embedding_path, map_location=args.device), strict=False)
    embeds = all_model.embedding.weight
    embeds_torch = embeds.to(args.device)
    
    total_loss = 0
    loss_dict = dict()
    with torch.no_grad():
        model.eval()
        for i, ((x, y, cultivar_id, freq), cultivar) in enumerate(zip(valLoader,args.cultivar_list)):
            x_torch = x.to(args.device)
            y_torch = y.to(args.device)
            #adjust id if holding a cultivar out
            batch_size = x.shape[0]
            #print(batch_size)
            if left_out_idx is not None:
                for b in range(batch_size):
                    cultivar_id[b,0] = left_out_idx
            cultivar_id_torch = cultivar_id.to(args.device)
            
            no_phenos = y_torch.shape[2]-3 #subtract lte from count
            out_embed, _ = model(x_torch, y_torch[:,:,3:3+no_phenos], cultivar_label=cultivar_id_torch)
            if args.experiment in ['basic_cultivar']:
                out_embed = out_embed[:,-1,:]
            print("Pred: ",out_embed)
            print("Actual: ",embeds_torch[cultivar_id_torch[:,0],:])
            #could do loss on all days or just the last one
            loss_embed = criterion(out_embed, embeds_torch[cultivar_id_torch[:,0],:])
            
            loss = loss_embed.mean().item()
            total_loss += loss
            loss_dict[cultivar] = dict()
            loss_dict[cultivar]["loss"] = np.sqrt(loss)
            loss_dict[cultivar]["pred"] = out_embed.detach().cpu().numpy()
            loss_dict[cultivar]["actual"] = embeds_torch[cultivar_id_torch[:,0],:].detach().cpu().numpy()
    loss_dict['overall'] = np.sqrt(total_loss)
    return loss_dict
    
def run_state_vectors_eval(args, cultivar_id, finetune=False):
    dataset = args.dataset
    feature_len = dataset['train']['x'].shape[-1]
    no_of_cultivars = args.no_of_cultivars
    model = args.nn_model(feature_len, no_of_cultivars, args.nonlinear)
    #model.load_state_dict(torch.load('./eval/mtl_all.pt'))
    #model.load_state_dict(torch.load('./eval/single/'+args.current_cultivar+'/'+args.trial+'/single.pt'))
    #model.load_state_dict(torch.load('./models/15_Nov_2022_10_54_08/Riesling/trial_0/single.pt', map_location=torch.device('cpu')))
    model.load_state_dict(torch.load(args.pretrained_path, map_location=args.device))
    model.to(args.device)
    
    train_dataset = MyDataset(dataset['train'])
    val_dataset = MyDataset(dataset['test'])
    trainLoader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    valLoader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    internal_dict = dict()
    state_dict = dict()
    LTE_dict = dict()
    #cultivar_id_torch = cultivar_id.to(args.device)
    #cultivar_id_torch = cultivar_id
    # Validation Loop
    with torch.no_grad():
        model.eval()
        count = 0
        for i, (x, y, cultivar_label, freq) in enumerate(trainLoader):
            x_torch = x.to(args.device)
            y_torch = y.to(args.device)
            for c1 in range(cultivar_label.shape[0]):
                for c2 in range(cultivar_label.shape[1]):
                    cultivar_label[c1,c2] = cultivar_id
            #cultivar_id_torch = (torch.ones((x.shape[0], x.shape[1], 1))*cultivar_id).to(args.device)
            cultivar_id_torch = cultivar_label.to(args.device)
            #print(cultivar_label.shape)
            #print(y_torch[1,:,1])
            #cultivar_id_torch = cultivar_id.to(args.device)
            out_lt_10, out_lt_50, out_lt_90, out_ph, _, int_vector, state_vector = model(x_torch, cultivar_label=cultivar_id_torch)
            print(int_vector.shape)
            print(state_vector.shape)
            print(out_lt_10.shape)
            internal_dict[str(count)] = int_vector.detach().cpu().numpy()
            state_dict[str(count)] = state_vector.detach().cpu().numpy()
            LTE_dict[str(count)] = {'10': out_lt_10.detach().cpu().numpy()[0,:,0], '50': out_lt_50.detach().cpu().numpy()[0,:,0], '90': out_lt_90.detach().cpu().numpy()[0,:,0]}
            
            
            count += 1
        for i, (x, y, cultivar_label, freq) in enumerate(valLoader):
            x_torch = x.to(args.device)
            y_torch = y.to(args.device)
            #print(y_torch[1,:,1])
            #cultivar_id_torch = cultivar_id.to(args.device)
            out_lt_10, out_lt_50, out_lt_90, out_ph, _, int_vector, state_vector = model(x_torch, cultivar_label=cultivar_id_torch)
            internal_dict[str(count)] = int_vector.detach().cpu().numpy()
            state_dict[str(count)] = state_vector.detach().cpu().numpy()
            LTE_dict[str(count)] = {'10': out_lt_10.detach().cpu().numpy()[0,:,0], '50': out_lt_50.detach().cpu().numpy()[0,:,0], '90': out_lt_90.detach().cpu().numpy()[0,:,0]}
            count += 1
    print(len(internal_dict))        
    return internal_dict, state_dict, LTE_dict