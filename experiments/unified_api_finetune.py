import torch
from util.create_dataset import MyDataset, get_not_nan
from util.data_processing import evaluate
from nn.models import embedding_net_finetune
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.nn as nn
import os
from pathlib import Path


def run_experiment(args, finetune=False):
    dataset = args.dataset
    feature_len = dataset['train']['x'].shape[-1]
    no_of_cultivars = args.no_of_cultivars
    model = args.nn_model_finetune(feature_len, no_of_cultivars)
    model.to(args.device)
    model.load_state_dict(torch.load(args.pretrained_path), strict=False)
    trainable_params = sum([np.prod(p.size()) for p in filter(
        lambda p: p.requires_grad, model.parameters())])
    print("Trainable Parameters:", trainable_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss(reduction='none')
    criterion.to(args.device)
    log_dir = os.path.join('./tensorboard/',args.name, args.experiment+"_finetune", args.trial, args.current_cultivar)
    writer = SummaryWriter(log_dir)
    train_dataset = MyDataset(dataset['train'])
    trainLoader = DataLoader(train_dataset, batch_size=args.batch_size)
    val_dataset = MyDataset(dataset['test'])
    valLoader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    for epoch in range(args.epochs):
        # Training Loop
        model.train()
        total_loss = 0
        count = 0
        for i, (x, y, cultivar_id, freq) in enumerate(trainLoader):
            x_torch = x.to(args.device)
            y_torch = y.to(args.device)
            freq = freq.to(args.device)
            cultivar_id_torch = cultivar_id.to(args.device)
            count += 1
            out_lt_10, out_lt_50, out_lt_90, out_ph, _ = model(x_torch, cultivar_label=cultivar_id_torch)

            optimizer.zero_grad()       # zero the parameter gradients

            loss_lt_10 = criterion(out_lt_10[:,:,0], y_torch[:, :, 0])  # LT10 GT
            loss_lt_50 = criterion(out_lt_50[:,:,0], y_torch[:, :, 1])
            loss_lt_90 = criterion(out_lt_90[:,:,0], y_torch[:, :, 2])
            if args.weighting=='inverse_freq':
                loss = torch.mul(loss_lt_10 + loss_lt_50 + loss_lt_90, freq).nanmean()
            elif args.weighting=='uncertainty':
                pass
            else:
                loss = (loss_lt_10 + loss_lt_50 + loss_lt_90).nanmean()
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
                cultivar_id_torch = cultivar_id.to(args.device)
                count += 1
                out_lt_10, out_lt_50, out_lt_90, out_ph, _ = model(x_torch, cultivar_label=cultivar_id_torch)
                # getting non nan values is slow right now due to copying to cpu, write pytorch gpu version
                loss_lt_10 = criterion(out_lt_10[:,:,0], y_torch[:, :, 0])  # LT10 GT
                loss_lt_50 = criterion(out_lt_50[:,:,0], y_torch[:, :, 1])
                loss_lt_90 = criterion(out_lt_90[:,:,0], y_torch[:, :, 2])
                loss = loss_lt_10 + loss_lt_50 + loss_lt_90
                total_loss += loss.nanmean().item()
            writer.add_scalar('Val_Loss', total_loss / count, epoch)
    loss_dict = dict()
    modelSavePath = "./models/"
    Path(os.path.join(modelSavePath, args.name, args.current_cultivar,args.trial)).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(modelSavePath, args.name, args.current_cultivar, args.trial, args.experiment+"_finetune.pt"))
    # Validation Loop
    total_loss_lt_10, total_loss_lt_50, total_loss_lt_90 = 0, 0, 0
    with torch.no_grad():
        model.eval()
        for i, ((x, y, cultivar_id, freq), cultivar) in enumerate(zip(valLoader,args.cultivar_list)):
            x_torch = x.to(args.device)
            y_torch = y.to(args.device)
            cultivar_id_torch = cultivar_id.to(args.device)
            out_lt_10, out_lt_50, out_lt_90, out_ph, _ = model(x_torch, cultivar_label=cultivar_id_torch)
            loss_lt_10 = criterion(out_lt_10[:,:,0], y_torch[:, :, 0]).mean().item()  # LT10 GT
            loss_lt_50 = criterion(out_lt_50[:,:,0], y_torch[:, :, 1]).mean().item()
            loss_lt_90 = criterion(out_lt_90[:,:,0], y_torch[:, :, 2]).mean().item()
            total_loss_lt_10+=loss_lt_10
            total_loss_lt_50+=loss_lt_50
            total_loss_lt_90+=loss_lt_90
            loss_dict[cultivar] = list([np.sqrt(loss_lt_10), np.sqrt(loss_lt_50), np.sqrt(loss_lt_90)])
    loss_dict['overall'] = list([np.sqrt(total_loss_lt_10), np.sqrt(total_loss_lt_50), np.sqrt(total_loss_lt_90)])
    return loss_dict