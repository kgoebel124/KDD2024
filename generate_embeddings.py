import torch
import os
import numpy as np
from nn.models import pheno_embedding_net as emb_net
import random
import pickle


def rand_constrained(cultivars, model_path, file_name):
    #get learned embeddings
    feature_len = 12
    no_of_cultivars = 17
    no_new_cultivars = 100#272
    no_of_phenos = 4
    all_embeds = {}
    
    for cultivar in cultivars:
        model = emb_net(feature_len, no_of_cultivars, no_of_phenos)
        
        path = model_path + cultivar + file_name
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')), strict=False)
        #print(model.embedding.weight.shape)
        weights = model.embedding.weight

        #get constraints
        maxes = np.empty((feature_len))
        mins = np.empty((feature_len))
        for f in range(feature_len):
            #get max and min
            maxes[f] = torch.max(weights[:,f])
            mins[f] = torch.min(weights[:,f])
        #print(cultivar)
        #print("maxes: ", maxes)
        #print("mins: ", mins)
        
        #generate random embedding in the constraints
        new_embs = torch.empty((no_new_cultivars, feature_len), dtype=torch.float32)
        for c in range(no_new_cultivars):
            for f in range(feature_len):
                new_embs[c,f] = random.uniform(mins[f], maxes[f])
        
        #save embeddings
        all_embeds[cultivar] = new_embs
    #save
    with open(os.path.join('.\\models', 'extra_embeds',"uniform_" + str(no_new_cultivars) + ".pkl"), 'wb') as f:
        pickle.dump(all_embeds, f)
        
        
def rand_lin_comb(cultivars, model_path, file_name):
    #get learned embeddings
    feature_len = 12
    no_of_cultivars = 17
    no_new_cultivars = 272
    no_of_phenos = 4
    all_embeds = {}
    num_averaged = 17
    
    for cultivar in cultivars:
        model = emb_net(feature_len, no_of_cultivars, no_of_phenos)
        
        path = model_path + cultivar + file_name
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')), strict=False)
        #print(model.embedding.weight.shape)
        weights = model.embedding.weight

        
        
        #generate random embedding in the constraints
        new_embs = torch.empty((no_new_cultivars, feature_len), dtype=torch.float32)
        for c in range(no_new_cultivars):
            #pick three embeddings
            #chosen_embeds = random.choices(range(no_of_cultivars),k=num_averaged)
            chosen_embeds = range(no_of_cultivars)
            #pick three rand weights
            weighting = np.random.uniform(0,1,num_averaged)
            #normalize weights
            total = sum(weighting)
            weighting = [w/total for w in weighting]
            
            for f in range(feature_len):
                new_embs[c,f] = sum([w*e for (w,e) in zip(weighting, weights[chosen_embeds,f])])#random.uniform(mins[f], maxes[f])
        
        #save embeddings
        all_embeds[cultivar] = new_embs
    #save
    
    with open(os.path.join('.\\models', 'extra_embeds',"lincomb_17_" + str(no_new_cultivars) + ".pkl"), 'wb') as f:
        pickle.dump(all_embeds, f)
    
def rand_convex_comb(cultivars, model_path, file_name):
    #get learned embeddings
    feature_len = 12
    no_of_cultivars = 17
    
    no_new_cultivars = 19
    no_of_phenos = 4
    
    all_embeds = {}
    
    for cultivar in cultivars:
        model = emb_net(feature_len, no_of_cultivars, no_of_phenos)
        
        path = model_path + cultivar + file_name
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')), strict=False)
        #print(model.embedding.weight.shape)
        weights = model.embedding.weight

        #generate random embedding in the constraints
        new_embs = torch.empty((no_new_cultivars + 2, feature_len), dtype=torch.float32)
        chosen_embeds = random.choices(range(no_of_cultivars),k=2)
        while (chosen_embeds[0] == chosen_embeds[1]):
            chosen_embeds[1] = random.choices(range(no_of_cultivars),k=1)
        
        new_embs[0,:] = weights[chosen_embeds[1],:]
        new_embs[-1,:] = weights[chosen_embeds[0],:]
        for c in range(1,no_new_cultivars+1):
            theta = c/20
            for f in range(feature_len):
                new_embs[c,f] = theta*weights[chosen_embeds[0],f] + (1-theta)*weights[chosen_embeds[1],f]
        #print(new_embs)
        #save embeddings
        all_embeds[cultivar] = new_embs
    #save
    #print(all_embeds['Barbera'])
    with open(os.path.join('.\\models', 'extra_embeds',"convex_comb_trial_2_" + str(no_new_cultivars) + ".pkl"), 'wb') as f:
        pickle.dump(all_embeds, f)

def print_embs(cultivar, model_path, file_name, extra_file):
    feature_len = 12
    no_of_cultivars = 17
    
    no_new_cultivars = 19
    no_of_phenos = 4
    model = emb_net(feature_len, no_of_cultivars, no_of_phenos)
        
    path = model_path + cultivar + file_name
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')), strict=False)
    #print(model.embedding.weight.shape)
    weights = model.embedding.weight
    
    #open extra file
    with open(extra_file,'rb') as f:
        embeds = pickle.load(f)['Zinfandel']
    
    
    print("good embs:")
    idx = [3, 13, 19, 26, 32, 35, 37, 39, 44, 50, 62, 63, 69, 82]
    #print("bad embs:")
    #idx = [6,12, 17,24,27,28, 30, 40,52,55,64,71,76,77, 83]
    #print("weird emb:")
    #idx = [21]
    
    for i in idx:
        if (i < 17):
            weights[i,:]
        else:
            print(embeds[i-17,:])
    
    
    


#print("call fxns here")
valid_cultivars = ['Zinfandel','Concord','Malbec','Barbera','Semillon','Merlot','Chenin Blanc','Riesling','Nebbiolo',
                   'Cabernet Sauvignon','Chardonnay','Viognier','Mourvedre','Pinot Gris','Grenache','Syrah','Sangiovese','Sauvignon Blanc']
valid_cultivars.sort()

path = "C:\\MyWork\\OSU\\Research\\Grape_Pheno\\models\\emb_models063023\\"
file_name = "\\trial_0\\pheno_embedding_setting_leaveoneout_variant_none_weighting_none_unfreeze_no_nonlinear_no_scratch_no.pt"
rand_constrained(valid_cultivars, path, file_name)
#rand_lin_comb(valid_cultivars, path, file_name)
#rand_convex_comb(valid_cultivars, path, file_name)
#print_embs('Zinfandel', path, file_name,os.path.join('.\\models', 'extra_embeds',"uniform_68.pkl"))
