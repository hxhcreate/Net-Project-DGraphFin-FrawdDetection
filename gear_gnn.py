# dataset name: DGraphFin
import random
from utils.gear_dgraphfin import GEARDGraphFin
from utils.utils import prepare_folder
from utils.evaluator import Evaluator
from models import MLP, MLPLinear, GCN, SAGE, GAT, GATv2, SIGN
from utils.logger import Logger
import numpy as np
from torch_geometric.utils.subgraph import k_hop_subgraph

from utils.feat_func import data_process
from models import GEARSage
from utils.gear_dgraphfin import GEARDGraphFin
from utils.evaluator import Evaluator
from utils.utils import prepare_folder
import argparse

import torch
import torch.nn.functional as F
import torch.nn as nn

import torch_geometric.transforms as T
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
import pandas as pd

eval_metric = 'auc'

gear_parameters = {'lr': 0.01
                , 'num_layers': 3
                , 'hidden_size': 96
                , 'dropout': 0.3
                , 'nhop': 3
                , 'l2': 0.0
}

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train(model, data, optimizer):
    # data.y is labels of shape (N, ) 
    model.train()
    optimizer.zero_grad()
    neg_idx = data.train_neg[
        torch.randperm(data.train_neg.size(0))[: data.train_pos.size(0)]
    ]
    train_idx = torch.cat([data.train_pos, neg_idx], dim=0)
    nodeandneighbor, edge_index, node_map, mask = k_hop_subgraph(
        train_idx, 3, data.edge_index, relabel_nodes=True, num_nodes=data.x.size(0)
    )
    out = model(
        data.x[nodeandneighbor],
        edge_index,
        data.edge_attr[mask],
        data.edge_timestamp[mask],
        data.edge_direct[mask],
    )
    loss = F.nll_loss(out[node_map], data.y[train_idx])
    loss.backward()
    
    nn.utils.clip_grad_norm_(model.parameters(), 2.0)
    
    optimizer.step()
    torch.cuda.empty_cache()
    return loss.item()

    


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    # data.y is labels of shape (N, )
    model.eval()
    out = model(
        data.x, data.edge_index, data.edge_attr, data.edge_timestamp, data.edge_direct,
    )
        
    y_pred = out.exp()  # (N,num_classes)
    
    losses, eval_results = dict(), dict()
    for key in ['train', 'valid', 'test']:
        node_id = split_idx[key]
        losses[key] = F.nll_loss(out[node_id], data.y[node_id]).item()
        eval_results[key] = evaluator.eval(data.y[node_id], y_pred[node_id])[eval_metric]
            
    return eval_results, losses, y_pred
        
            
def main():
    parser = argparse.ArgumentParser(description='gnn_models')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='DGraphFin')
    parser.add_argument('--log_steps', type=int, default=10)
    parser.add_argument('--model', type=str, default='gear')
    parser.add_argument('--use_embeddings', action='store_true')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--fold', type=int, default=0)
    
    args = parser.parse_args()
    print(args)
    
    no_conv = False
    if args.model in ['mlp']: no_conv = True        
    
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    # model_dir = prepare_folder(args.dataset, args.model)  
    # print("model_dir:", model_dir)
    
    set_seed(42)
    dataset = GEARDGraphFin(root="./dataset", name="DGraphFin")
    data = dataset[0]
    nlabels = 2     
    
    split_idx = {
        "train": data.train_mask,
        "valid": data.valid_mask,
        "test": data.test_mask,
    }
    fold = args.fold
    if split_idx['train'].dim()>1 and split_idx['train'].shape[1] >1:
        print('There are {} folds of splits'.format(split_idx['train'].shape[1]))
        split_idx['train'] = split_idx['train'][:, fold]
        split_idx['valid'] = split_idx['valid'][:, fold]
        split_idx['test'] = split_idx['test'][:, fold]
    data = data_process(data).to(device)
    train_idx = split_idx["train"].to(device)
    
    data.train_pos = train_idx[data.y[train_idx] == 1]
    data.train_neg = train_idx[data.y[train_idx] == 0]
    
    
            
    result_dir = prepare_folder(args.dataset, args.model)
    print('result_dir:', result_dir)
        
    if args.model == "gear":
        para_dict = gear_parameters
        model_para = gear_parameters.copy()
        model_para.pop('lr')
        model_para.pop('l2')   
        model = GEARSage(
            in_channels=data.x.size(-1),
            hidden_channels=gear_parameters["hidden_size"],
            out_channels=nlabels,
            num_layers=gear_parameters["num_layers"],
            dropout=gear_parameters["dropout"],
            activation="elu",
            bn=True,
        ).to(device)
    
    print(f'Model {args.model} initialized')

    evaluator = Evaluator(eval_metric)
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        import gc
        gc.collect()
        print(sum(p.numel() for p in model.parameters()))

        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0)
        best_valid = 0
        min_valid_loss = 1e8
        best_out = None

        for epoch in range(1, args.epochs+1):
            loss = train(model, data, optimizer)
            eval_results, losses, out = test(model, data, split_idx, evaluator)
            train_eval, valid_eval, test_eval = eval_results['train'], eval_results['valid'], eval_results['test']
            train_loss, valid_loss, test_loss = losses['train'], losses['valid'], losses['test']

            if valid_loss < min_valid_loss:
                min_valid_loss = valid_loss
                best_out = out.cpu()
            
            if epoch % args.log_steps == 0:
                print(f'Run: {run + 1:02d}, '
                          f'Epoch: {epoch:02d}, '
                          f'Loss: {loss:.4f}, '
                          f'Train: {100 * train_eval:.3f}%, '
                          f'Valid: {100 * valid_eval:.3f}% '
                          f'Test: {100 * test_eval:.3f}%')
            logger.add_result(run, [train_eval, valid_eval, test_eval])

        logger.print_statistics(run)

    final_results = logger.print_statistics()
    print('final_results:', final_results)
    para_dict.update(final_results)
    pd.DataFrame(para_dict, index=[args.model]).to_csv(result_dir+'/results.csv')


if __name__ == "__main__":
    main()
