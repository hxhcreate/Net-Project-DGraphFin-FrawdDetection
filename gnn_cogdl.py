# dataset name: DGraphFin

from utils import Dgraph_Dataloader
from utils.utils import prepare_folder
from utils.evaluator import Evaluator
from logger import Logger
from cogdl.data import Graph
from cogdl.datasets import NodeDataset
from models import SIGN

import argparse

import torch
import torch.nn.functional as F
import torch.nn as nn

import pandas as pd
import torch
from cogdl.data import Graph
import numpy as np

def mask_change(id_mask, node_size):
    mask = torch.zeros(node_size).bool()
    for i in id_mask:
        mask[i] = True
    return mask

eval_metric = 'auc'

sign_parameters = {'lr': 0.01
    , 'num_layers': 2
    , 'hidden_size': 128
    , 'dropout': 0.0
    , 'dropedge_rate': 0.2
    , 'nhop': 3
    , 'l2': 5e-7
    }



def train(model, data, train_idx, optimizer, no_conv=False):
    # data.y is labels of shape (N, )
    model.train()

    optimizer.zero_grad()
    if no_conv:
        out = model(data.x[train_idx])
    else:
        out = model(data)[train_idx]
    loss = F.nll_loss(out, data.y[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator, no_conv=False):
    # data.y is labels of shape (N, )
    model.eval()

    if no_conv:
        out = model(data.x)
    else:
        out = model(data)

    #y_pred = F.log_softmax(out, dim=-1)
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
    parser.add_argument('--model', type=str, default='mlp')
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
    datapath = 'cogdl_dataset/dgraphfin.npz'

    x,edge_index,y,train_mask,valid_mask,test_mask = Dgraph_Dataloader(datapath)
    #x,edge_index,y,train_mask,valid_mask,test_mask = res[0],res[1],res[2],res[3],res[4],res[5]

    data = Graph(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=valid_mask, test_mask=test_mask)

    nlabels = 2
    if args.dataset in ['DGraphFin']: nlabels = 2

    # data.adj_t = data.adj_t.to_symmetric()

    if args.dataset in ['DGraphFin']:
        x = data.x
        x = (x - x.mean(0)) / x.std(0)
        data.x = x
    if data.y.dim() == 2:
        data.y = data.y.squeeze(1)

    split_idx = {'train': data.train_mask, 'valid': data.val_mask, 'test': data.test_mask}

    fold = args.fold
    if split_idx['train'].dim() > 1 and split_idx['train'].shape[1] > 1:
        kfolds = True
        print('There are {} folds of splits'.format(split_idx['train'].shape[1]))
        split_idx['train'] = split_idx['train'][:, fold]
        split_idx['valid'] = split_idx['valid'][:, fold]
        split_idx['test'] = split_idx['test'][:, fold]
    else:
        kfolds = False

    data = data.to(device)
    train_idx = split_idx['train'].to(device)

    result_dir = prepare_folder(args.dataset, args.model)
    print('result_dir:', result_dir)
    if args.model == 'sign':
        para_dict = sign_parameters
        model_para = sign_parameters.copy()
        model_para.pop('lr')
        model_para.pop('l2')
        model = SIGN(num_features=data.x.size(-1), num_classes=nlabels, **model_para).to(device)
    print(f'Model {args.model} initialized')

    evaluator = Evaluator(eval_metric)
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        import gc
        gc.collect()
        print(sum(p.numel() for p in model.parameters()))

        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=para_dict['lr'], weight_decay=para_dict['l2'])
        best_valid = 0
        min_valid_loss = 1e8
        best_out = None

        for epoch in range(1, args.epochs + 1):
            loss = train(model, data, train_idx, optimizer, no_conv)
            eval_results, losses, out = test(model, data, split_idx, evaluator, no_conv)
            train_eval, valid_eval, test_eval = eval_results['train'], eval_results['valid'], eval_results['test']
            train_loss, valid_loss, test_loss = losses['train'], losses['valid'], losses['test']

            #                 if valid_eval > best_valid:
            #                     best_valid = valid_result
            #                     best_out = out.cpu().exp()
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
    pd.DataFrame(para_dict, index=[args.model]).to_csv(result_dir + '/results.csv')


if __name__ == "__main__":
    main()