This repo provides a collection of baselines for DGraphFin dataset. Please download the dataset from the [DGraph](http://dgraph.xinye.com) web and place it under the folder './dataset/DGraphFin/raw'.  

## Project Description
- `model_results`: store the .csv file which include the final result for each model
- `models`: model file included by gnn.py
- `utils`: include the preprocess file / dataset loader / logger / Evaluator
- `output`: the log file to run each model


## Environments
Python environment:  
- pytorch = 1.13.0
- cudatookit = 11.6.1
- torch_geometric = 1.7.2  (when you are running GEARSAGE please use 2.0.4)
- torch_scatter = 2.1.0+pt113cu116
- torch_sparse = 0.6.15+pt113cu116
- cogdl = 0.5.3
- pyg-lib = 0.1.0+pt113cu116

GPU environment:
- GPU: RTX A5000 24G  * 1


## Training

- **GearSage**
```bash
python gear_gnn.py --model gear --dataset DGraphFin --epochs 200 --runs 10 --device 0
```

- **SIGN**
```bash
python gnn.py --model sign --dataset DGraphFin --epochs 200 --runs 10 --device 0
```

- **MLP**
```bash
python gnn.py --model mlp --dataset DGraphFin --epochs 200 --runs 10 --device 0
```

- **GCN**
```bash
python gnn.py --model gcn --dataset DGraphFin --epochs 200 --runs 10 --device 0
```

- **GraphSAGE**
```bash
python gnn.py --model sage --dataset DGraphFin --epochs 200 --runs 10 --device 0
```

- **GraphSAGE (NeighborSampler)**
```bash
python gnn_mini_batch.py --model sage_neighsampler --dataset DGraphFin --epochs 200 --runs 10 --device 0
```

- **GAT (NeighborSampler)**
```bash
python gnn_mini_batch.py --model gat_neighsampler --dataset DGraphFin --epochs 200 --runs 10 --device 0
```

- **GATv2 (NeighborSampler)**
```bash
python gnn_mini_batch.py --model gatv2_neighsampler --dataset DGraphFin --epochs 200 --runs 10 --device 0
```



## MyResults:
Performance on **DGraphFin**(10 runs) (%):
(ranked by test AUC )

| rk | Methods   | Train AUC  | Valid AUC  | Test AUC  |
|  :----  | ----  |  ---- | ---- | ---- |
| 1 | GEARSAGE | 84.7251 ± 0.0776 | 83.3331 ±  0.0747 | **84.1887 ± 0.0565** |
| 2 | GraphSAGE (NeighborSampler)  | 78.6245 ± 0.1391 | 76.8072 ± 0.08 | 77.6441 ± 0.1343 |
| 3 | SIGN | 77.2373 ± 0.2803 | 75.5652 ± 0.1840 | 76.9460 ± 0.3002 |
| 4 | GraphSAGE| 76.7854 ± 0.1881  | 75.4739 ± 0.1894 | 76.2051 ± 0.2010 |
| 5 | GATv2 (NeighborSampler)      | 76.3698 ± 0.7377 | 74.7529 ± 0.788 | 75.7034 ± 0.6571 |
| 6 | GAT (NeighborSampler)        | 74.2509 ± 0.3803 | 72.5287 ± 0.2654 | 73.6141 ± 0.3018 |
| 7 | MLP | 72.1234 ± 0.0912 | 71.2699 ± 0.0924 | 71.8815 ± 0.0858 |
| 8 | GCN | 71.0831 ± 0.3224 | 70.7958 ± 0.3028 | 70.7996 ± 0.2721 |








## Results from [Origin Repo](https://github.com/DGraphXinye/DGraphFin_baseline):
Performance on **DGraphFin**(10 runs):

| Methods   | Train AUC  | Valid AUC  | Test AUC  |
|  :----  | ----  |  ---- | ---- |
| MLP | 0.7221 ± 0.0014 | 0.7135 ± 0.0010 | 0.7192 ± 0.0009 |
| GCN | 0.7108 ± 0.0027 | 0.7078 ± 0.0027 | 0.7078 ± 0.0023 |
| GraphSAGE| 0.7682 ± 0.0014 | 0.7548 ± 0.0013 | 0.7621 ± 0.0017 |
| GraphSAGE (NeighborSampler)  | 0.7845 ± 0.0013 | 0.7674 ± 0.0005 | **0.7761 ± 0.0018** |
| GAT (NeighborSampler)        | 0.7396 ± 0.0018 | 0.7233 ± 0.0012 | 0.7333 ± 0.0024 |
| GATv2 (NeighborSampler)      | 0.7698 ± 0.0083 | 0.7526 ± 0.0089 | 0.7624 ± 0.0081 |


