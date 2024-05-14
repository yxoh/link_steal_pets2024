This is the official implementation for the paper: [Link Stealing Attacks Against Inductive Graph Neural Networks](https://arxiv.org/pdf/2405.05784).

## Setup
```
conda env create -f link_steal.yaml
conda activate link_steal

git clone https://github.com/EdisonLeeeee/GraphGallery.git && cd GraphGallery
pip install -e . --verbose
```

## Train target GNNs
```

python train_gnn.py --dataset lastfm --model graphsage --mode target --gpu 0
python train_gnn.py --dataset lastfm --model graphsage --mode shadow --gpu 0
```

## Attack

1. Attack-0

```
python mlp_attack.py --dataset lastfm --node_topology 0-hop --edge_feature all --target_model graphsage --shadow_model graphsage --lr 0.006 --optim adam --scheduler --gpu 0
```

2. Attack-1
```
python mlp_attack.py --dataset lastfm --node_topology 1-hop --edge_feature all --target_model graphsage --shadow_model graphsage --lr 0.006 --optim adam --scheduler --gpu 0
```

3. Attack-2
```
python mlp_attack.py --dataset lastfm --node_topology 2-hop --edge_feature all --target_model graphsage --shadow_model graphsage --lr 0.006 --optim adam --scheduler --gpu 0
```

4. Attack-3
```
python mlp_attack.py --dataset lastfm --node_topology 0-hop --edge_feature all --target_model graphsage --shadow_model graphsage --lr 0.006 --optim adam --scheduler --gpu 0 --plus
```


5. Attack-4
```
python mlp_attack.py --dataset lastfm --node_topology 1-hop --edge_feature all --target_model graphsage --shadow_model graphsage --lr 0.006 --optim adam --scheduler --gpu 0 --plus
```


6. Attack-5
```
python mlp_attack.py --dataset lastfm --node_topology 2-hop --edge_feature all --target_model graphsage --shadow_model graphsage --lr 0.006 --optim adam --scheduler --gpu 0 --plus
```


7. Attack-6
```
python mlp_attack.py --dataset lastfm --node_topology 1-hop --edge_feature all --target_model graphsage --shadow_model graphsage --lr 0.006 --optim adam --scheduler --gpu 0 --plus2
```


8. Attack-7
```
python mlp_attack.py --dataset lastfm --node_topology 2-hop --edge_feature all --target_model graphsage --shadow_model graphsage --lr 0.006 --optim adam --scheduler --gpu 0 --plus2
```


9. Attack-8
```
python mlp_attack.py --dataset lastfm --node_topology 1-hop --edge_feature all --target_model graphsage --shadow_model graphsage --lr 0.006 --optim adam --scheduler --gpu 0 --all
```

10. Attack-9
```
python mlp_attack.py --dataset lastfm --node_topology 2-hop --edge_feature all --target_model graphsage --shadow_model graphsage --lr 0.006 --optim adam --scheduler --gpu 0 --all
```
