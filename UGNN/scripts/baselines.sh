model=OrderedGNN
gpu=0
nohup python -u run_baselines.py --model $model --gpu $gpu --dataset all >logs/$model.log &
model=GraphSAGE
gpu=0
nohup python -u run_baselines.py --model $model --gpu $gpu --dataset all >logs/$model.log &
model=APPNP
gpu=0
nohup python -u run_baselines.py --model $model --gpu $gpu --dataset all >logs/$model.log &

model=ACMGCN
gpu=1
nohup python -u run_baselines.py --model $model --gpu $gpu --dataset all >logs/$model.log &
model=GBKGNN
gpu=1
nohup python -u run_baselines.py --model $model --gpu $gpu --dataset all >logs/$model.log &

model=HOGGCN
gpu=2
nohup python -u run_baselines.py --model $model --gpu $gpu --dataset all >logs/$model.log &

model=GCNII
gpu=3
nohup python -u run_baselines.py --model $model --gpu $gpu --dataset all >logs/$model.log &
model=GCN
gpu=3
nohup python -u run_baselines.py --model $model --gpu $gpu --dataset all >logs/$model.log &

model=GGCN
gpu=4
nohup python -u run_baselines.py --model $model --gpu $gpu --dataset all >logs/$model.log &

model=GloGNN
gpu=5
nohup python -u run_baselines.py --model $model --gpu $gpu --dataset all >logs/$model.log &
model=GAT
gpu=5
nohup python -u run_baselines.py --model $model --gpu $gpu --dataset all >logs/$model.log &

model=GPRGNN
gpu=6
nohup python -u run_baselines.py --model $model --gpu $gpu --dataset all >logs/$model.log &
model=MLP
gpu=6
nohup python -u run_baselines.py --model $model --gpu $gpu --dataset all >logs/$model.log &

model=H2GCN
gpu=7
nohup python -u run_baselines.py --model $model --gpu $gpu --dataset all >logs/$model.log &
model=MixHop
gpu=7
nohup python -u run_baselines.py --model $model --gpu $gpu --dataset all >logs/$model.log &

model=IncepGCN
gpu=0
nohup python -u run_baselines.py --model $model --gpu $gpu --dataset all >logs/$model.log &

model=DAGNN
gpu=0
nohup python -u run_baselines.py --model $model --gpu $gpu --dataset all >logs/$model.log &

model=SGFormer
gpu=0
nohup python -u run_baselines.py --model $model --gpu $gpu --dataset all >logs/$model.log &

model=DIFFormer
gpu=0
nohup python -u run_baselines.py --model $model --gpu $gpu --dataset all >logs/$model.log &

model=NodeFormer
gpu=0
nohup python -u run_baselines.py --model $model --gpu $gpu --dataset all >logs/$model.log &
