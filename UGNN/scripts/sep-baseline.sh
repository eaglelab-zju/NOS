
model=OrderedGNN

gpu=0
d=actor s=pyg
nohup python -u run_baselines.py --model $model --gpu $gpu --dataset $d --source $s >logs/$model-$d.log &
d=squirrel s=critical
nohup python -u run_baselines.py --model $model --gpu $gpu --dataset $d --source $s >logs/$model-$d.log &
d=chameleon s=critical
nohup python -u run_baselines.py --model $model --gpu $gpu --dataset $d --source $s >logs/$model-$d.log &

gpu=1
d=photo s=pyg
nohup python -u run_baselines.py --model $model --gpu $gpu --dataset $d --source $s >logs/$model-$d.log &

gpu=2
d=pubmed s=pyg
nohup python -u run_baselines.py --model $model --gpu $gpu --dataset $d --source $s >logs/$model-$d.log &

gpu=3
d=wikics s=pyg
nohup python -u run_baselines.py --model $model --gpu $gpu --dataset $d --source $s >logs/$model-$d.log &

gpu=4
d=blogcatalog s=cola
nohup python -u run_baselines.py --model $model --gpu $gpu --dataset $d --source $s >logs/$model-$d.log &

gpu=5
d=flickr s=cola
nohup python -u run_baselines.py --model $model --gpu $gpu --dataset $d --source $s >logs/$model-$d.log &

gpu=6
d=roman-empire s=critical
nohup python -u run_baselines.py --model $model --gpu $gpu --dataset $d --source $s >logs/$model-$d.log &

gpu=7
d=amazon-ratings s=critical
nohup python -u run_baselines.py --model $model --gpu $gpu --dataset $d --source $s >logs/$model-$d.log &
