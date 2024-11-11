# nohup python -u main.py -d a11y  --split 1 -M train -m gnn -s ignn -g 2 > logs/ignn/train/a11y_random_1.log  2>&1 &
# nohup python -u main.py -d a11y  --split 2 -M train -m gnn -s ignn -g 2 > logs/ignn/train/a11y_random_2.log  2>&1 &
# nohup python -u main.py -d a11y  --split 3 -M train -m gnn -s ignn -g 2 > logs/ignn/train/a11y_random_3.log  2>&1 &
# nohup python -u main.py -d rico  --split 1 -M train -m gnn -s ignn -g 0 > logs/ignn/train/rico_random_1.log  2>&1 &
# nohup python -u main.py -d rico  --split 2 -M train -m gnn -s ignn -g 0 > logs/ignn/train/rico_random_2.log  2>&1 &
# nohup python -u main.py -d rico  --split 3 -M train -m gnn -s ignn -g 0 > logs/ignn/train/rico_random_3.log  2>&1 &
# nohup python -u main.py -d mixed --split 1 -M train -m gnn -s ignn -g 2 > logs/ignn/train/mixed_random_1.log 2>&1 &
# nohup python -u main.py -d mixed --split 2 -M train -m gnn -s ignn -g 2 > logs/ignn/train/mixed_random_2.log 2>&1 &
# nohup python -u main.py -d mixed --split 3 -M train -m gnn -s ignn -g 2 > logs/ignn/train/mixed_random_3.log 2>&1 &

# nohup python -u main.py -cp 1730981349.536691  -d a11y  --split 1 -M predict -m gnn -s ignn -g 1 > logs/ignn/predict/a11y_random_1.log  2>&1 &
# nohup python -u main.py -cp 1731055055.9964457 -d a11y  --split 2 -M predict -m gnn -s ignn -g 1 > logs/ignn/predict/a11y_random_2.log  2>&1 &
# nohup python -u main.py -cp 1731055055.2954    -d a11y  --split 3 -M predict -m gnn -s ignn -g 1 > logs/ignn/predict/a11y_random_3.log  2>&1 &
# nohup python -u main.py -cp 1731203451.9103584 -d rico  --split 1 -M predict -m gnn -s ignn -g 0 > logs/ignn/predict/rico_random_1.log  2>&1 &
# nohup python -u main.py -cp 1731203451.6861553 -d rico  --split 2 -M predict -m gnn -s ignn -g 0 > logs/ignn/predict/rico_random_2.log  2>&1 &
# nohup python -u main.py -cp 1731203450.9416814 -d rico  --split 3 -M predict -m gnn -s ignn -g 0 > logs/ignn/predict/rico_random_3.log  2>&1 &
# nohup python -u main.py -cp 1731055055.2542074 -d mixed --split 1 -M predict -m gnn -s ignn -g 1 > logs/ignn/predict/mixed_random_1.log 2>&1 &
# nohup python -u main.py -cp 1731055055.2542017 -d mixed --split 2 -M predict -m gnn -s ignn -g 1 > logs/ignn/predict/mixed_random_2.log 2>&1 &
# nohup python -u main.py -cp 1731055055.8035808 -d mixed --split 3 -M predict -m gnn -s ignn -g 1 > logs/ignn/predict/mixed_random_3.log 2>&1 &


#### 消融 train
# 去除 image
# nohup python -u main.py -d a11y   --split 1 -M train -m gnn -s ignn -g 2 > logs/ignn/blation/train/a11y_no_image.log  2>&1 &
# nohup python -u main.py -d rico   --split 1 -M train -m gnn -s ignn -g 2 > logs/ignn/blation/train/rico_no_image.log  2>&1 &
# nohup python -u main.py -d mixed  --split 1 -M train -m gnn -s ignn -g 2 > logs/ignn/blation/train/mixed_no_image.log 2>&1 &

# 去除 text
# nohup python -u main.py -d a11y   --split 1 -M train -m gnn -s ignn -g 2 > logs/ignn/blation/train/a11y_no_text.log  2>&1 &
# nohup python -u main.py -d rico   --split 1 -M train -m gnn -s ignn -g 0 > logs/ignn/blation/train/rico_no_text.log  2>&1 &
# nohup python -u main.py -d mixed  --split 1 -M train -m gnn -s ignn -g 0 > logs/ignn/blation/train/mixed_no_text.log 2>&1 &

# 去除 attributes
# nohup python -u main.py -d a11y   --split 1 -M train -m gnn -s ignn -g 0 > logs/ignn/blation/train/a11y_no_attributes.log  2>&1 &
# nohup python -u main.py -d rico   --split 1 -M train -m gnn -s ignn -g 0 > logs/ignn/blation/train/rico_no_attributes.log  2>&1 &
# nohup python -u main.py -d mixed  --split 1 -M train -m gnn -s ignn -g 0 > logs/ignn/blation/train/mixed_no_attributes.log 2>&1 &

# 去除 coordinate
# nohup python -u main.py -d a11y   --split 1 -M train -m gnn -s ignn -g 0 > logs/ignn/blation/train/a11y_no_coordinate.log  2>&1 &
# nohup python -u main.py -d rico   --split 1 -M train -m gnn -s ignn -g 0 > logs/ignn/blation/train/rico_no_coordinate.log  2>&1 &
# nohup python -u main.py -d mixed  --split 1 -M train -m gnn -s ignn -g 0 > logs/ignn/blation/train/mixed_no_coordinate.log 2>&1 &
#### 消融 train


#### 消融 predict
# 去除 image
# nohup python -u main.py -cp 1730983582.5033724 -d a11y  --split 1 -M predict -m gnn -s ignn -g 0 > logs/ignn/blation/predict/a11y_no_image.log  2>&1 &
# nohup python -u main.py -cp 1730983582.2062366 -d rico  --split 1 -M predict -m gnn -s ignn -g 0 > logs/ignn/blation/predict/rico_no_image.log  2>&1 &
# nohup python -u main.py -cp 1731055727.5324903 -d mixed --split 1 -M predict -m gnn -s ignn -g 0 > logs/ignn/blation/predict/mixed_no_image.log 2>&1 &

# 去除 text
# nohup python -u main.py -cp 1731128942.8799288 -d a11y  --split 1 -M predict -m gnn -s ignn -g 0 > logs/ignn/blation/predict/a11y_no_text.log  2>&1 &
# nohup python -u main.py -cp 1731128942.724855  -d rico  --split 1 -M predict -m gnn -s ignn -g 0 > logs/ignn/blation/predict/rico_no_text.log  2>&1 &
# nohup python -u main.py -cp 1731128943.013998  -d mixed --split 1 -M predict -m gnn -s ignn -g 0 > logs/ignn/blation/predict/mixed_no_text.log 2>&1 &

# 去除 attributes
# nohup python -u main.py -cp 1731163051.1285827 -d a11y  --split 1 -M predict -m gnn -s ignn -g 0 > logs/ignn/blation/predict/a11y_no_attributes.log  2>&1 &
# nohup python -u main.py -cp 1731163051.1664743 -d rico  --split 1 -M predict -m gnn -s ignn -g 0 > logs/ignn/blation/predict/rico_no_attributes.log  2>&1 &
# nohup python -u main.py -cp 1731163051.2846143 -d mixed --split 1 -M predict -m gnn -s ignn -g 0 > logs/ignn/blation/predict/mixed_no_attributes.log 2>&1 &

# 去除 coordinate
# nohup python -u main.py -cp 1731163233.4876735 -d a11y  --split 1 -M predict -m gnn -s ignn -g 0 > logs/ignn/blation/predict/a11y_no_coordinate.log  2>&1 &
# nohup python -u main.py -cp 1731163233.3670316 -d rico  --split 1 -M predict -m gnn -s ignn -g 0 > logs/ignn/blation/predict/rico_no_coordinate.log  2>&1 &
# nohup python -u main.py -cp 1731163233.5736125 -d mixed --split 1 -M predict -m gnn -s ignn -g 0 > logs/ignn/blation/predict/mixed_no_coordinate.log 2>&1 &
#### 消融 predict

