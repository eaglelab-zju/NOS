## Installation

- python>=3.8
- for installation scripts see .ci/install-dev.sh, .ci/install.sh

```sh
bash .ci/install-dev.sh
bash .ci/install.sh
```

## Dataset

```sh
.
├── extra # download or generate
│   ├── nos-raw-labeled
│   │   ├── box
│   │   ├── box_feat
│   │   ├── graph
│   │   └── text_feat
│   └── rico-labeled
│       ├── box
│       ├── box_feat
│       ├── graph
│       └── text_feat
├── nos-raw
│   └── $app_name
│       ├── $page_id.png
│       └── $page_id.json
├── nos-raw-labeled
│   ├── hierarchy
│   └── screenshot
├── rico-labeled
│   ├── hierarchy
│   └── screenshot
└── weights # trained model weights
    ├── mixed
    ├── nos-raw-labeled
    └── rico-labeled
```

- you should put the dataset into `./dataset`

```sh
mkdir dataset
cp -r $download_dir/nos-raw-labeled ./dataset/
cp -r $download_dir/rico-labeled    ./dataset/
```

- if you has downloaded the `extra` part, you need to run:

```sh
cp -r $download_dir/extra/nos-raw-labeled/* ./dataset/nos-raw-labeled
cp -r $download_dir/extra/rico-labeled/*    ./dataset/rico-labeled
```

- otherwise, you need generate the `extra` part by:

```sh
# first edit the file `preprocess.py`
line 14 dataset_dir = "./dataset/nos-raw-labeled" <-- choose one dataset
line 15 # dataset_dir = "./dataset/rico-labeled" <-- and comment the another line

# then run the preprocess script to generate `box` `box_feat` `graph` `text_feat` (about 30min per dataset)
python preprocess.py

# for dataset `nos-raw` and `rico`, you should repeat the above steps twice
```

- finally, the workspace will be:

```sh
.
├── dataset
│   ├── nos-raw-labeled
│   │   ├── box
│   │   ├── box_feat
│   │   ├── graph
│   │   ├── hierarchy
│   │   ├── screenshot
│   │   └── text_feat
│   └── rico-labeled
│       ├── box
│       ├── box_feat
│       ├── graph
│       ├── hierarchy
│       ├── screenshot
│       └── text_feat
├── main.py
├── config.py
├── data_loader.py
├── model.py
├── predict.py
├── utils.py
└── IGNN
```

## Predict

- copy the checkpoint file

```sh
mkdir -p ./model_checkpoint/ignn/nos-raw-labeled_split_1
mkdir -p ./model_checkpoint/ignn/rico-labeled_split_1

cp $download_dir/weights/nos-raw-labeled/split_1.pt ./model_checkpoint/ignn/nos-raw-labeled_split_1
cp $download_dir/weights/rico-labeled/split_1.pt    ./model_checkpoint/ignn/rico-labeled_split_1
```

- run command

```sh
source .env/bin/activate

python -u main.py --mode predict --checkpoint split_1 --dataset nos-raw --split 1 --gpu 0
python -u main.py --mode predict --checkpoint split_1 --dataset nos-raw --split 1 --gpu 0
```

- then you can see the result in `./predict_result/ignn`

## Train

- run command

```sh
source .env/bin/activate

python -u main.py --dataset nos-raw  --split 1 --mode train --gpu 0
python -u main.py --dataset rico     --split 1 --mode train --gpu 0
```

- the checkpoint will be saved in `./model_checkpoint/ignn/`