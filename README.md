## Installation

- python>=3.8
- for installation scripts see .ci/install-dev.sh, .ci/install.sh

```sh
bash .ci/install-dev.sh
bash .ci/install.sh
```

## Dataset

### Download

- extra-nos-raw-labeled
    - [download-url](https://drive.google.com/file/d/1PGznhGlPKjEXatq0Sh3cVUuqx5beOfHF/view?usp=sharing)
    - train/predict now-raw-labeled needed, you can download or generate by script
- extra-rico-labeled
    - [download-url](https://drive.google.com/file/d/1MwuhkmBbyLtWzBKyRWf8oaVofwKu1rp5/view?usp=sharing)
    - train/predict rico-labeled needed, you can download or generate by script
- nos-raw-labeled
    - [download-url](https://drive.google.com/file/d/1L0TofPa66H98vM86beLMjswpveLjoHTT/view?usp=sharing)
    - train/predict now-raw-labeled needed
- rico-labeled
    - [download-url](https://drive.google.com/file/d/11MLqeLgibZo2jPf9UuQtJPe08aPqxNJS/view?usp=sharing)
    - train/predict rico-labeled needed
- mixed-split 
    - [download-url](https://drive.google.com/file/d/19CQATtbjdSmuMV5CF6m4rRe7mqk_uL3T/view?usp=sharing)
    - train/predict mixed needed, dataset split for mixed experiments
- weights
    - [download-url](https://drive.google.com/file/d/1zV2s5r74GLxNlwJJsUKS8o4TrAR85R0X/view?usp=sharing)
    - predict needed, trained model weights

### Organize the directory structure

- downloaded directory structure

```sh
.
├── extra-nos-raw-labeled
│   ├── box
│   ├── box_feat
│   ├── graph
│   └── text_feat
├── extra-rico-labeled
│   ├── box
│   ├── box_feat
│   ├── graph
│   └── text_feat
├── nos-raw-labeled
│   ├── hierarchy
│   └── screenshot
├── rico-labeled
│   ├── hierarchy
│   └── screenshot
├── mixed-split
└── weights
    ├── mixed
    ├── nos-raw-labeled
    └── rico-labeled
```

- you should put the dataset into `NOS/dataset`

```sh
cd NOS
mkdir dataset
mv ${nos-raw-labeled-path} ./dataset/nos-raw-labeled
mv ${rico-labeled-path}    ./dataset/rico-labeled
```

- if you has downloaded the `extra` part, you need to run:

```sh
mv ${extra-nos-raw-labeled-path}/* ./dataset/nos-raw-labeled
mv ${extra-rico-labeled-path}/*    ./dataset/rico-labeled
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

- finally, the `NOS` workspace will be:

```sh
.
├── dataset
│   ├── nos-raw-labeled
│   │   ├── box
│   │   ├── box_feat
│   │   ├── graph
│   │   ├── hierarchy
│   │   ├── screenshot
│   │   ├── text_feat
│   │   └── dataset_split_{1,2,3}.json
│   ├── rico-labeled
│   │   ├── box
│   │   ├── box_feat
│   │   ├── graph
│   │   ├── hierarchy
│   │   ├── screenshot
│   │   ├── text_feat
│   │   └── dataset_split_{1,2,3}.json
│   └── mixed # combine the contents of the two folders `nos-raw-labeled` and `rico-labeled`
│       ├── box
│       ├── box_feat
│       ├── graph
│       ├── hierarchy
│       ├── screenshot
│       ├── text_feat
│       └── dataset_split_{1,2,3}.json
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

mv ${weights-path}/nos-raw-labeled/split_1.pt ./model_checkpoint/ignn/nos-raw-labeled_split_1
mv ${weights-path}/rico-labeled/split_1.pt    ./model_checkpoint/ignn/rico-labeled_split_1
```

- run command

```sh
source .env/bin/activate

python -u main.py --mode predict --checkpoint split_1 --dataset nos-raw --split 1 --gpu 0
python -u main.py --mode predict --checkpoint split_1 --dataset rico    --split 1 --gpu 0
```

- then you can see the result in `./predict_result/ignn`

## Train

- run command

```sh
source .env/bin/activate

python -u main.py --dataset nos-raw  --split 1 --mode train --gpu 0
python -u main.py --dataset rico     --split 1 --mode train --gpu 0
```

- the checkpoint will be saved in `./model_checkpoint/ignn`