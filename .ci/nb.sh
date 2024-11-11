IP=$1
PORT=$2
nohup jupyter notebook --ip=${IP:=10.214.199.218} --port=${PORT:=8888} --no-browser --config=./configs/nb.py > logs/nb.log &
