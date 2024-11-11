f=512 v=99.3
nie=gcn-nie-nst

g=0
nrl=None
hops=None
d=actor s=pyg
m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -nie $nie  -nrl $nrl -hops $hops > logs/$d.log &

g=0
nrl=None
hops=None
d=chameleon s=critical
m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -nie $nie  -nrl $nrl -hops $hops > logs/$d.log &

g=1
nrl=None
hops=None
d=squirrel s=critical
m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -nie $nie  -nrl $nrl -hops $hops > logs/$d.log &

g=2
nrl=None
hops=None
d=flickr s=cola
m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -nie $nie  -nrl $nrl -hops $hops > logs/$d.log &

g=2
nrl=None
hops=None
d=blogcatalog s=cola
m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -nie $nie  -nrl $nrl -hops $hops > logs/$d.log &

g=3
nrl=None
hops=None
d=roman-empire s=critical
m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -nie $nie  -nrl $nrl -hops $hops > logs/$d.log &

g=4
nrl=None
hops=None
d=amazon-ratings s=critical
m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -nie $nie  -nrl $nrl -hops $hops > logs/$d.log &

g=4
nrl=None
hops=None
d=photo s=pyg
m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -nie $nie  -nrl $nrl -hops $hops > logs/$d.log &

g=5
nrl=None
hops=None
d=pubmed s=pyg
m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -nie $nie  -nrl $nrl -hops $hops > logs/$d.log &

g=5
nrl=None
hops=None
d=wikics s=pyg
m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -nie $nie  -nrl $nrl -hops $hops > logs/$d.log &
