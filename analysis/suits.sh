datadir=cifar10-small
dataset=cifar10


for a in 0.2 0.4 0.6 0.8
do
    python grid_search.py --datadir ${datadir} --dataset ${dataset} --y train_loss --graph fixed_a --a $a
    python grid_search.py --datadir ${datadir} --dataset ${dataset} --y test_loss --graph fixed_a --a $a
    python grid_search.py --datadir ${datadir} --dataset ${dataset} --y test_acc --graph fixed_a --a $a
done
python grid_search.py --datadir ${datadir} --dataset ${dataset} --y train_loss --graph grid
python grid_search.py --datadir ${datadir} --dataset ${dataset} --y test_loss --graph grid
python grid_search.py --datadir ${datadir} --dataset ${dataset} --y test_acc --graph grid
