NAME="full"

python train.py --name $NAME
python inference.py --name $NAME
python eval.py --name $NAME