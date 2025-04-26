PROPORTION=.05
METHOD="rand"
NAME="${METHOD}_${PROPORTION}"

rm -rf models/"$NAME"
accelerate --multi_gpu --num_processes 6 launch train.py --proportion $PROPORTION --method $METHOD --name $NAME
python inference.py --name $NAME
python eval.py --name $NAME