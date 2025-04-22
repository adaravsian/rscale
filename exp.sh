PROPORTION=1.0
METHOD="rand"
NAME="${METHOD}_${PROPORTION}"

python train.py --proportion $PROPORTION --method $METHOD --name $NAME
python inference.py --name $NAME
python eval.py --name $NAME


PROPORTION=.8
METHOD="rand"
NAME="${METHOD}_${PROPORTION}"

python train.py --proportion $PROPORTION --method $METHOD --name $NAME
python inference.py --name $NAME
python eval.py --name $NAME


PROPORTION=.6
METHOD="rand"
NAME="${METHOD}_${PROPORTION}"

python train.py --proportion $PROPORTION --method $METHOD --name $NAME
python inference.py --name $NAME
python eval.py --name $NAME


PROPORTION=.4
METHOD="rand"
NAME="${METHOD}_${PROPORTION}"

python train.py --proportion $PROPORTION --method $METHOD --name $NAME
python inference.py --name $NAME
python eval.py --name $NAME

PROPORTION=.2
METHOD="rand"
NAME="${METHOD}_${PROPORTION}"

python train.py --proportion $PROPORTION --method $METHOD --name $NAME
python inference.py --name $NAME
python eval.py --name $NAME