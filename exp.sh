PROPORTION=.2
METHOD="len"
NAME="${METHOD}_${PROPORTION}"

rm -rf models/"$NAME"
accelerate launch \
  --multi_gpu \
  --num_processes 8 \
  train.py \
    --proportion $PROPORTION \
    --method $METHOD \
    --name $NAME

python inference.py --name $NAME
python eval.py --name $NAME

# python eval.py --name rand_.05
# python eval.py --name rand_.2
# python eval.py --name rand_.4
# python eval.py --name rand_.6
# python eval.py --name rand_.8
# python eval.py --name rand_1.0
# python eval.py --name len_.05
# python eval.py --name len_.2
# python eval.py --name len_.4
# python eval.py --name len_.6
# python eval.py --name len_.8