PROPORTION=1.0
METHOD="rand"
NAME="${METHOD}_${PROPORTION}"

# rm -rf models/"$NAME"
accelerate launch \
  --multi_gpu \
  --num_processes 8 \
  train.py \
    --proportion $PROPORTION \
    --method $RAND \
    --name $NAME

python inference.py --name $NAME
python eval.py --name $NAME