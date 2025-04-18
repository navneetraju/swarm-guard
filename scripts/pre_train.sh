if [ ! -d ~/models/graph_encoder ]; then
  mkdir ~/models/graph_encoder
fi
python3 -m src.pre_train_graph_encoder --tuning --dataset-root=~/data --model-output-path ~/models/graph_encoder/ --epochs 200 --tune-max-epochs 15 --patience 15