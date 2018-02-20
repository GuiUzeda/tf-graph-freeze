# tf-graph-freeze
Tensorflow Graph Freezer for serving it on GCP


python freeze_graph.py -i image_tensor -o concat -n nodes_list.p

python model_to_tensorboard.py

python nodes_extract.py  -l save/ global_step Assign train