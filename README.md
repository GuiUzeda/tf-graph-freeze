# Tensorflow Graph Freezer

### Tensorflow Graph Freezer for serving it on GCP

This is a simple helper for:
    
* Visualizing a saved checkpoint in Tensorboar
* Selecting the nodes that will be on the freezed model
* Freezing a Tensorflow Graph to serve on GCP

## Module 1: "model_to_tensorboard.py":

This module exports the graph to be visualized on Tensorboard

    optional arguments:
    -h, --help            Show this help message and exit
    -d, --dir             Tensorflow models's directory
    -s, --savedir         Logdir of tensorboard

## Module 2: "nodes_list.py":

Generate a pickled nodes list from the provided node

    optional arguments:
    -h, --help            show this help message and exit
    -l [LIST [LIST ...]], --list [LIST [LIST ...]]
                            List of words in nodes names to be excluded. If
                            omited, all nodes will be kept
    -d DIR, --dir DIR       Directory to save the nodes list
    -s SAVE_DIR, --save_dir SAVE_DIR
                            Save dir for the generated list. If 'None' is
                            provided, no list will be saved

## Module 3: "nodes_extract.py":

A simple Graph freezer for Tensorflow on GCP

    optional arguments:
    -h, --help              show this help message and exit
    -d DIR, --dir DIR       Tensorflow domel's directory
    -s SAVE_DIR, --save_dir SAVE_DIR
                            Directory where the frozen graph will be saved
    -i [INPUTS [INPUTS ...]], --inputs [INPUTS [INPUTS ...]]
                            Input tensor list
    -o [OUTPUTS [OUTPUTS ...]], --outputs [OUTPUTS [OUTPUTS ...]]
                            Output tensor list
    -n NODES, --nodes NODES
                            Pickled file wich contains the nodes to be kept

# Usage

### Step 1: Visualize the tensorboard to see wich nodes will be kept and wich are the input and output nodes

Extract the model to tensorboard:

    $ python model_to_tensorboard.py -d {model_dir} -s {save_dir}

Run Tensorboard

    $ tensorboard --logdir {save_dir}

### Step 2: Create the list containing the nodes to be kept in the final graph  

Create the list passing the node names or node names keywords to remove them from the graph

    $ python nodes_extract.py -l {list} -d {model_dir} -s {save_dir}

### Step 3: Freeze the graph to be served on GCP

    $ python freeze_graph.py -d {model_dir} -i  {input_node} -o {outout_node} -n  {node_list} -s {save_dir}



