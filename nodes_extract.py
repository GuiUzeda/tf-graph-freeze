import os
import argparse
import ast
import tensorflow as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
import pickle


def main(args):
    model_dir = args.model
    nodes_list = args.list
    save_dir = args.save_dir

    checkpoint = tf.train.get_checkpoint_state(model_dir)

    input_checkpoint = checkpoint.model_checkpoint_path

    clear_devices = True

    with tf.Session(graph=tf.Graph()) as sess:

        saver = tf.train.import_meta_graph(
            input_checkpoint + '.meta', clear_devices=clear_devices)

        saver.restore(sess, input_checkpoint)

        ops_list = []
        for op in tf.get_default_graph().get_operations():
            name = op.name
            fileter_result = filter(lambda x: x.lower() in name.lower(), nodes_list)
            if len(fileter_result) == 0:
                ops_list.append(name)
                print name


    if save_dir.lower() != "none":
        pickle.dump(ops_list, open(save_dir  + "/nodes_list.p", "wb"))
    print ""  
    print "-->A total of {} noder were kept<--".format(len(ops_list))

if __name__ == "__main__":
    cwd = os.getcwd()
    parser = argparse.ArgumentParser(
        description='Generate a pickled nodes list from the provided node')
    parser.add_argument("-l", "--list", help="List of words in nodes names to be excluded. If omited, all nodes will be kept", nargs="*")
    parser.add_argument("-m", "--model", help="Print and save a list of all nodes in the provided graph directory",
                        default=os.path.join(cwd, "model_example"))
    parser.add_argument("-s", "--save_dir", help="Save dir for the generated list. If 'None' is provided, no list will be saved",
                        default=os.path.join(cwd))
    args = parser.parse_args()
    main(args)
