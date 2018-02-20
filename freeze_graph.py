import os
import argparse

import tensorflow as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
import pickle


def main(args):
   
    if not os.path.exists(args.dir):
        raise Exception("The path provided to save is not valid")

    model_dir = args.dir

    export_dir = args.freeze
    print(args.nodes)
    if not args.nodes:
        raise Exception("A Pickled node list must be provided")

    nodes = pickle.load(open(args.nodes, "rb"))

    declared_inputs = args.inputs
    declared_outputs = args.outputs
    inputs = {}
    outputs = {}

    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

    checkpoint = tf.train.get_checkpoint_state(model_dir)

    input_checkpoint = checkpoint.model_checkpoint_path

    sigs = {}

    clear_devices = True

    with tf.Session(graph=tf.Graph()) as sess:

        saver = tf.train.import_meta_graph(
            input_checkpoint + '.meta', clear_devices=clear_devices)

        saver.restore(sess, input_checkpoint)
    
        ops_list = []
        for op in tf.get_default_graph().get_operations():
            name = op.name
            ops_list.append(name)
        for n in nodes:
            print n
            if n not in ops_list:
                raise Exception("Nodes provided are not in the graph")
        
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            tf.get_default_graph().as_graph_def(),
            nodes,
            variable_names_blacklist=["global_step"],
        )

        with tf.Graph().as_default() as new_graph:

            tf.import_graph_def(output_graph_def, name="")

            for i in range(len(declared_inputs)):
                inputs["input{}".format(i)] = new_graph.get_tensor_by_name(declared_inputs[i] +
                                                                           (":0" if not declared_inputs[i].endswith(
                                                                               ":0") else "")
                                                                           )

            for i in range(len(declared_outputs)):
                outputs["input{}".format(i)] = new_graph.get_tensor_by_name(declared_outputs[i] +
                                                                            (":0" if not declared_outputs[i].endswith(
                                                                                ":0") else "")
                                                                            )

            sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
                tf.saved_model.signature_def_utils.predict_signature_def(
                    inputs, outputs)

            builder.add_meta_graph_and_variables(sess,
                                                 [tag_constants.SERVING],
                                                 signature_def_map=sigs)

            builder.save()


if __name__ == "__main__":
    cwd = os.getcwd()
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-d", "--dir", help="Tensorflow domel's directory",
                        default=os.path.join(cwd, "model_example"))
    parser.add_argument("-f", "--freeze", help="Directory where the frozen graph will be saved",
                        default=os.path.join(cwd, "frozen_graph"))
    parser.add_argument(
        "-i", "--inputs", help="Input tensor list", required=True, nargs="*")
    parser.add_argument("-o", "--outputs",
                        help="Output tensor list", required=True, nargs="*")
    parser.add_argument(
        "-n", "--nodes", help="Pickled file wich contains the nodes to be kept", required=False)

    args = parser.parse_args()

    main(args)
