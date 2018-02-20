import os
import argparse

import tensorflow as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
import pickle


def main(args):
    model_dir = args.dir


    checkpoint = tf.train.get_checkpoint_state(model_dir)

    input_checkpoint = checkpoint.model_checkpoint_path

    clear_devices = True
    with tf.Session(graph=tf.Graph()) as sess:

        saver = tf.train.import_meta_graph(
            input_checkpoint + '.meta', clear_devices=clear_devices)

        saver.restore(sess, input_checkpoint)

        writer = tf.summary.FileWriter(args.savedir, sess.graph)
        sess.run(tf.local_variables_initializer())
        writer.close()

if __name__ == "__main__":
    cwd = os.getcwd()
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-d", "--dir", help="Tensorflow models's directory",
                        default=os.path.join(cwd, "model_example"))

    parser.add_argument("-s", "--savedir", help="Logdir tensorboard",
                    default=os.path.join(cwd, "model_example"))
    args = parser.parse_args()

    main(args)
