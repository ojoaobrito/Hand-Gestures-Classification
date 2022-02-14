import os, sys
import numpy as np
import tensorflow as tf
import argparse
from PIL import Image

args_parser = argparse.ArgumentParser()

args_parser.add_argument("--image", required=False, help="Path to the image", default="test/0_0_4_1_0_0f52.jpg")

args = vars(args_parser.parse_args())

IMAGE_PATH = args["image"]

with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph("inception_model/inceptionv3_1.meta")
    new_saver.restore(sess, "inception_model/inceptionv3_1")

    x = tf.get_default_graph().get_tensor_by_name("x:0")
    is_training = tf.get_default_graph().get_tensor_by_name("is_training:0")

    prediction_op = tf.get_collection("prediction_op")[0]
    pred = sess.run(prediction_op, feed_dict={x: [np.asarray(Image.open(IMAGE_PATH).convert("L").resize((299,299),Image.LANCZOS))], is_training: False})
    print(pred)