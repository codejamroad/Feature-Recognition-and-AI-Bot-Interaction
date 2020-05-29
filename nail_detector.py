# -*- coding: utf-8 -*-

# import the necessary packages
# from object_detection.utils import label_map_util
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import cv2
from imutils.video import WebcamVideoStream
import glob

def palm_dorsal_identifier(input_path):
    args = {
        "model": "./model/export_model_008/frozen_inference_graph.pb",
        # "model":"/media/todd/38714CA0C89E958E/147/yl_tmp/readingbook/model/export_model_015/frozen_inference_graph.pb",
        "labels": "./record/classes.pbtxt",
        # "labels":"record/classes.pbtxt" ,
        "num_classes": 1,
        "min_confidence": 0.6,
        "class_model": "../model/class_model/p_class_model_1552620432_.h5"}

    COLORS = np.random.uniform(0, 255, size=(args["num_classes"], 3))

    if __name__ == '__main__':
        model = tf.Graph()

        with model.as_default():
            print("> ====== loading NAIL frozen graph into memory")
            graphDef = tf.GraphDef()

            with tf.gfile.GFile(args["model"], "rb") as f:
                serializedGraph = f.read()
                graphDef.ParseFromString(serializedGraph)
                tf.import_graph_def(graphDef, name="")
            # sess = tf.Session(graph=graphDef)
            print(">  ====== NAIL Inference graph loaded.")
            # return graphDef, sess


        with model.as_default():
            with tf.Session(graph=model) as sess:
                imageTensor = model.get_tensor_by_name("image_tensor:0")
                boxesTensor = model.get_tensor_by_name("detection_boxes:0")
                # for each bounding box we would like to know the score
                # (i.e., probability) and class label
                scoresTensor = model.get_tensor_by_name("detection_scores:0")
                classesTensor = model.get_tensor_by_name("detection_classes:0")
                numDetections = model.get_tensor_by_name("num_detections:0")
                files =glob.glob(input_path)
                print(files)
                for file in files:
                    print("Working on file {}".format(file))
                    values = []
                    cap = cv2.VideoCapture(file)
                    ret, frame = cap.read()
                    while ret:
                        ret, frame = cap.read()
                        if frame is None:
                            continue
                        frame = cv2.flip(frame, 1)
                        image = frame
                        output = image.copy()
                        img_ff, bin_mask, res = ff.find_hand_old(image.copy())
                        image = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
                        image = np.expand_dims(image, axis=0)
                        (boxes, scores, labels, N) = sess.run(
                            [boxesTensor, scoresTensor, classesTensor, numDetections],
                            feed_dict={imageTensor: image})
                        scores = np.squeeze(scores)
                        prob = scores.max()
                        thresholding = lambda x: x > 0.5
                        side =thresholding(prob)
                        values.append(side)
                        dorsal_count = np.count_nonzero(values)
                        palm_count = (np.size(values) - np.count_nonzero(values))
                    print("Dorsal values {}".format(dorsal_count))
                    print("Palm values {}".format(palm_count))
                    if dorsal_count > palm_count:
                        return 1 
                    else:
                         return 0