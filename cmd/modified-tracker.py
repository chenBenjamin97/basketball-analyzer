import os
import torch #! MUST BE IMPORTED BEFORE TENSORFLOW
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags
from absl.flags import FLAGS
import core.utils as utils
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
#from tensorflow.compat.v1 import InteractiveSession
#deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

flags.DEFINE_string('weights', './checkpoints/yolov4-416', 'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('info', True, 'show detailed info of tracked objects')
flags.DEFINE_string('custom_yolo_weights', 'yolov5_custom_weights.pt', 'weights file path')

def main(_argv):
    #Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    
    # load yolov5 custom model that finds hoop, ball and referees
    yolov5_model = torch.hub.load('ultralytics/yolov5', 'custom', path=FLAGS.custom_yolo_weights, verbose=False)  # custom model
    
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    #session = InteractiveSession(config=config)
    #STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    #using regular tf, not tflite
    saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    frame_num = 0
    # while video is running
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            print('EOF')
            break
        frame_num +=1
        print('Frame #:', frame_num)
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        #find ball, referees and hoops
        custom_model_result = yolov5_model(frame)
        result_df = custom_model_result.pandas().xyxy[0]
        
        #keep only one ball detection
        all_ball_predictions_indices = result_df.index[result_df['class'] == 0].tolist()
        if len(all_ball_predictions_indices) > 1:
            confidence_dict = {}
            for i in all_ball_predictions_indices:
                 confidence_dict[i] = result_df.at[i, 'confidence']
            
            #sort dict by it's value in decreasing order
            #x[0] = key, x[1] = value
            sorted_dict = sorted(confidence_dict.items(), key=lambda x: x[1], reverse=True)
            
            #drop from dataframe lower confidence predictions
            #i[0] = key, i[1] = value
            for i in sorted_dict[1:]:
                result_df = result_df.drop(index = i[0])

        #keep only three (maximum) referee detection
        all_referee_predictions_indices = result_df.index[result_df['class'] == 2].tolist()
        if len(all_referee_predictions_indices) > 3:
            confidence_dict = {}
            for i in all_referee_predictions_indices:
                 confidence_dict[i] = result_df.at[i, 'confidence']
            
            #sort dict by it's value in decreasing order
            #x[0] = key, x[1] = value
            sorted_dict = sorted(confidence_dict.items(), key=lambda x: x[1], reverse=True)
            
            #drop from dataframe lower confidence predictions
            #i[0] = key, i[1] = value
            for i in sorted_dict[3:]:
                result_df = result_df.drop(index = i[0])

        #keep only two (maximum) referee detection
        all_hoop_predictions_indices = result_df.index[result_df['class'] == 1].tolist()
        if len(all_hoop_predictions_indices) > 2:
            confidence_dict = {}
            for i in all_hoop_predictions_indices:
                 confidence_dict[i] = result_df.at[i, 'confidence']
            
            #sort dict by it's value in decreasing order
            #x[0] = key, x[1] = value
            sorted_dict = sorted(confidence_dict.items(), key=lambda x: x[1], reverse=True)
            
            #drop from dataframe lower confidence predictions
            #i[0] = key, i[1] = value
            for i in sorted_dict[2:]:
                result_df = result_df.drop(index = i[0])
            
        for _, row in result_df.iterrows():
            #handle situation when part of object is out of screen so tracker calculates it's place on the axis with negative value
            if int(row['xmin'])<0: row['xmin'] = '0'
            if int(row['ymin'])<0: row['ymin'] = '0'
            if int(row['xmax'])<0: row['xmax'] = '0'
            if int(row['ymax'])<0: row['ymax'] = '0'

            print('{{"Class":{0}, "Confidence":{1:.2f}, "Xmin":{2}, "Ymin":{3}, "Xmax":{4}, "Ymax":{5}}}'.format(row['class'], row['confidence']*100, int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])))

        #using regular tf, not tflite
        batch_data = tf.constant(image_data)
        pred_bbox = infer(batch_data)
        for _, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to customize tracker for only people)
        allowed_classes = ['person']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()
            
            #handle situation when part of object is out of screen so tracker calculates it's place on the axis with negative value
            if int(bbox[0])<0: bbox[0] = '0'
            if int(bbox[1])<0: bbox[0] = '0'
            if int(bbox[2])<0: bbox[0] = '0'
            if int(bbox[3])<0: bbox[0] = '0'

            #get biggest contour from frame (supposed to be the basketball court)
            biggest_contour = get_biggest_contour(frame)

            if (cv2.pointPolygonTest(biggest_contour, (int(bbox[0]), int(bbox[1])), False) == 1) or (cv2.pointPolygonTest(biggest_contour, (int(bbox[2]), int(bbox[3])), False) == 1):
                bbox_in_court = 'true'
            else:
                bbox_in_court = 'false'
                
            # if enable info flag then print details about each track
            if FLAGS.info:
                print('{{"ID":{0}, "Xmin":{1}, "Ymin":{2}, "Xmax":{3}, "Ymax":{4}, "InCourt":{5}}}'.format(str(track.track_id), int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), bbox_in_court))

            # calculate frames per second of running detections
            fps = 1.0 / (time.time() - start_time)
            print("FPS: %.2f" % fps)

def get_biggest_contour(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary_frame = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY)
    contours,_ = cv2.findContours(binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours_by_area = sorted(contours, key=cv2.contourArea, reverse=True)
    return sorted_contours_by_area[0]

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass