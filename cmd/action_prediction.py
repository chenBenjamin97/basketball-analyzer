import sys
import tensorflow as tf
import argparse
import numpy as np
from sklearn.preprocessing import RobustScaler

parser = argparse.ArgumentParser(description='Make a prediction, reading data via stdin and printing result to stdout')
parser.add_argument('--model', action='store', dest='model_path', required=True)
parser.add_argument('--timestamps', action='store', dest='amount_of_timestamps', required=True, type=int)
parser.add_argument('--persons', action='store', dest='amount_of_persons_to_predict', required=True, type=int)
parser.add_argument('--keypoints', action='store', dest='amount_of_person_keypoints', required=True, type=int)

parsed_flags = parser.parse_args()

trained_model = tf.keras.models.load_model(parsed_flags.model_path)
class_names = {0 : "Block", 1 : "Pass", 2 : "Run", 3: "Dribble",4: "Shoot",5 : "Ball In Hand", 6 : "Defense", 7: "Pick" , 8 : "No Action" , 9: "Walk"} #there are no samples of class "discard" in our dataset


for i in range(parsed_flags.amount_of_persons_to_predict):
  timestamps_counter = 0
  personID = ""
  all_timestamps_joints_stack = np.ndarray((1,parsed_flags.amount_of_person_keypoints*2), dtype='int')
  
  for line in sys.stdin:
      timestamps_counter+=1
      #manipulations in order to get the data same as our google colab notebook
      #data from stdin seems like: personID;x1,y1;x2,y2;..x13,y13
      current_timestamp_joints = line.rstrip().split(";")
      personID = current_timestamp_joints[0]
      del current_timestamp_joints[0] #first value is not a point, it's the person's ID, remove it from sequence
      current_timestamp_joints = map(lambda x: eval(x), current_timestamp_joints)
      current_timestamp_joints = np.asarray(list(current_timestamp_joints))
      current_timestamp_joints = RobustScaler().fit_transform(current_timestamp_joints)
      current_timestamp_joints = current_timestamp_joints.reshape(parsed_flags.amount_of_person_keypoints*2,) #reshape to one dimensional array
      all_timestamps_joints_stack = np.vstack((all_timestamps_joints_stack, current_timestamp_joints))
      if timestamps_counter == parsed_flags.amount_of_timestamps:
        break

  all_timestamps_joints_stack = np.delete(all_timestamps_joints_stack, 0, axis=0) #remove first element from initialization of stack
  all_timestamps_joints_stack = np.expand_dims(all_timestamps_joints_stack, axis=0) #in order to pack it stack as one sample

  all_predictions = trained_model.predict(all_timestamps_joints_stack)
  
  #find highest confidence prediction
  predicted_class_index = np.argmax(all_predictions)
  highest_conf_text = class_names[predicted_class_index]+';'+"{:.2f}%".format(all_predictions[0][predicted_class_index]*100)

  #find second highest confidence prediction
  all_predictions[0][predicted_class_index] = -1
  predicted_class_index = np.argmax(all_predictions)
  second_highest_conf_text = class_names[predicted_class_index]+';'+"{:.2f}%".format(all_predictions[0][predicted_class_index]*100)

  #find second highest confidence prediction
  all_predictions[0][predicted_class_index] = -1
  predicted_class_index = np.argmax(all_predictions)
  third_highest_conf_text = class_names[predicted_class_index]+';'+"{:.2f}%".format(all_predictions[0][predicted_class_index]*100)

  print(personID+';'+ highest_conf_text + ';' + second_highest_conf_text + ';' + third_highest_conf_text) #we multiply the softmax output by 100 in order to show it by percentage