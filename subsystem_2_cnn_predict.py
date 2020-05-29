from keras.models import model_from_json
import numpy as np 
from keras.applications.inception_v3 import decode_predictions
import pandas as pd
from tqdm import tqdm 
import csv
from sklearn.preprocessing import LabelEncoder

def predict(test_data):
  json_file = open('model_cnn.json', 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  loaded_model = model_from_json(loaded_model_json)
  # load weights into new model
  loaded_model.load_weights("model_cnn.h5")
  print("Loaded model from disk")
  loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

  df_org = pd.read_csv(test_data, header=None)
  #df_org = df_org.loc[:,4:]
  #df_org = df_org.iloc[250:300]
  #sample = df_org.iloc[1:]
  #for i in range(0,100):
  #   for j in range(40,80):
    #      df_org[i,j] = 0
  #display(df_org)
  #sample = df_org.iloc[0:100,0:80]
  #display(sample)
  sample = np.array(df_org)
  #display(sample)
  print(sample.shape)
  sample = sample.reshape(-1,80,1)

  predicted_classes = loaded_model.predict(sample, verbose=2)
  predicted_classes = np.round(predicted_classes)

  encoder = LabelEncoder()
  encoded_Y = encoder.inverse_transform(predicted_classes)
  print(encoded_Y)