import subsystem_2_cnn_predict as cnn

def start(landmarks_data, output_file_name):
    label = cnn.predict(landmarks_data)
    file = open(output_file_name,"w")
    file.write(label)
    file.close()