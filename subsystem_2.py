import subsystem_2_cnn_predict as cnn

def start(landmarks_data, output_file_name):
    label = cnn.predict(landmarks_data)
    file_writer = open(output_file_name,"w")
    file_writer(label)
    file_writer.close()