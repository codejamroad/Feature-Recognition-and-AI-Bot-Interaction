import subsystem_2_cnn_predict

def start(landmarks_data, output_file_name):
    label = subsystem_2_cnn_predict.predict(landmarks_data)
    file = open(output_file_name,"w")
    file.write(label)
    file.close() 