import subsystem_2_cnn_predict

def start(landmarks_data):
    label = subsystem_2_cnn_predict.predict(landmarks_data)
    file = open("myfile.txt","w")
    file.write(label)
    file.close() 