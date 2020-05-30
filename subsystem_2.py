import subsystem_2_cnn_predict as cnn

def start(landmarks_data, output_file_name):
    label = cnn.predict(landmarks_data)
    with open('dog_breeds_reversed.txt', 'w') as writer:
        writer.write(label)