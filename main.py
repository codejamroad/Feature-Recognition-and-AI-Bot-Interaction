import subsystem_1
import subsystem_2

if __name__ == '__main__':
    testdata_path = "three_fingers_palm.webm"
    #landmarks_file = "three_fingers_dorsal.csv"
    landmarks_file= subsystem_1.start(testdata_path, enable_object_detection=True, framesToProcess=50)
    subsystem_2.start(landmarks_file)
