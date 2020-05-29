import subsystem_1
import subsystem_2

if __name__ == '__main__':
    testdata_path = "three_fingers_dorsal.webm"
    subsystem_1.start(testdata_path, enable_object_detection=True, framesToProcess=50)
    #subsystem_2.start()