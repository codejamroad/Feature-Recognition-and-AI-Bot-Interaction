import cv2
import landmark_extraction as pointsCalculator
import nail_detector as nd
import hand_detector


def start(testdata_path, enable_object_detection=False, framesToProcess = 100):
    frames = []

    if enable_object_detection:
        ("Starting Hand Detector**********************")
        frames = hand_detector.detect_hands_create_boundingbox(testdata_path)
        ("************************** End ")
    else:
        cap = cv2.VideoCapture(testdata_path)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        ret, frame = cap.read()
        while ret:
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            frames.append(frame)
            ret, frame = cap.read()

        #return hand_side 
    # if 1 - Dorsal
    # if 0 - Palm
    print("Nail Detector Starting ***********************")
    hand_side = nd.palm_dorsal_identifier(testdata_path)
    print("Nail Detection End*********************")

    print("**************Extracting Landmarks*******************")
    pointsCalculator.calculate_points(frames[:framesToProcess], hand_side)
    print("***************** Process Complete***********************")