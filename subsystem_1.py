import csv
import cv2
import landmark_extraction as pointsCalculator
import nail_detector as nd
import hand_detector

def zerolistmaker(n):
    listofzeros = [0] * n
    return listofzeros

def start(testdata_path, enable_object_detection=False):
    frames = []

    if enable_object_detection:
        frames = hand_detector.detect_hands_create_boundingbox(testdata_path)
    else:
        cap = cv2.VideoCapture(testdata_path)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        ret, frame = cap.read()
        while ret:
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            frames.append(frame)
            ret, frame = cap.read()
   
        
    keypoints = pointsCalculator.calculate_points(frames)
    
    #return hand_side 
    # if 1 - Dorsal
    # if 0 - Palm
    hand_side = nd.palm_dorsal_identifier(testdata_path)


    if(hand_side):
        points_to_csv =  zerolistmaker(40) + keypoints
    else:
        points_to_csv =  keypoints + zerolistmaker(40) 

    with open('landmarks.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(points_to_csv)

