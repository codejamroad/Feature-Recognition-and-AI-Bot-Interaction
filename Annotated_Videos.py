import os
import csv


landmarkInfo = pd.read_csv("Anil Kumar_Shruthi_8781_annotations.csv").dropna()
files = [f for f in os.listdir('.') if os.path.splitext(f)[1]==".webm"]

for f in files :
    Startfrom=0
    reader = csv.DictReader(open("Anil Kumar_Shruthi_8781_annotations.csv")) #landmark files to be read 
    positions1 = [] #holds the csv data
    for line in reader:
        if line["source"] == f:
            positions1.append(line)
    path = "annotated/"
    print(f)           # f is the original video file
    if "dorsal" in f : #dorsal values are updated from column 40 in csv while palm values from 0
        Startfrom = 40

    writer = cv2.VideoWriter(path+f+"_annotated.webm", cv2.CAP_ANY, cv2.VideoWriter_fourcc(*"VP80"), 24, (640,480))#create annotated video file
    cap = cv2.VideoCapture(f)    #convert original video to frame
    ret, frame = cap.read()
    count = 0  #variabe used to interate through frame
    while ret:
        # list of finger landmarks created which is cleared every frame
        ThumbList=[]
        IndexList = []
        MiddleList = []
        RingList = []
        PinkyList =[]

        ret, frame = cap.read()
        if not ret:
            break
         
        #below set of lines helps label the video frames with string annotated
        text_string = "Annotated" #label to write
        position = (5, 50) #top left corner of the framr
        font = cv2.FONT_HERSHEY_SIMPLEX # font of label
        font_size = 2 # size of label
        font_color = (255, 255, 255) # Remember its BGR color of label
        font_thickness = 2 #thickness of label
        cv2.putText(frame, text_string, position,
        font, font_size, font_color, font_thickness) # label the frame

       # popping unwanted columns from csv data extracted  
        CurrentList = list(positions1[count].values())
        CurrentList.pop(0)
        CurrentList.pop(0)
        CurrentList.pop(0)
        CurrentList.pop(0)

        positions = CurrentList # this list has only landmarks from the current frame
        StartingPoint = (int(float(positions[Startfrom+0])),int(float(positions[Startfrom+1]))) #Starting point to draw
        #StartingPoint is the root landmark of the palm or dorsal side comes as the first 2 columns in the set
        cv2.circle(frame, StartingPoint, 2, (255,255,0), 2)
        
        #Below is the logic to create the ThumbList 
        for i in range(Startfrom+2,Startfrom+8):
            if(int(float(positions[i]))!= 0):
                ThumbList.append(int(float(positions[i])))

        NextChange = i #here it should be 7
        # NextChange variable is used to track teh offset of columns in the csv file which is now staored in positions..
        #Below is the logic to create the rest of the finger list 
        for finger in range(2,6):
            for i in range(NextChange+1,NextChange+9):
                if int(float(positions[i]))!= 0:
                    if finger == 2 :
                        IndexList.append(int(float(positions[i])))
                    elif finger == 3:
                        MiddleList.append(int(float(positions[i])))
                    elif finger == 4:
                        
                        RingList.append(int(float(positions[i])))
                    else:
                        PinkyList.append(int(float(positions[i])))
            NextChange = i # finger 2 = 15, finger3 =23, finger4 = 32, finger5 = 40

        #Below we are creating pair of elements elements from the landmarks obtained as (x,y) 
        ThumbList =list(zip(ThumbList[0::2], ThumbList[1::2]) )
        IndexList =list(zip(IndexList[0::2],IndexList[1::2]))
        MiddleList=list(zip(MiddleList[0::2],MiddleList[1::2])) 
        RingList = list(zip(RingList[0::2],RingList[1::2]))  
        PinkyList = list(zip(PinkyList[0::2],PinkyList[1::2])) 
  
        fingerdict = {}
        fingerdict[1]=ThumbList
        fingerdict[2]= IndexList
        fingerdict[3]= MiddleList 
        fingerdict[4]= RingList 
        fingerdict[5]= PinkyList 
 
        for key in fingerdict:
            listPosition = fingerdict[key]  

        #print(listPosition)
            flag = 0
                           
            while(flag <= len(listPosition)-1): 
                cv2.circle(frame, listPosition[flag], 2, (255,255,0), 2)
                if "palm" in f and key!=1 and len(fingerdict[key]) < 4:
                    #if we do not find a starting landmark in fingers to join we will not do so
                    pass
                else:
                    cv2.line(frame, StartingPoint,fingerdict[key][0], (255,0,0), 1)
              
                if "palm" in f and len(fingerdict[key]) < 3:
                    # in this video we have incomplete landmarks on index ,middle and ring finger we do not want to join these
                    pass
                else:
                    if not (flag+1) == len(listPosition):
                        cv2.line(frame, listPosition[flag], listPosition[flag+1], (255,0,0), 1)#join landmarks startpoint, end points are coordinate

                flag = flag +1
                cv2.circle(frame, listPosition[len(listPosition)-1], 2, (255,255,0), 2)
            listPosition.clear()

        writer.write(frame) # wriite the frame to the video to create
        count = count + 1

    writer.release()
    cap.release()
    cv2.destroyAllWindows()