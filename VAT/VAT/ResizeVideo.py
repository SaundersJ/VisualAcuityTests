import cv2 as cv
from pathlib import Path
from datetime import datetime

def resizeVideo(fileName):
    split = fileName.split(".")
    resizedFileName = split[0] + "_resized.avi"
    my_file = Path(resizedFileName)

    if not my_file.is_file():
        print("Resizing File")
        window_resize = "Resize"
        cv.namedWindow(window_resize)
        cap = cv.VideoCapture(fileName)
        out = cv.VideoWriter(resizedFileName, cv.VideoWriter_fourcc(*'XVID'), 30.0, (480, 270))

        frameCount = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

        #Resize the file
        millis = datetime.now().microsecond
        i = 1
        while True:
            i = i + 1
            print(i/frameCount)
            ret, frame = cap.read()
            frame = cv.resize(frame, (480, 270), interpolation = cv.INTER_LINEAR)
            out.write(frame)
            key = cv.waitKey(30)
            cv.imshow(window_resize, frame)
            if i == frameCount or key == ord('q') or key == 27:
                break
            if i % 1000 == 0:
                newMillis = datetime.now().microsecond
                timeDifference = millis - newMillis
                remainingFrames = (frameCount - i)
                totalEstimatedTime = ((remainingFrames/1000) * timeDifference) / 1000
                print("=======[Seconds Remaining]======")
                print(i)
                print("------")
                print(frameCount)
                print(totalEstimatedTime)
                print("================================")
                millis = newMillis
                #get time per 1000 frames
                #multiiply the time by number of 1000 frames left
            
            
        cap.release()
        out.release()
        cv.destroyAllWindows()
        print("Finished Resize")
    return resizedFileName