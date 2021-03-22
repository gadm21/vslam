

from utils import *
from processor import Processor






pro = Processor()


if __name__ == "__main__":

    cap = cv2.VideoCapture('data/vslam_test.mkv')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret : break
    

        newImage = pro.process(frame)
        cv2.imshow('video', newImage)
        if cv2.waitKey(25) & 0xFF == ord('q'): break
    

    cap.release() 
    cv2.destroyAllWindows()
    print("done")
