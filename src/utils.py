

import time
import cv2
import numpy as np 



def process_frame(frame, orb):

    image = cv2.resize(frame, (W, H))
    newImage = image.copy()

    feats = cv2.goodFeaturesToTrack(np.mean(image, axis = 2).astype(np.uint8), 100, qualityLevel = 0.2, minDistance = 2)
    kps = [cv2.KeyPoint(x,y,3) for x,y in [feats[i,0,:] for i in range(feats.shape[0])]] 
    kps, deses = orb.compute(image, kps) 
    


    cv2.drawKeypoints(newImage, kps, color = [0, 255, 0], outImage = newImage)

    return {'original': image, 'newFrame': newImage, 'feats': feats}
    

