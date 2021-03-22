
from utils import * 
from math import sqrt



class Processor(object):

    def __init__(self):

        self.W = 500
        self.H = 500

        self.last = None
        
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher( )
    
    def process(self, image, drawKeyPoints = False, drawLines = True):
        image = cv2.resize(image, (self.W, self.H))
        processedImage = image.copy()
        
        feats = cv2.goodFeaturesToTrack(np.mean(image, axis = 2).astype(np.uint8), 1000, qualityLevel = 0.1, minDistance = 2)
        kps = [cv2.KeyPoint(x,y,6) for x,y in [feats[i,0,:] for i in range(feats.shape[0])]] 
        kps, dess = self.orb.compute(image, kps) 

        if self.last is not None :
            matches = self.bf.knnMatch(dess, self.last['dess'], k= 2)
            goodMatches = [m for m,n in matches if m.distance < 0.5*n.distance]
            
            currentKeyPointsMatched = [kps[goodMatch.queryIdx] for goodMatch in goodMatches]
            lastKeyPointsMatched = [self.last['kps'][goodMatch.trainIdx] for goodMatch in goodMatches]
            dists = np.array([self.dist(kp1, kp2) for kp1, kp2 in zip(currentKeyPointsMatched, lastKeyPointsMatched)])
            threshold = np.mean(dists) + (np.std(dists)*3)

            if drawLines :
                for i in range(len(dists)):
                    if dists[i] > threshold : break 
                    kp1 = currentKeyPointsMatched[i]
                    kp2 = lastKeyPointsMatched[i]
                    startPoint = (int(kp1.pt[0]), int(kp1.pt[1]))
                    endPoint = (int(kp2.pt[0]), int(kp2.pt[1]))

                    cv2.line(processedImage, startPoint, endPoint, (255, 0, 0), thickness = 1)
        
            if drawKeyPoints:
                cv2.drawKeypoints(processedImage, kps, color = (0,255,0), outImage = processedImage)
                cv2.drawKeypoints(processedImage, self.last['kps'], color = (0,255,0), outImage = processedImage)

        self.last = {'image': image, 'kps': kps, 'dess': dess}
        return processedImage

    def drawLastKeyPoints(self, image):
        if self.last is None: return
        cv2.drawKeypoints(image, self.last['kps'], color = [0, 255, 0], outImage = image)
    
    def dist(self, kp1, kp2):
        p1 = kp1.pt
        p2 = kp2.pt
        return sqrt( ((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2) )
    