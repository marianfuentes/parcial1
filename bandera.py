#### LIBRARIES #####

import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from numpy.core._multiarray_umath import ndarray
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from kneed import KneeLocator #need to be installed!!!!
# Method for calculating distances between every point (in RGB space) and it´s closest cluster center (for an image)
def CalcDistance(centers, labels, rows, cols, image, n):
    distSum = np.zeros(n, dtype=np.float64) # aux accumulative variable
    label_idx = int(0) # index aux variable
    for i in range(rows): #For every pixel in a given image
        for j in range(cols):
            aux = labels[label_idx] # closest cluster to the pixel
            x1 = centers[aux,0] # take every x,y,z (RGB component) of the center of the cluster
            y1 = centers[aux,1]
            z1 = centers[aux,2]
            x2 = image[i,j,0] #take every x,y,z (again in RGB space) of the given pixel.
            y2 = image[i,j,1]
            z2 = image[i,j,2]
            # Calculate distance between pixel and cluster center
            dist = sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
            distSum[aux] = distSum[aux] + dist # Accumulate (sum) every distance for each of n clusters
            label_idx = label_idx + 1 # next pixel

    TotalSum = np.sum(distSum) # sum all of the n cluster distances
    return TotalSum

# Method for re-constructing clustered images, provided by Julian Quiroga
def recreate_image(centers, labels, rows, cols):
    d = centers.shape[1]
    image_clusters = np.zeros((rows, cols, d))
    label_idx = 0
    for i in range(rows):
        for j in range(cols):
            image_clusters[i][j] = centers[labels[label_idx]]
            label_idx += 1
    return image_clusters



class Flag:
    
    ########  Constructor  #######
    def __init__(self,path):
        #Receives path image of type string
        self.im_path = path
        
        #self.im_name = im_name
        #self.im_path = os.path.join(path, im_name)
       
        #Saves image in self
        self.img = cv2.imread(self.im_path)
    
    
    ####  Methods   ######
    
    #Uses Kmeans to determine automatically 4 number of colors
    def colors(self):
        
        nmax_colors = 4
        
        # Convert to floats instead of the default 8 bits integer coding. Dividing by
        # 255 is important so that plt.imshow behaves works well on float data (need to
        # be in the range [0-1])
        self.img = np.array(self.img, dtype=np.float64) / 255
        
        # Load Image and transform to a 2D numpy array.
        rows, cols, ch = self.img.shape
        image_array = np.reshape(self.img, (rows * cols, ch))
        
        image_array_sample = shuffle(image_array, random_state=0)[:int(rows*cols*0.05)]
        
        # in this variable we will store the sum of distances for eache n = 1,2, ... 10 clusters
        distance = np.zeros(10)
        
        n = int(0) # n clusters
        # this code is not efficient so we give a head up to user, calculating distances takes a lot of time !

        for f in range (nmax_colors): # for n clusters
            n = n + 1 # next cluster
            model = KMeans(n_clusters=n, random_state=0).fit(image_array_sample) # train model
            self.labels = model.predict(image_array) # get model labels
            centers = model.cluster_centers_ # get model center(s)
            distance[f] = CalcDistance(centers, self.labels, rows, cols, self.img, n) # calculate distances
            
        ### OPTIMAL CLUSTERING ####
        x = range(1, len(distance)+1)
        #optimal n_colors
        kn = KneeLocator(x, distance, curve='convex', direction='decreasing')
        self.n_colors = kn.knee
        self.newimage = recreate_image(centers, self.labels, rows, cols)
        return self.n_colors
        
    
        
        #Returns flag percetange colors
    def percentage(self):
        self.percentage = []

        for i in range(self.n_colors):
            self.l = self.labels.tolist()
            self.values = self.l.count(i) 
            for j in self.values:
                p = j*100/len(self.labels)
                self.percentage.append(p)
        return self.percentage
    
    
    def orientation(self):
        
        high_thresh = 300
        bw_edges = cv2.Canny(self.img, high_thresh * 0.3, high_thresh, L2gradient=True)

        hough = hough(bw_edges)
        accumulator = hough.standard_HT()

        acc_thresh = 50
        N_peaks = 11
        nhood = [25, 9]
        peaks = hough.find_peaks(accumulator, nhood, acc_thresh, N_peaks)

        [_, cols] = self.img.shape[:2]
        image_draw = np.copy(self.img)
        for i in range(len(peaks)):
            rho = peaks[i][0]
            theta_ = hough.theta[peaks[i][1]]

            theta_pi = np.pi * theta_ / 180
            theta_ = theta_ - 180
            a = np.cos(theta_pi)
            b = np.sin(theta_pi)
            x0 = a * rho + hough.center_x
            y0 = b * rho + hough.center_y
            c = -rho
            x1 = int(round(x0 + cols * (-b)))
            y1 = int(round(y0 + cols * a))
            x2 = int(round(x0 - cols * (-b)))
            y2 = int(round(y0 - cols * a))

            if np.abs(theta_) > 80 and < 100:
                return "vertical"
            else:
                return "horizontal"

