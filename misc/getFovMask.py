# # # # Disclaimer:
# #  This code is provided "as is". It can be used for research purposes only and all the authors
# #  must be acknowledged.
# # # # Authors:
# # Luca Giancardo
# # # # Date:
# # 2010-03-01
# # # # Version:
# # 1.0
# # # # Description:
# # get a binary image of the Field of View mask
from turtle import Shape, shape
import numpy as np
import cv2
from matplotlib import pyplot as plt
from sympy import is_nthpow_residue

def hist(GrayScaled):
        r=[]
        im=GrayScaled
        im=np.asarray(im)
        im=np.round(im)
        im=im.astype(int)
        h = [0]*256  
        for x in range(im.shape[0]):        
            for y in range(im.shape[1]):            
                i = round(im[x,y])   
                #specfic intensity
                if i<=256:
                    h[i] = h[i]+1     #had 1 of a specfic intensity then add one and repeat
                else:
                    pass
        for i in range (len(h)):
            r.append(i)
        newh=np.asanyarray(h)
        normalizedH=newh/(im.shape[0]*im.shape[1])
       
    
        return h

# remove the outside circle or remove the dark part of histogram with is coresponding to vessels?? 
def getFovMask(gImg,erodeFlag,seSize): 
    #GETFOVMASK get a binary image of the Field of View mask
# gImg: green challe uint8 image
# erodeFlag: if set it will erode the mask
#Param
    lowThresh = 0
    if  seSize==None:
        seSize = 10
    
    histRes = hist(gImg)
    plt.imshow(np.uint8(gImg))
    #print("hist" ,histRes)
    d = np.diff(histRes)
    #print("diff" ,d)
    lvlFound = np.where(d >= lowThresh)
    lvlFound=lvlFound[0]
    lvl=lvlFound[0]
    #print("level" ,lvl)
    fovMask = np.logical_not(gImg <= lvl) 
    fovMask2 = gImg > lvl 
    if (fovMask.any() == 1):
        print("total black")
    
    if ( erodeFlag > 0):
        # se = np.strel('disk',seSize)#########################
        se=cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE,ksize=(seSize,seSize))
        fovMask = cv2.erode(np.float32(fovMask),se)
      
        #erode also borders
        fovMask[1:seSize*2,:] = 0
        fovMask[:,1:seSize*2] = 0
        # fovMask[fovMask[-1,-1]-seSize*np.arange(2,fovMask[-1,-1]),:]
        # fovMask[:,fovMask[-1,-1]-seSize*np.arange(2,fovMask[-1,-1])] 
        fovMask[-1-seSize*2:-1,:] = 0
    
        
        fovMask[:,-1-seSize*2:-1] = 0
        # fovMask[-seSize*2,:]=0
        # fovMask[:,-seSize*2]=0
    print(np.uint8(fovMask)[0:100,0:100])
    plt.imshow(np.uint8(fovMask)   )
    plt.imshow(np.uint8(fovMask2))   
    return fovMask
    
