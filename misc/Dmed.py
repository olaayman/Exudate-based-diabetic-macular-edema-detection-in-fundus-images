import numpy as np
import os
import matplotlib.pyplot as plt
from .DatasetRet import DatasetRet
import cv2
import gzip
from .ReadGNDFile import ReadGNDFile
import re
from pathlib import Path
import win32com.client
import matplotlib.image as mpimg

# Class to access the Diabetic Macular Edema Dataset (DMED)
class Dmed(DatasetRet):
     
    def __init__(self,dirIn): 
        # DatasetCSV  constructor
      
        # Set constants(extensions)
        self.roiExt = '.jpg.ROI'
        self.imgExt = '.jpg'
        self.metaExt = '.meta'
        self.gndExt = '.GND'
        self.mapGzExt = '.map.gz'
        self.mapExt = '.map'
        self.baseDir =dirIn
        
        # Declare a dictionary data type to store the images
        self.data=dict([])
        idxData = 0

        for file in os.listdir(self.baseDir): # Loop through the image
            if file.endswith(self.imgExt): # Check if its jpg image
                dirList=os.path.join(self.baseDir,file) # Concatuate the image path
                self.data[idxData],ext= os.path.splitext(file) # Save file name in data
                idxData=idxData+1 # Increment the index
       
        self.origImgNum = len(self.data) # Save No. of images
        self.imgNum = self.origImgNum 
        self.idMap = np.arange(0,self.imgNum) # Save image id

            
    def getNumOfImgs(self): 
        # Return no. of images.

        imgNum = self.imgNum
        return imgNum 
        
        
    def getImg(self ,id ): 
        # Return the original image.

        if (id < 0 or id > self.imgNum):
            # Exception handling (if image index was incorrect)
            img = []
            raise Exception('Index exceeds dataset size of '+str(self.imgNum))
        else: 
            # Concatenate image address and save it.
            imgAddress=os.path.join(self.baseDir,self.data[self.idMap[id]]+self.imgExt)
            # Read the image
            img = plt.imread(imgAddress)
            
        
        return img
        
            
    def getONloc(self,id): 
        # Return row & col.

        # Initialize empty row & col array
        onRow = []
        onCol = []
        if (id < 0 or id > self.imgNum):
            # Exception handling (if image index was incorrect)
            raise Exception('Index exceeds dataset size of '+str(self.imgNum))
        else:
            # Save the meta file address
            metaFile=self.baseDir+'/'+self.data[self.idMap[id]]+self.metaExt
            fMeta = open(metaFile,'r') # Open the meta file
            if (fMeta):
                res=fMeta.read() # Read the meta file and save its contain
                fMeta.close() # Close the meta file

                # Search for the last 2 lines, which will contain row & col.
                tokRow = re.search('ONrow\W+([0-9\.]+)',res) 
                tokCol = re.search('ONcol\W+([0-9\.]+)',res)
                if (tokRow  and  tokCol):
                    # Separate the row & col and save them
                    onRow = int(tokRow.group().split('~')[1])
                    onCol = int(tokCol.group().split('~')[1])
        
        return onRow,onCol

  