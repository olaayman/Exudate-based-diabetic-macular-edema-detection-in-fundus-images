"Exudate-based diabetic macular edema detection in fundus images using publicly available datasets"

This is the python code referenced from the paper with the previous title
Authors:
Mohamed Abdelhalem Hafez
Noha Tarek Elboghdady
Ola Ayman Elmaghraby

//////////////////////////////////////////////////////////////

prerequsities:
cv2
numpy
matplotlib
scipy
os
skimage
re 

//////////////////////////////////////////////////////////////

File structure:
project --> DMED (images dataset)
	--> misc --> DatasetRet.py
		 --> Dmed.py
		 --> getFovMask.py
		 --> kirschEdges.py
		 --> ReadGNDFile.py
	--> exDetect.py
	--> test.py

//////////////////////////////////////////////////////////////

How to run:
run test.py script
then the original image and the exudate detected image will show in a figure
you need to close the figure to proceed to the next image till they finish
to terminate the run ctrl+c in the terminal

//////////////////////////////////////////////////////////////

Code Flow:
load data set using Dmed class
call function getNumofImages to iterate over the images:
	call getImage(img_id) function
	get the location of optic nerve from the meta data of the image using getONloc(img_id) function
	call exDetect(rgbImg, 1, onY, onX) Function which call getLesions() function:
		map the height to 750 with keeping image ratio
		call findGoodResolutionForWavelet() 
		resize the image to the new size
		get the Green Channel
		convert RGB to HSV color model
		get the intensity channel
		get the coordinates of the optic nerve to be removed
		get the FOVMask using getFovMask() function
		remove the optic nerve from the FOV mask
		estimate the backgroud using median filter medfilt2d() with kernel size 1/30 from image height
		then we will enhance the normalization with morphological-reconstrunction using imreconstruct() function
		subtract the backgroud image from the original image
		detect exudates edges using kirschEdges() function
		label the 8 connectivity of the exudates edges after removing ON 
		get the propability of lessions candidates from 8 connectivity using regionprops() function
		resize the image
		Display original image and exudate detected image 