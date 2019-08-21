import os
import time

import datetime
import utils.videosplit_sklearn
import ImageDetect
import cv2
from PIL import Image
import pymongo


if __name__ == '__main__':
    name = str(input('Enter the name of the video: '))
    (vdolength,totalFrames) = utils.videosplit_sklearn.Launch(name)
	# The name of the folder to store the frames of the video
    os.chdir('data')

 