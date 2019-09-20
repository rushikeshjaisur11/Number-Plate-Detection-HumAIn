# Number-Plate-Detection-HumAIn

# **Introduction:**
Given project is divided into two parts 

1.Data Extraction and Model Generation :- 

	In this step we extract dataset from the Indian_Number_plates.json file.
	
	We used VehicleDataDownload.ipynb jupyter notebook for downloading the vehicle images from the json file
	
	In Model generator foder we created a notebook named as model_generator.ipynb, this create a CNN model for the OCR.
	
2.Prediction 

	In this step we creted a PlateRecognize.py file that takes input from the user and output the predicted number plate of the vehicle.
	
	

# **Softwares and Technology used :**
- OpenCV 
- PIL
- Python 3
- Tensor Flow 1.14
- Tesseract OCR
- Anaconda package manager for python 3

# **How to run program :**

- Step 1.Open command line (in windows ) or Terminal 
- Step 2.Run the command python PlateRecognize.py
- Step 3.Enter the file name ( image name ) with extension (ENTER Image Name if image in same directory else write dirname/imagename.extension) 
- Step 4.Get the result
- OR 
- Step 1. Open PlateRecognize.py in idle or any other IDE and run the program
- Step 2. Enter image name (ENTER Image Name if image in same directory else write dirname/imagename.extension)
- Step 3. Get the result