This project contains the code for the individual assignment of the course by 
the MAVLAB: Autonomous Flight of Autonomous Vehicles. The code present a gate 
detector algorithm from the data setup, passing by the training, testing, until
the analysis of the data. The algorithm uses a YOLO CNN developed in: 
http://pjreddie.com/darknet/.

To run the program, the user first has to upload a folder with the images and 
the corresponding csv of ground truth gates to the main folder. The name of this
folder should be 'WashingtonOBRace' or can be otherwise changed by changing the 
variable named 'folder_imgs' in the file 'SetupData.py' to the direction of the 
corresponding folder. Extracting the folder provided by the lecturers of the 
course is enough.


4 Steps are necessary to perform the whole analysis of the data. Step 1 deals 
with setting up the data in the correct format. Step 2 (optional) trains the 
NN weights. Step 3 runs the algorithm on the data setup. Step 4 analyse the 
results.

1. Run the command with your preferred values on the UI. Default will be the 
default image size and a video playback speed of 1.

    >python3 src/SetupData.py

2. (Optional) Train the darknet network to the images you uploaded. It is first 
important to 'make' the darkent directory, so run the command on the folder. If 
you want to make use of the GPU on your computer, CUDNN or OpenCV, change the 
corresponding values in the Makefile to 1, and make the directory again with 
every change you perform. You are now ready to train your network. Run the 
command:

    >sh train.sh

    This will start training the data. It will take a while. The training will 
    stop after 8000 batches.


3. 1. If you trained your own CNN, it is time to predict the test data to 
perform the analysis on. Run the command 
        >sh test.sh -w backup/obj_final.weights
    
        The output will be placed on the folder 'predictions_SIZE_XxSIZE_Y' of 
        'GateDetector'. A list of pretrained weights are available in "weights/" 
        with the corresponding image size. When no command -w is used, the 
        default is 'weights/size_360.weights'

    2. If you want to **see the gate detector in action live** you can run the 
   command. Step 1 must be performed and OpenCV must be activated from step 2.
   (Only works for square images)
        >sh video.s

4. To perform data analysis, run the command
    >python3 src/AnalyseResults.py

    Enter the name of the folder with the data generated, or see the default 
    examples used. An image showing the results of an exemplary gate is showed.
    If you want to perform data analysis on your own custom size, you first need
    to run the command from section 3.1 ("sh test.sh")

If you have further questions about the usage of any of the bash files, you can 
use the help documentation running for example:
>sh train.sh -h