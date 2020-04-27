This project contains the code for the individual assignment of the course by the MAVLAB:
Autonomous Flight of Autonomous Vehicles. The code present a gate detector algorithm
from the data setup, passing by the training, testing, until the analysis of the data.
The algorithm uses a YOLO CNN developed in: http://pjreddie.com/darknet/.

For the user to use the program, (s)he first has to upload a folder with the images
and the corresponding csv of ground truth gates to the main folder. The name of this 
folder should be 'WashingtonOBRace' or can be otherwise changed by changing the variable 
named 'folder_imgs' in the file SetupData.py to the direction of the corresponding
folder. 



4 Steps are necessary to perform the whole analysis of the data.

1. Run the command: <python3 src/SetupData.py> with your preferred values.
Default will be the default image size and a video playback speed of 1.

(Optional)
2. Train the darknet network to the images you uploaded. To do this you need to go to the 
darknet folder. It is first important to make the directory, so run the <make> command.
If you want to make use of the GPU on your computer or CUDNN, change the corresponding 
values in the Makefile to 1. You are now ready to train your network.
Run the command <sh train.sh>. This will start training the data. It will take a while.
The training will stop after 8000 batches.

3. If you trained your own CNN, it is time to predict the test data to perform the 
analysis on. Run the command <sh test.sh -w backup/obj_final.weights>. The output will
be placed on the folder 'predictions_SIZE_XxSIZE_Y' of 'GateDetector'. A list of 
pretrained weights are available in "weights/" with the corresponding image size.

3.2 If you want to see your gate detector in action live you can run the command
<sh video.sh> and will see the detector working on a video. 

4. To perform data analysis, got back to the 'GateDetector' folder and run the command
<python3 AnalyseResults.py>. Enter the name of the folder with the data generated, or see
the default examples used. An image showing the results of an exemplary gate is showed.