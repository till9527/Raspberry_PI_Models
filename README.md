Setting up Raspberry PI:

Make sure to use bookworm. This is because bookworm uses python 3.11.2

Follow along with this tutorial:

https://m.youtube.com/watch?v=ALsH6zU4TVM

Refer to my GitHub for the code:

https://github.com/till9527/Raspberry_PI_Models/tree/master

Training the model:

The same process for training models through roboflow, except set it to yolov8n.pt in the training code (Training_model.py from the QCar_virtual repository). You can then replace the best.pt under “model” from the GitHub code with the newly trained one

To get the code running with the ncnns, you just gotta run the ncnn_training.py script once. This will then generate the ncnn model in the “model” folder. The imgsz that you specify is what the models like “best_ncnn_64” refer to (in that case, the imgsz was set to 64)

Note: with the ESP32 camera make sure you’re on Quanser_UVS WiFi, otherwise it won’t work

For setting up the ESP32 camera, refer to this video. Note, you won’t need any wires or breakout boards, the ESP32s we have already come with the pins programmed (since we have a second board on them):

https://m.youtube.com/watch?v=hSr557hppwY

Running basic code on the PI AI camera:

First, run sudo apt-install imx500-all

Then, run this:

rpicam-hello -t 0s --post-process-file /usr/share/rpi-camera-assets/imx500_mobilenet_ssd.json --viewfinder-width 1920 --viewfinder-height 1080 --framerate 30

Training a custom model:

This roughly follows this video, except unlike him we are running on windows and setting up wsl:

https://www.youtube.com/watch?v=I69lAtA2pP0&themeRefresh=1

First, install WSL following this tutorial, and choose Ubuntu as the distribution. On the lab computer I just made the username and password both “user”

https://www.youtube.com/watch?v=QM3mzEJCzjY

Then, you wanna make a new directory and cd into it:

mkdir imx_project_linux 

cd imx_project_linux

Then, install jre version 21:

sudo apt install -y python3 python3-pip openjdk-21-jre

Then install deadsnakes, to get an older python version since 3.12 won’t work:

sudo add-apt-repository ppa:deadsnakes/ppa -y

sudo apt install python3.11 python3.11-venv -y

Then, make a python virtual environment:

python3.11 -m venv imx_export_venv

source imx_export_venv/bin/activate

Then, install ultralytics and the converter tool:

pip install ultralytics

pip install imx500-converter[pt]

Then, you need to install the correct version of pytorch, since 2.9 is incompatible with imx:

pip uninstall torch torchvision torchaudio -y

pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.2.0 --extra-index-url https://download.pytorch.org/whl/cu121

Then, delete numpy and reinstall it to the correct version:

pip uninstall numpy -y

pip install numpy==1.26.4

So now, you’ll want to run the code found in the repository (yolo_train.py) with these specific arguments:

python yolo_train.py --export_only --init_model best.pt --export_format imx --int8_weights --export_config coco8.yaml --image_size 640x640

Note: I put my best.pt in there, but you will want to train your own best.pt using yolov8n. The best.pt must be in the same directory as yolo_train.py

Once that is done, copy it somewhere to the pc, for example I did:

cp -r best_imx_model /mnt/c/Users/user/Desktop/

Now, go to your raspberry pi that has bookworm OS installed and follow these instructions:

https://docs.ultralytics.com/integrations/sony-imx500/#supported-tasks

To make a venv for installing the aitrios library, do this:

python -m venv ai_cam

source ai_cam/bin/activate

When running the code, make sure that you don’t have any spaces in the directory, because it will confuse them for commands. 

Once that is done, make sure you’re in the virtual environment and cd into the directory from my GitHub repository, and you can run python run_ai_model.py
