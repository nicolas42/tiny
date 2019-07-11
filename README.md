Using cfg instructions from 

https://medium.com/@manivannan_data/how-to-train-yolov2-to-detect-custom-objects-9010df784f36

    wget https://pjreddie.com/media/files/darknet19_448.conv.23
    ./darknet detector train tiny/obj.data tiny/yolo-obj.cfg tiny/darknet19_448.conv.23



# Colab AlexeyAB 

Edit > Notebook Settings > select GPU (also increases storage to ~350gb for some reason)

!git clone https://github.com/nicolas42/alexeyAB-darknet
!wget https://pjreddie.com/media/files/darknet19_448.conv.23
!mv darknet19_448.conv.23 alexeyAB-darknet/tiny

#%%capture
!./darknet detector train tiny/obj.data tiny/yolo-obj.cfg tiny/darknet19_448.conv.23
# -map -dont_show


# Edit a file in Colab

    p = """
    Yadda yadda
    whatever you want just don't use triple quotes.
    """

    c = """text_file = open("text.text", "w+");text_file.write(p);text_file.close()""" 

    exec(c)


# Upload Results to Dropbox
You'll need to follow the instructions to get an authentication key if you don't already have one.

    !git clone https://github.com/andreafabrizi/Dropbox-Uploader.git
    %cd Dropbox-Uploader/
    !chmod +x dropbox_uploader.sh
    !./dropbox_uploader.sh
    <need dropbox authentication key>
    ./dropbox_uploader.sh upload src dest




Material from Ivan Goncharov 
from https://github.com/ivangrov/YOLOv3-GoogleColab/blob/master/YOLOv3_GoogleColab.ipynb





#Here's how you can mount into your Google Drive, if you wanna
#from google.colab import drive
#drive.mount('/content/drive')

#Let's define some functions that will let us show images, and upload and 
#download files
def imShow(path):
  import cv2
  import matplotlib.pyplot as plt
  %matplotlib inline

  image = cv2.imread(path)
  height, width = image.shape[:2]
  resized_image = cv2.resize(image,(3*width, 3*height), interpolation = cv2.INTER_CUBIC)

  fig = plt.gcf()
  fig.set_size_inches(18, 10)
  plt.axis("off")
  #plt.rcParams['figure.figsize'] = [10, 5]
  plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
  plt.show()
  
  
def upload():
  from google.colab import files
  uploaded = files.upload() 
  for name, data in uploaded.items():
    with open(name, 'wb') as f:
      f.write(data)
      print ('saved file', name)
def download(path):
  from google.colab import files
  files.download(path)
In [0]:
#You should see a person, a dog and a horse here and they might even see you.....
imShow('predictions.jpg')















In [0]:
#!./darknet detector calc_anchors data/obj.data -num_of_clusters 9 -width 416 -height 416
In [0]:
#!cat data/obj.names
In [0]:
#!ls
#upload()
In [0]:
#!wget https://pjreddie.com/media/files/darknet53.conv.74
#!ls
In [0]:
#%%capture - uncomment if you wanna supress the output
#!./darknet detector train data/obj.data yolov3-obj.cfg backup/yolov3-obj_last6.weights -dont_show
In [0]:
#Get YOLOv3-tiny weights
#!wget https://pjreddie.com/media/files/yolov3-tiny.weights
#!ls
In [0]:
#Partial the weights
#!ls
#!./darknet partial yolov3-tiny-obj.cfg yolov3-tiny.weights  yolov3-tiny.conv.15 15
#!ls
In [0]:
#Process a video names test.mp4 and save is as result3.mp4
#!./darknet detector demo data/obj.data yolov3-tiny-obj.cfg backup/yolov3-tiny-obj_4000.weights -dont_show test.mp4 -out_filename result3.mp4
In [0]:
#Now, if you're here and it all works, then you've basically gotten yourself to the most important part:
#You have it all set up and installed and can do some damage
#Next I'll be showing how to process video and train your YOLO model here
In [0]:
#Now, let's figure out how to process some videos and then we'll get to the trainin'
#First, you gotta get the video somewhere here are your options:
#Spend a year manually coding every bit of a 'video' to Colab, upload one 
#from your machine, or download one from the Internet
#To upload one from your machine just call the upload() function, to
#download from the web use !wget and a download link

!ls
#Like this, though the video's not here
#!wget https://sv85.onlinevideoconverter.com/download?file=e4a0b1j9h7g6g6b1
!ls
#Also, don't forget to throw around a bunch of !ls's to just understand what's
#going on just a little
In [0]:
#Here in my case I had to rename the video
#!mv download?file=e4a0b1j9h7g6g6b1 fiddlevideo.mp4
#!ls
In [0]:
#Here's the command for processing video, make sure you have the weights and
#It'll be saved as result.avi

!./darknet detector demo cfg/coco.data cfg/yolov3.cfg yolov3.weights -dont_show fiddlevideo.mp4 -i 0 -out_filename fiddlevideo1.avi
In [0]:
#Then you can download the video to your machine, just call
#download('name_of_the_video') , and keep throwing !ls's !!
!ls
In [0]:
download('fiddlevideo1.avi')
In [0]:
#After you've trained your model you can also run it against an image
!wget https://s9.stc.all.kpcdn.net/share/i/12/10186046/inx960x640.jpg
!./darknet detector test data/obj.data yolov3-tiny-obj.cfg backup/yolov3-tiny-obj_1000.weights inx960x640.jpg -i 0 -thresh 0.15
#Show the image
imShow('predictions.jpg')
In [0]:
#Okay, so if you're familiar with AlexeyAB's darknet fork you know that we'll
#need 5 things:
#Obj folder (training data)
#Obj.data, Obj.name file
#train.txt file
#Weights and the edited CFG model 
#Go here https://github.com/AlexeyAB/darknet/ to figure out what you need
#So, you'll basically need to edit all that stuff on your machine and prepare
#it to be  uploaded, unless you're a ninja can do that all here, which I can't -_-


# Use %cd <directory> and %cd ../ to navigate the directories to put everything
#in place
In [0]:
#Upload cfg file (make sure that you're in darknet folder)
#upload()
#and check the results:
#!ls
#Just copy this code and repeat it for as long as you need
In [0]:
#Then call
#!unzip obj.zip
#And there you have a folder with your data
In [0]:
#Now, once you've put everything in place, you can run the command, in my case that was:
!./darknet detector train data/obj.data yolov3-tiny-obj.cfg yolov3-tiny.conv.15 -dont_show
#Don't forget the dont_show flag, it's good for your health ;)
In [0]:
#Once the model's trained you can go to the backup folder
#and call download('weights') to get the trained weights you want, in my case
download('yolov3-tiny-obj_1500.weights')
In [0]:
#So, there we go! I mean, the hard part is to get everything up and going 
#And you can do pretty much anything once you did that))
#Feel free to correct the errors (when something will stop working in the future)
#And if that was useful, my dear AI wizard, go and do some good and cool stuff with it! =)
















This is a collection of software to assist in the training of a single object detector neural net using the yolo model.  NFPA_net is an example of a net we've made.  boobs_annotator is a HTML based yolo anontation tool.  Image downloader is a golang concurrent image downloader.  split-data-from-snowman-demo is a script that hopefully will split our data into training and testing sets.  yolo_files is a collection of files needed to train yolo weights.

Basically you want to get a bunch of images, annotate them, choose a configuration (e.g. tiny or normal neural network) then change yolos configuration files accordingly and set it running on a big GPU.

Depending on what you've got you might want to start with the image downloader or annotator or whatever.


# Installation, Packages

It's probably a good idea to remove packages to prevent conflicts between different dependency management systems.  That means you apt remove.

    apt remove

Things I built last time I installed Linux Mint 19.1 Cinnamon (based on Ubuntu 18.04 LTS)

    apt install
    build-essential
    git
    golang-go
    python-tk
    python-pip
    libopencv-dev python3-opencv

How to install opencv https://milq.github.io/install-opencv-ubuntu-debian/

I think these packages were for an annotation tool which I don't plan to use now

    pip install
    setuptools
    wheel

To disable conda from automatically popping up whenever the shell is launched set auto_activate_base to false

    conda config --set auto_activate_base true

Use conda to install anaconda and jupyter.  I'm still iffy about opencv and pil.  I don't know what they're doing.

    conda install
    -c anaconda anaconda
    -c anaconda jupyter
    -c pytorch -c fastai fastai


fix conflicts by updating conda.  this will also downgrade packages as needed.

    conda update --all


to start a new jupyter notebook.  optionally put the notebook file you'd like to open after notebook

    jupyter notebook



# Getting Images

## Google Images Download

git clone https://github.com/hardikvasa/google-images-download.git
cd google-images-download && sudo python setup.py install

There's a limit of 100? images, a bit slow, execute example from within directory

## get images using OIDv4

https://www.learnopencv.com/fast-image-downloader-for-open-images-v4/

https://medium.com/@intprogrammer/how-to-scrape-google-for-images-to-train-your-machine-learning-classifiers-on-565076972ce

## Get image urls from a google images search

    urls = Array.from(document.querySelectorAll('.rg_di .rg_meta')).map(el=>JSON.parse(el.textContent).ou);
    window.open('data:text/csv;charset=utf-8,' + escape(urls.join('\n')));


# Annotation, Labelling

Boobs Annotation Tool

Made from HTML and javascript.  It seems to work pretty well.

    https://github.com/drainingsun/boobs.git

bbox label tool seems to be popular

Golang Links
The computer vision systems made by the Go people might be more stable then the python guys but I haven't tested this yet.

    http://cavaliercoder.com/blog/downloading-large-files-in-go.html
    https://gocv.io/
    https://awesome-go.com


Fast AI Lessons, Jupyter notebook

There is a comprehensive guide to machine learning in this course.  Videos and information is available at fast.ai

    git clone https://github.com/fastai/fastai.git


# Getting a video out of yolo

(from LinderPi 10-jan-2019 https://github.com/pjreddie/darknet/issues/1235)

    ./darknet detector demo cfg/coco.data cfg/yolov3.cfg yolov3.weights <video.mp4> -prefix pictures

Where the hell are the specs for the command line parameters for yolo??? Do I need to look up the source?



# Make jpgs into mp4

from https://askubuntu.com/questions/610903/how-can-i-create-a-video-file-from-a-set-of-jpg-images

You can use ffmpeg, which you can install with

    sudo apt install ffmpeg

This is the command all together:

    ffmpeg -framerate 25 -i image-%05d.jpg -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p output.mp4

Let me break it down:

-framerate
is the number of frames (images) per second

-i scene_%05d.jpg
this determines the file name sequence it looks for. image- means all the files start with this. 0 is the number repeated, and the 5 is number of times (so it is looking for any file starting at image-00000.jpg. The d is telling it to count up in whole numbers, so the files it will detect are everything from image-00001 to image-99999.

-c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p
-c:v libx264 - the video codec is libx264 (H.264).

-profile:v high - use H.264 High Profile (advanced features, better quality).

-crf 20 - constant quality mode, very high quality (lower numbers are higher quality, 18 is the smallest you would want to use).

-pix_fmt yuv420p - use YUV pixel format and 4:2:0 Chroma subsampling

output.mp4
The file name (output.mp4)

Remember that ffmpeg needs a continuous sequence of images to load in. If it jumps from image-00001 to image-00003 it will stop.

If you images are like this:

image-1
image-2
...
image-35
then change the -i part to -i image-%00d.

Update. Your edit says the pattern is image-01.jpg to image-02.jpg. That means you need the image-%02d.jpg pattern.




# Cut a video using ffmpeg

(from https://stackoverflow.com/questions/18444194/cutting-the-videos-based-on-start-and-end-time-using-ffmpeg)

    ffmpeg -i movie.mp4 -ss 00:00:03 -t 00:00:08 -async 1 cut.mp4

You probably do not have a keyframe at the 3 second mark. Because non-keyframes encode differences from other frames, they require all of the data starting with the previous keyframe.

With the mp4 container it is possible to cut at a non-keyframe without re-encoding using an edit list. In other words, if the closest keyframe before 3s is at 0s then it will copy the video starting at 0s and use an edit list to tell the player to start playing 3 seconds in.

If you are using the latest ffmpeg from git master it will do this using an edit list when invoked using the command that you provided. If this is not working for you then you are probably either using an older version of ffmpeg, or your player does not support edit lists. Some players will ignore the edit list and always play all of the media in the file from beginning to end.

If you want to cut precisely starting at a non-keyframe and want it to play starting at the desired point on a player that does not support edit lists, or want to ensure that the cut portion is not actually in the output file (for example if it contains confidential information), then you can do that by re-encoding so that there will be a keyframe precisely at the desired start time. Re-encoding is the default if you do not specify copy. For example:

ffmpeg -i movie.mp4 -ss 00:00:03 -t 00:00:08 -async 1 cut.mp4
When re-encoding you may also wish to include additional quality-related options or a particular AAC encoder. For details, see ffmpeg's x264 Encoding Guide for video and AAC Encoding Guide for audio.

Also, the -t option specifies a duration, not an end time. The above command will encode 8s of video starting at 3s. To start at 3s and end at 8s use -t 5. If you are using a current version of ffmpeg you can also replace -t with -to in the above command to end at the specified time.




# how to use an ssh key with github

In .git/config
change the line "url = " to point to an ssh url like this

    url = https://github.com/JcTechnology/jctech-ml.git 
    ->
    url = ssh://git@github.com/JcTechnology/jctech-ml.git

where git is actually the word git.

(https://stackoverflow.com/questions/9960897/why-doesnt-my-ssh-key-work-for-connecting-to-github)




# Goterm
https://github.com/buger/goterm
advanced terminal output.  looks cool.


https://en.wikipedia.org/wiki/File_system_permissions



# Remove last commit from remote git repository

Be careful that this will create an "alternate reality" for people who have already fetch/pulled/cloned from the remote repository. But in fact, it's quite simple:

git reset HEAD^ # remove commit locally
git push origin +HEAD # force-push the new HEAD commit
If you want to still have it in your local repository and only remove it from the remote, then you can use:

git push origin +HEAD^:<name of your branch, most likely 'master'>


# What Bash colors mean

Blue: Directory
Green: Executable or recognized data file
Sky Blue: Symbolic link file
Yellow with black background: Device
Pink: Graphic image file
Red: Archive file
Red with black background: Broken link

# Find command in unix

    $ find [where to start searching from] [expression determines what to find] [-options] [what to find]

    e.g. find ./GFG -name *.txt 

https://www.geeksforgeeks.org/find-command-in-linux-with-examples/

# How to install opencv in linux

(from https://milq.github.io/install-opencv-ubuntu-debian/)

Install OpenCV on Ubuntu or Debian is a bit long but very easy. You can install OpenCV from the Ubuntu or Debian repository or from the official site.

Option 1. Install OpenCV from the Ubuntu or Debian repository
You can install OpenCV from the Ubuntu or Debian repository:

    sudo apt-get install libopencv-dev python3-opencv

However, you will probably not have installed the latest version of OpenCV and you may miss some features.

Option 2. Install OpenCV from the official site
To install the latest version of OpenCV be sure that you have removed the library from the repository with 

    sudo apt-get autoremove libopencv-dev python-opencv and follow the steps below.

2.1. Run an installation script
The most simple and elegant way to install a library is running an installation script.

Download the installation script install-opencv.sh, open your terminal and execute:

    bash install-opencv.sh

Type your sudo password and you will have installed OpenCV. This operation may take a long time due to the packages to be installed and the compilation process.

2.2. Execute some OpenCV examples
Go to your OpenCV directory and execute a C++ example:

cd build/bin
./example_cpp_edge ../../samples/data/fruits.jpg
Now, go to your OpenCV directory and execute a Python example:

cd samples/python
python3 video.py
Finally, go to your OpenCV directory and execute a Java example:

cd samples/java/ant
ant -DocvJarDir=../../../build/bin -DocvLibDir=../../../build/lib
2.3. Compile a demonstration
Download the files demo.cpp and CMakeLists.txt and put them into a folder. Now, open your terminal, go to the folder and execute:

mkdir build && cd build && cmake .. && make
Finally, run the demo: ./demo.

And that's it! You have installed OpenCV, run some examples, and compiled OpenCV code!

Do you like this article? Share it with this link. Thanks for reading!

# I have to learn more about git reverts and resets

I need to read up on this.

https://www.atlassian.com/git/tutorials/undoing-changes

https://www.atlassian.com/git/tutorials/undoing-changes/git-revert

This will show all of the hashes of your commits elegantly
 
    git log --oneline




# How to use with ec2

ssh ubuntu@13.236.3.140

clone jctech-ml

    git clone https://github.com/JcTechnology/jctech-ml
    cd jctech-ml

clone darknet and make it

    git clone https://github.com/AlexeyAB/darknet.git

download this and put it in the bitwise dir

    wget https://pjreddie.com/media/files/darknet19_448.conv.23
    mv darknet19_448.conv.23 bitwise 

put the bitwise directory in the darknet directory then run this from the darknet directory

    cp -r bitwise darknet
    cd darknet
    ./darknet detector train bitwise/obj.data bitwise/yolo-obj.cfg bitwise/darknet19_448.conv.23



# Other Stuff

    sudo apt-get install eog scrot

you can check on progress from another computer with
gnome viewer tool eog

    scp ubuntu@13.236.3.140:~/jctech-ml/darknet/chart.png chart.png; eog chart.png


Taking screenshots easily is also good

    scrot screenshot.png; eog screenshot.png





# Misc Notes


based on this https://medium.com/@manivannan_data/how-to-train-yolov2-to-detect-custom-objects-9010df784f36


get some premade training weights
wget https://pjreddie.com/media/files/darknet19_448.conv.23


put these files in the darknet/cfg directory
obj.data
obj.names
yolo-obj.cfg

put these files and folders in thet darknet directory
train.txt
test.txt
darknet19_448.conv.23
nfpa_dataset

run this
./darknet detector train cfg/obj.data cfg/yolo-obj.cfg darknet19_448.conv.23

./darknet detector train obj.data yolo-obj.cfg darknet19_448.conv.23


## How To

copy from ec2
scp -r -i ~/.ssh/id_rsa.pub ubuntu@13.236.66.105:/home/ubuntu/output ~/output

test weights
./darknet detector test cfg/obj.data cfg/yolo-obj.cfg yolo-obj_11000.weights darknet_training_demo/nfpa_dataset/pos-1.jpg


test weights
./darknet detector test obj.data yolo-obj.cfg yolo-obj_11000.weights pos-286.jpg

./darknet detector test bitwise/obj.data bitwise/yolo-obj.cfg bitwise/darknet19_448.conv.23


# Commands

generic commands

    scp <file> ubuntu@13.211.94.212:~

    ./darknet detector demo cfg/coco.data cfg/yolov3.cfg yolov3.weights <video.mp4> -prefix pictures

    ffmpeg -framerate 25 -i image-%05d.jpg -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p output.mp4

    ./darknet detector demo ../bitwise/obj.data ../bitwise/yolo-obj.cfg ../bitwise/backup/yolo-obj_20000.weights VIRB0040.mp4 -prefix pictures


# Command History

join images together to make a video

    ffmpeg -framerate 25 -i pictures_%08d.jpg -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p output.mp4


    scp ubuntu@13.211.94.212:~/tmp/jctech-ml-vineyard-video-40/darknet/pictures/output.mp4 ~/dev/output.mp4


run detection on video and output images.  requires opencv

    ./darknet detector demo ../bitwise/obj.data ../bitwise/yolo-obj.cfg ../bitwise/backup/yolo-obj_20000.weights VIRB0040.mp4 -prefix pictures
    ./darknet detector demo bitwise/obj.data bitwise/yolo-obj.cfg backup/yolo-obj_90000.weights VIRB0030.mp4 -prefix pictures

# Notes 29-June-2019

How to prevent overfitting on yolo
https://www.one-tab.com/page/JfFOCjHjQEilzTJV-QhRFg


# Notes 7-July-2019

[x] annotate video 30
annotate video 37
annotate google images dataset
[ ] check for false negatives and false positives and retrain network including those corrected images.
test larger core
find out about overfitting
test ultralytics code

[x] investigate annotation with polygons instead of rectangles
[ ] Use capture image function in vlc.  run through vlc and use capture image function (in advanced tools) to select images.  this way perhaps we can get better quality images.
[ ] post downloader tool to reddit
[ ] increase maximum resolution of yolo layers to 1080p??? at least the size of our images???


The darknet detector can be trained, tested, or demoed.  Training takes a data, config, and weights file.  The data file points to the training files (images, labels) and various other files necessary,  the config file describes the architecture of the neural net and the weights file is the actual values of the neural net and is usually quite large.  Testing takes those same arguments plus an image that we want to detect objects in.  Demoing takes a video file.  I don't know how to output video yet for various reasons but we can output a number of images with the option -prefix <prefix>.  <prefix> is the beginning of the filenames of the output files, e.g. the prefix "picture" would give filenames picture0001 picture0002 and so on.


Upon first glance there doesn't seem to be an easy way to do polygonal bounding boxes.  An oriented rectangle was mentioned which would mean only one more paramter would be needed, i.e. an angle.  Still, this is an interesting direction to look into in the future.


Put annotation data into "annotation data" folder in github so I don't lose it.  Nearly went insane this morning annotating grapes.

Boobs annotator has a "Restore" button which will restore all the bounding box annotations from memory if you accidentally refresh or close your browser. :)

detection small object from big images discussion https://github.com/pjreddie/darknet/issues/1535

4-July-2019

Data annotation, renaming..
To merge different datasets you may have to rename the files.  Prepending filenames in bash is easy.

    for f in *; do echo $f ; done

    mv $f prepend$f




9-July-2019

a=1
for i in *.jpg; do
    
  newjpg=$(printf "%04d.jpg" "$a") #04 pad to length of 4
  newtxt=$(printf "%04d.txt" "$a") #04 pad to length of 4

  mv -i -- "$i" "$newjpg"
  mv -i -- "${i%.jpg}.txt" "$newtxt"


  let a=a+1
done



10-july-2019

./darknet detector demo bitwise/obj.data bitwise/yolov3.cfg backup/yolov3_900.weights /home/ubuntu/20190709/alexeyAB-darknet/VIRB0030.mp4 -prefix pictures

# Colab
!git clone https://github.com/nicolas42/nick-darknet.git
%cd nick-darknet
!make
cd bitwise
!wget https://pjreddie.com/media/files/darknet53.conv.74
cd ../
!./darknet detector train bitwise/obj.data bitwise/yolov3.cfg bitwise/darknet53.conv.74


# EC2

git clone https://github.com/nicolas42/nick-darknet.git
cd nick-darknet
make
cd bitwise
wget https://pjreddie.com/media/files/darknet53.conv.74
#wget https://pjreddie.com/media/files/darknet19_448.conv.23
#wget https://www.dropbox.com/s/yvca2um3dketvxi/first-test-core-yolo-obj_20000.weights?dl=0
cd ../
./darknet detector train bitwise/obj.data bitwise/yolov3.cfg bitwise/darknet53.conv.74


# Test the overfit tiny core

./darknet detector test bitwise/obj.data bitwise/yolo-obj.cfg bitwise/first-test-core-yolo-obj_20000.weights bitwise/test/40_scene01251.jpg

./darknet detector test bitwise/obj.data bitwise/yolo-obj.cfg bitwise/first-test-core-yolo-obj_20000.weights < bitwise/test.txt > bitwise/log.txt


# Plan for annotation with boobs

Boobs will add a txt file for each file that it loads so it is important to first delete all the images that contain the object that I don't want to annotate

1. rename images - make sure there's no intersections
2. REMOVE BAD IMAGES - which contain the object but I don't want to annotate
3. annotate using boob
4. update train.txt  "ls -1 path/i/want/to/list/*.jpg > train.txt"
5. update the data file and the cfg file
6. train  - darknet detector train data cfg weights 

## Rename files to a numbered pattern. Change a=1 as desired
a=1
for i in *.jpg; do
  newjpg=$(printf "%04d.jpg" "$a") #04 pad to length of 4
  mv -i -- "$i" "$newjpg"
  let a=a+1
done



# Ultralytics

conda install numpy opencv matplotlib tqdm pillow
conda install pytorch torchvision -c pytorch
conda install -c conda-forge scikit-image

python3 train.py --data data/obj.data
# tiny
# tiny
