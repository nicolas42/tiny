git clone https://github.com/ultralytics/yolov3.git
git clone https://github.com/pdollar/coco
git clone https://github.com/nicolas42/video-40.git

mkdir -p coco/images/val2014
mkdir -p coco/labels/val2014

cp video-40/dataset/*.jpg coco/images/val2014
cp video-40/dataset/*.txt coco/labels/val2014
cp video-40/config-ultralytics/* yolov3/data

cd yolov3
python3 train.py --data data/obj.data

to resume
python3 train.py --data data/obj.data --resume


# Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

conda install numpy opencv matplotlib tqdm pillow
conda install pytorch torchvision -c pytorch
conda install -c conda-forge scikit-image



# How to Upload Results to Dropbox
You'll need to follow the instructions to get an authentication key if you don't already have one.

git clone https://github.com/andreafabrizi/Dropbox-Uploader.git
cd Dropbox-Uploader/
chmod +x dropbox_uploader.sh
./dropbox_uploader.sh
<need dropbox authentication key>

/dropbox_uploader.sh upload src dest



# Use with AlexeyAB Darknet

git clone https://github.com/alexeyAB/darknet.git
git clone https://github.com/nicolas42/video-40.git
wget https://pjreddie.com/media/files/darknet19_448.conv.23

mv darknet19_448.conv.23 darknet
cp video-40/config-darknet/* darknet

./darknet detector train obj.data yolo-obj.cfg darknet19_448.conv.23



# Other config and weights files configurations














    git log --oneline




# How to use with ec2

ssh ubuntu@13.236.3.140

clone jctech-ml

    git clone https://github.com/alexeyAB/darknet.git
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


