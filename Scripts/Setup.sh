apt-get update
apt-get install ffmpeg -y
apt-get install imagemagick -y
apt-get install python3 -y
apt-get install python3-pip -y
pip3 install pip --upgrade

wget -PN /home/Code/Data https://pjreddie.com/media/files/mnist_train.csv
wget -PN /home/Code/Data https://pjreddie.com/media/files/mnist_test.csv