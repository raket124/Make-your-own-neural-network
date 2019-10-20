apt-get update
apt-get install python3 -y
apt-get install python3-pip -y
pip3 install pip --upgrade

pip3 install numpy
pip3 install matplotlib
pip3 install scipy
pip3 install dill

wget -PN /home/Code/Data https://pjreddie.com/media/files/mnist_train.csv
wget -PN /home/Code/Data https://pjreddie.com/media/files/mnist_test.csv