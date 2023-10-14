echo "****************** Installing pytorch ******************"
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116

echo ""
echo ""
echo "****************** Installing yaml ******************"
pip install PyYAML

echo ""
echo ""
echo "****************** Installing easydict ******************"
pip install easydict

echo "****************** Installing opencv-python ******************"
pip install opencv-python==4.5.5.64

echo ""
echo ""
echo "****************** Installing pandas ******************"
pip install pandas

echo "****************** Installing jpeg4py python wrapper ******************"
apt-get install libturbojpeg
pip install jpeg4py


echo "****************** Installing thop ******************"
pip install thop

echo ""
echo ""
echo "****************** Installing lmdb ******************"
pip install lmdb

echo ""
echo ""
echo "****************** Installing scipy ******************"
pip install scipy

echo ""
echo ""
echo "****************** Installing visdom ******************"
pip install visdom


echo "****************** Installing vot-toolkit python ******************"
pip install git+https://github.com/votchallenge/vot-toolkit-python

echo ""
echo ""
echo "****************** Installation complete! ******************"
