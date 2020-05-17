# Environment
    source ~/.local/bin/virtualenvwrapper.sh
    source ~/.bashrc
    mkvirtualenv tfr -p python3
    workon tfr
    pip install opencv-contrib-python
    pip install imutils
    pip install scikit-learn
    pip install spyder
# Run
python extract_train_evaluvate.py
python recognize_faces.py --image ./recognize-images/kate_leonardo.jpg