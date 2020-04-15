# Single-Modal Video Analysis of Personality Traits using Low-Level Visual Features

Description

This project aims to predict the Big-Five Personality model of humans in short video sequences.

Requirements

    python 2.7.x
    pip install keras tensorflow numpy matplotlib dlib h5py opencv-python

Structure

    ./3D_CNN: includes scripts to train and test the proposed 3D_CNN network
    ./CNN_V8: includes scripts to train and test the proposed image-based (VGG) CNN network
    ./FaceDBGenerator_V2: includes scripts to extract the detected faces in a given video
    
Datasets
    For this investigation the public dataset "First Impression" published by ChaLearn Looking at People is used.
    
    Download link: http://chalearnlap.cvc.uab.es/dataset/20/description/
    
    Sources:
    J.-I. Biel, O. Aran, and D. Gatica-Perez, You Are Known by How You Vlog: Personality Impressions and Nonverbal Behavior in YouTube in Proc. AAAI Int. Conf. on Weblogs and Social Media (ICWSM), Barcelona, Jul. 2011
    J.-I. Biel and D. Gatica-Perez, The YouTube Lens: Crowdsourced Personality Impressions and Audiovisual Analysis of Vlogs, IEEE Trans. on Multimedia, Vol. 15, No. 1, pp. 41-55, Jan. 2013

Usage

    Train and Test Models:
    Different options (e.g. train, test, visulization, ...) can be selected in the main.py source file. In order to start for example the training process do the following:
    Attention: you have to specify the correct paths.
 
    python main.py OR sh run.sh

    FaceDBGenerator:
    To run the face generator you need the valid dataset and to specify the paths as well as the image size (default 64x64)
    
    main.py -i <inputfile> -o <outputfile>
