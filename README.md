# SNN-for-sensory-neuron
This is a 3-layer SNN for MNIST
This code is based on spikingjelly, you need to download spikingjelly
SNN2.py is used to train the network and save the Loss and accuracy files
Before running SNN2.py, you need to copy class LIFNode2 to neuron.py (package that comes with spikingjelly)
load_npy.py is used to download the saved accuracy and Loss files
load_model.py is used to download the trained model and generate hotmap images (classification results)
