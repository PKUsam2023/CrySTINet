# CSTRNet
This repo contains demonstrations of an extensible Crystal Structure Type Recognition Network (CrySTINet), which consists of a variable number of submodels (RCNet: Resnet-Confidence Network).

This repository is adapted from the codebase used to produce the results in the paper "An Extensible Deep Learning Framework for Identifying Crystal Structure Types based on the X-Ray Powder Diffraction Patterns."

## Requirements

The code in this repo has been tested with the following software versions:
- Python>=3.7.0
- torch>=1.13.1
- tensorboard>=2.6.0
- pandas>=1.3.1
- numpy>=1.21.5
- matplotlib>=3.1.3
- tqdm>=4.65.0
- scipy>=1.4.1

The installation can be done quickly with the following statement.
```
pip install -r requirements.txt
```

We recommend using the Anaconda Python distribution, which is available for Windows, MacOS, and Linux. Installation for all required packages (listed above) has been tested using the standard instructions from the providers of each package.

## Data

The experiment data is at
```
./data/experiment/smooth_xrd/
```

Examples of the simulated XRD data are at
```
./data/simulated_examples/
```

To obtain full simulated XRD data, please contact 2101212695@stu.pku.edu.cn 


## Files

This repo should contain the following files:
- 1 ./CrySTINet/CrySTINet_batch.py - The code for batch testing with CrySTINet.
- 2 ./CrySTINet/CrySTINet_single.py - The code for structure types classification of single XRD with CrySTINet.
- 3 ./RCNet/getdata.py - The code for each RCNet to acquire training data.
- 4 ./RCNet/network.py - The code for each RCNet architecture.
- 5 ./RCNet/train.py - The code to train the each RCNet.
- 6 ./RCNet/utils.py - The code for logger.
- 7 ./RCNet/checkpoints/ - The model file of our trained 10 RCNets.

## Model
The trained model files of CrySTINet are at https://github.com/PKUsam2023/CrySTINet


If you find any bugs or have questions, please contact 2101212695@stu.pku.edu.cn 