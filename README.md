# Age Disentangled Variational Autoencoder for Age Progression of Normal Subjects

## Installation
After installing Miniconda and cloning the repo open a terminal and go to the project directory. 

Change the permissions of install_env.sh by running `chmod +x ./install_env.sh` 
and run it with:
```shell script
./install_env.sh
```
This will create a virtual environment with all the necessary libraries.

Note that it was tested with Python 3.8, CUDA 11.3, and Pytorch 1.12.1. The code 
should work also with newer versions of  Python, CUDA, and Pytorch. If you wish 
to try running the code with more recent versions of these libraries, change the 
CUDA, TORCH, and PYTHON_V variables in install_env.sh

Then activate the virtual environment :
```shell script
source ./id-generator-env/bin/activate
```

PyTorch3D needs to have version 0.4.0 or higher. If this is not the case after installing the environment using the install_env.sh file and you are not able to easily update it, run the following commands which can be found on the Pytorch3D instalation guide: https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md
```shell script
curl -LO https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz
tar xzf 1.10.0.tar.gz
export CUB_HOME=$PWD/cub-1.10.0

pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```
MAKE SURE Pillow==0.8.2

## Datasets

Current dataset:
  Current dataset being used for this code is The Liverpool-York Head Model dataset. 

Previous dataset used: 
  To obtain access to the UHM models and generate the dataset, please follow the 
  instructions on the 
  [github repo of UHM](https://github.com/steliosploumpis/Universal_Head_3DMM).

 Data will be automatically generated from the UHM during the first training. 
 In this case the training must be launched with the argument `--generate_data` 
 (see below).
 
 ## Prepare Your Configuration File
 
 !! code from Simone's work !!
 We made available a configuration file for each experiment (default.yaml is 
 the configuration file of the proposed method). Make sure 
 the paths in the config file are correct. In particular, you might have to 
 change `pca_path` according to the location where UHM was downloaded.
 
 ## Train and Test
 
 To start the training from the project repo simply run:
 ```shell script
python train.py --config=configurations/<A_CONFIG_FILE>.yaml --id=<NAME_OF_YOUR_EXPERIMENT>
```

If this is your first training and you wish to generate the data, run:
```shell script
python train.py --generate_data --config=configurations/<A_CONFIG_FILE>.yaml --id=<NAME_OF_YOUR_EXPERIMENT>
``` 

Basic tests will automatically run at the end of the training. If you wish to 
run additional tests presented in the paper you can uncomment any function call 
at the end of `test.py`. If your model has alredy been trained or you are using 
our pretrained model, you can run tests without training:
```shell script
python test.py --id=<NAME_OF_YOUR_EXPERIMENT>
```
Note that NAME_OF_YOUR_EXPERIMENT is also the name of the folder containing the
pretrained model.


## Additional Notes

We make available the files storing:
 - the precomputed down- and up-sampling transformation
 - the precomputed spirals
 - the mesh template with the face regions
 - the network weights

