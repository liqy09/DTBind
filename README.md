# DTBind
DTBind is a mechanism-driven drug–target interaction model that couples graph neural networks with specialized deep-learning modules to predict binding events, residue-level binding sites, and affinities from molecular graphs.
This project provides a unified pipeline for accurately predicting drug–target binding, residue-level binding sites, and binding affinities, or for retraining the model on new datasets.
![](https://github.com/liqy09/DTBind/tree/main/IMG/DTBind_framework.png "Overview of DTBind")  

## 1 Description 

  Accurate prediction of drug–target molecular recognition is pivotal to early-stage drug discovery, encompassing binding occurrence, binding site localization, and affinity estimation. However, current methods typically model only individual subtasks of molecular recognition, yield fragmented insights, and neglect key mechanistic determinants during model design on which molecular recognition depends. We present DTBind, a unified and mechanism-driven framework that encodes proteins and drugs according to recognition determinants, achieving sequence-driven binding occurrence prediction, structure-guided binding site localization, and complex-level affinity estimation. Across diverse benchmarks, DTBind consistently outperforms state-of-the-art methods in both predictive accuracy and generalization ability. 
  
## 2 Installation  

### 2.1 System requirements
For prediction process, you can predict functional binding residues from a protein structure within a few minutes with CPUs only. However, for training a new deep model from scratch, we recommend using a GPU for significantly faster training.
To use GraphRBF with GPUs, you will need: cuda >= 11.6, cuDNN.
### 2.2 Create an environment

We highly recommend to use a virtual environment for the installation of DTBind and its dependencies.

A virtual environment can be created and (de)activated as follows by using conda(https://conda.io/docs/):

        # create
        $ conda create -n DTBind_env python=3.8
        # activate
        $ conda activate GraphRBF
        # deactivate
        $ conda deactivate
        
### 2.3 Install DTBind dependencies
Note: Make sure environment is activated before running each command.

#### 2.3.1 Install requirements
Install pytorch 2.0.1 (For more details, please refer to https://pytorch.org/)

        For linux:
        # CUDA 11.6
        $ pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
        # CPU only
        $ pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu
Install torch_geometric 2.6.1 (For more details, please refer to https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)

        $ pip install torch-cluster -f https://data.pyg.org/whl/torch-1.13.0+cu116.whl
        $ pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
        $ pip install torch-sparse -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
        $ pip install torch-cluster -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
        $ pip install torch_geometric==2.6.1
Install other requirements

        $ pip install torchnet==0.0.4
        $ pip install tqdm
        $ pip install prettytable
        $ pip install pandas
        $ pip install scikit-learn

Note: Typical install requirements time on a "normal" desktop computer is 10 minutes.
        
## 3 Usage   

### 3.1 Predict drug-target binding occurence from a protein structure(predicted structure or experimental structure) based on trained deep models
We have packaged data extraction: XXX.py, model training: XXX.py, DTBind model: XXX.py, the validation module: metrices.py, and the prediction code: test.py.  
First, install the environment as described above, and after that, use the code from the prediction command 'prediction code.log' file in the folder:  


    cd ../DTBind-main  
    python test.py --querypath ../DTBind-main/example 
  
    Command list： 
    --querypath   The path of query structure  
    --filename    The file name of the query structure（we need user to upload its pdb(1ddl_A.pdb) and pssm and hmm file of each chain(1ddl_A.pssm and 1ddl_A.hmm)）  

### 3.2 Predict drug-target binding sites from a protein structure(experimental structure) based on trained deep models
We have packaged data extraction: XXX.py, model training: XXX.py, DTBind model: XXX.py, the validation module: metrices.py, and the prediction code: test.py.  
First, install the environment as described above, and after that, use the code from the prediction command 'prediction code.log' file in the folder:  


    cd ../DTBind-main  
    python test.py --querypath ../DTBind-main/example 
  
    Command list： 
    --querypath   The path of query structure  
    --filename    The file name of the query structure（we need user to upload its pdb(1ddl_A.pdb) and pssm and hmm file of each chain(1ddl_A.pssm and 1ddl_A.hmm)）  

### 3.1 Predict drug-target binding affinity from a complex structure based on trained deep models
We have packaged data extraction: XXX.py, model training: XXX.py, DTBind model: XXX.py, the validation module: metrices.py, and the prediction code: test.py.  
First, install the environment as described above, and after that, use the code from the prediction command 'prediction code.log' file in the folder:  


    cd ../DTBind-main  
    python test.py --querypath ../DTBind-main/example 
  
    Command list： 
    --querypath   The path of query structure  
    --filename    The file name of the query structure（we need user to upload its pdb(1ddl_A.pdb) and pssm and hmm file of each chain(1ddl_A.pssm and 1ddl_A.hmm)）  
	
### 3.2  Train a new deep model from scratch

#### 3.2.1 Download the datasets used in DTBind.

Donload the PDB files and the feature files (the pretrain feature h5 profiles, surface feature profiles) from http: and store the PDB files in the path of the corresponding data.

Example:

	The PDB files of XX

#### 3.2.2 Generate the training, validation and test data sets from original data sets

    Example:
        $ cd ../DTBind-main/scripts
        # demo 1
        $ python dti_train.py
        # demo 2
        $ python train.py 

    Output:
    The data sets are saved in ../Datasets/.

    Note: {featurecode} is the combination of the first letter of {features}.
    Expected run time for the demo 1 and demo 2 on a "normal" desktop computer are 30 and 40 minutes, respectively.

   
#### 3.2.3 Train the deep model

    Example:
        $ cd ../DTBind-main/scripts
        # demo 1
        $ python training.py
        # demo 2
        $ python training_guassian.py

    Output:
    The trained model is saved in ../Datasets/.
    The log file of training details is saved in ../Datasets/.log.

    Note: {starttime} is the time when training.py started be executed.
    Expected run time for demo 1 and demo 2 on a "normal" desktop computer with a GPU are 30 and 12 hours, respectively.


### 4 Frequently Asked Questions
(1) If the script is interrupted by "Segmentation fault (core dumped)" when torch of CUDA version is used, it may be raised because the version of gcc (our version of gcc is 5.5.0) and you can try to set CUDA_VISIBLE_DEVICES to CPU before execute the script to avoid it by:
        $ export CUDA_VISIBLE_DEVICES="-1"
(2) If your CUDA version is not 11.6, please refer to the homepages of Pytorch(https://pytorch.org/) and torch_geometric (https://pytorch-geometric.readthedocs.io/en/latest/) to make sure that the installed dependencies match the CUDA version. Otherwise, the environment could be problematic due to the inconsistency.



