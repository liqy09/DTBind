# DTBind
DTBind is a mechanism-driven drug–target interaction model that couples graph neural networks with specialized deep-learning modules to predict binding events, residue-level binding sites, and affinities from molecular graphs.
This project provides a unified pipeline for accurately predicting drug–target binding, residue-level binding sites, and binding affinities, or for retraining the model on new datasets.

## 1 Description 

  Accurate prediction of drug–target molecular recognition is pivotal to early-stage drug discovery, encompassing binding occurrence, binding site localization, and affinity estimation. However, current methods typically model only individual subtasks of molecular recognition, yield fragmented insights, and neglect key mechanistic determinants during model design on which molecular recognition depends. We present DTBind, a unified and mechanism-driven framework that encodes proteins and drugs according to recognition determinants, achieving sequence-driven binding occurrence prediction, structure-guided binding site localization, and complex-level affinity estimation. Across diverse benchmarks, DTBind consistently outperforms state-of-the-art methods in both predictive accuracy and generalization ability. 
  
## 2 Installation  
### 2.1 System requirements
For prediction process, you can predict functional binding residues from a protein structure within a few minutes with CPUs only. However, for training a new deep model from scratch, we recommend using a GPU for significantly faster training.
DTBind can run on both CPU and GPU.
For model training, a GPU with CUDA ≥ 11.6 and cuDNN is recommended for efficiency.
For inference or feature extraction, CPUs are sufficient.

### 2.2 Software Dependencies
Core Environment
DTBind requires the following core dependencies for model training and inference:
* [Conda*](https://docs.conda.io/en/latest/miniconda.html) Conda is recommended for environment management.
* [Python*](https://www.python.org/) (v3.9.13). Base language.
* [Pytorch*](https://pytorch.org/) (v1.13.0+cu116). Pytorch with GPU version, deep learning backend.
* [Pytorch-geometric*](https://pytorch-geometric.readthedocs.io/en/latest/index.html) (v2.6.1). For geometric neural networks.
* [scikit-learn*](https://scikit-learn.org/) (v1.6.1). For point cloud space searching and model evaluation.

#### Create an environment and Install DTBind dependencies

We highly recommend to use a virtual environment for the installation of DTBind and its dependencies.
A virtual environment can be created and deactivated as follows by using conda(https://conda.io/docs/):

        # create
        $ conda create -n DTBind_env python=3.9
        # activate
        $ conda activate DTBind_env

Install pytorch 1.13.0 (For more details, please refer to https://pytorch.org/)

        For linux:
        # CUDA 11.6
        $ pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
        # CPU only
        $ pip install torch==1.13.1+cpu torchvision==0.14.0+cpu torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cpu
		
Install PyTorch Geometric (for CUDA 11.6):

        $ pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
        $ pip install torch-geometric==2.6.1

Note: the first-time compilation and installation may take 10-30 minutes, depending on the device.

Other core dependencies:

        $ pip install numpy==1.23.4 torchnet==0.0.4 tqdm prettytable biopython==1.83 pandas scikit-learn rdkit==2024.3.2 h5py

Note: please make sure to keep the NumPy version < 2.0, otherwise it may be incompatible with other package versions and cause warnings or errors.
Note: Typical install requirements time on a "normal" desktop computer is 10 minutes. 
        
### 2.3 Optional Dependencies (for Data Preprocessing Only)

The following tools and libraries are only required if you plan to rebuild datasets (e.g., extract protein surface features or PLIP interaction labels).
If you only need training and testing, these can be skipped.

* [PLIP*](https://github.com/pharmai/plip). Compute noncovalent interactions.
* [reduce*](https://github.com/rlabduke/reduce). Add protons to proteins.
* [MSMS*](https://ccsb.scripps.edu/msms/downloads/). Protein surface mesh generation.
* [Pymesh*](https://github.com/PyMesh/PyMesh). Mesh processing for surface features.
* [BioPython*](https://github.com/biopython/biopython). To parse PDB files.
* [pykeops*](https://www.kernel-operations.io/keops/index.html). For computation of all point interactions of a protein surface.

Note:
If you only plan to train or evaluate DTBind, you only need the core dependencies.
If you intend to rebuild raw datasets or extract geometric/surface features, install the optional preprocessing tools (Reduce, MSMS, PLIP, PyMesh, PyKeOps).
  
## 3 Usage   
### 3.1  Predict with Pretrained DTBind Models

We provide a pretrained DTBind model as a predictor. And we have prepared several preprocessed test examples in the ./sample_test folder for your reference and for testing DTBind.
If you want to test your own samples, you need to preprocess the data accordingly.For details, please refer to DTBind-main/data_process/README.md.

Simply run:

        $ python DTBind_predict.py <task>

Replace <task> with one of the following:

        occurrence - Predict drug-target binding occurrence from predictive protein structures
        site - Predict residue-level binding sites from experimental protein structures
        affinity - Predict binding affinity from protein-ligand complex structures

Example:

        $ python DTBind_predict.py occurrence


### 3.2  Train a New Model from Scratch
To train DTBind on your own dataset:

        $ python DTBind_train.py <task>

Replace <task> with occurrence, site, or affinity to train the corresponding model.

Example:

        $ python DTBind_train.py occurrence

Note: 

For training, please refer to the Data Preparation for details. We provide preprocessed training data for your convenience. If you want to train on your own dataset, you need to preprocess the data accordingly. More details can be found in the Data Preparation section.

### 3.3 Model Details
Pretrained models are available in DTBind-main/models/:

        occurrence_model.pth - Binding occurrence prediction
        site_model.pth - Binding site prediction
        affinity_model.pth - Binding affinity prediction

### 3.4 Data Preparation

For detailed data processing procedures, please refer to DTBind-main/data_process/.

The complete DTBind dataset can be downloaded from:(https://zenodo.org/records/10826801)

### 4 Frequently Asked Questions
(1) If the script is interrupted by "Segmentation fault (core dumped)" when torch of CUDA version is used, it may be raised because the version of gcc (our version of gcc is 5.5.0) and you can try to set CUDA_VISIBLE_DEVICES to CPU before execute the script to avoid it by:
        $ export CUDA_VISIBLE_DEVICES="-1"
(2) If your CUDA version is not 11.6, please refer to the homepages of Pytorch(https://pytorch.org/) and torch_geometric (https://pytorch-geometric.readthedocs.io/en/latest/) to make sure that the installed dependencies match the CUDA version. Otherwise, the environment could be problematic due to the inconsistency.



