# Data Preparation

The complete DTBind dataset can be downloaded from:(https://zenodo.org/records/10826801).

This section describes the data preparation process for the three prediction tasks used in DTBind, including dataset partitioning, feature extraction, and graph construction.

## 1. Binding Occurrence Prediction

Data source: BioSnap database
Reference: MolTrans dataset format

Data files:

        ./Data/dti/train_label.tsv 
        ./Data/dti/val_label.tsv  
        ./Data/dti/test_label.tsv  

Each file contains: DrugBank ID, Protein ID, Binary binding label (0/1)
Supporting files:

        ./Data/dti/biosnap_uniprotid.txt   # Protein UniProt IDs
        ./Data/dti/drug_smiles.tsv         # DrugBank IDs and SMILES strings

Protein structures:Proteins are downloaded from AlphaFoldDB using their UniProt IDs.

Create folder:

        $ mkdir -p ./Data/pdb_files/

Example command:

        $ wget -i ./Data/dti/wget_biosnap.txt -P ./pdb_files/biosnap_pdb

## 2. Binding Site & Binding Affinity Datasets
### (1) PDBBind (v2020)

Download from: https://www.pdbbind-plus.org.cn/ 
Save files in:

        ./Data/pdbbind_files/
        ./Data/pdbbind_index/

Each complex includes protein .pdb files, pocket .pdb files and ligand .sdf files.

### (2) PDB Data

Download protein and ligand PDBs:

        $ wget -i ./Data/affinity/pdbbind_wget_complex.txt -P ./pdb_files/complex
        $ wget -i ./Data/affinity/pdbbind_wget_ligand.txt -P ./pdb_files/ligand

### (3) Dataset Splits

Each task uses predefined train/val/test splits, located at:

        ./Data/affinity/
        ./Data/site/

## 3. Protein Pretrained Embedding Extraction

Protein sequences are extracted as follows:

        # Binding Occurrence Prediction
        ./Data/dti/biosnap_protein_seq.fasta
        # Binding Site Prediction
        ./Data/site/pdbbind_protein.fasta
        # Binding Affinity Prediction
        ./Data/affinity/pdbbind_pocket_seq.fasta

We use ProtTrans pretrained protein language models (e.g., ProtT5-XL-UniRef50) to generate residue-level embeddings.
Reference implementation: https://github.com/agemagician/ProtTrans
Example:

        python extract_prottrans_embedding.py \
            --input_fasta ./Data/BioSnap/biosnap_protein_seq.fasta \
            --model prot_t5_xl_uniref50 \
            --output ./Data/BioSnap/biosnap_embeddings.h5

The resulting .h5 files store per-residue embeddings for each protein.

## 4. Protein Surface Feature Extraction (based on dMaSIF)

Protein surface features are extracted based on dMaSIF.

For affinity prediction, surface features are extracted from the full protein structure but later mapped to pocket residues.

Move to the directory:

        $ cd ./data_process/surface_feature_extraction

Then run the following steps:
Surface mesh generation (using MSMS):

        $ python 1_extract_msms.py

Surface geometry computation:

        $ python 2_compute.py

Feature packaging:

        # For full-protein features:
        $ python 3_all_protein_surface_feature.py

        # For pocket-level features (affinity task):
        $ python 3_all_pocket_surface_feature.py

All residue-level surface features are stored in .pkl format.

Note: Pocket-level surface features are calculated based on the full protein structure but only the features of the pocket residues are saved.

## 5. Binding Site Label Extraction via PLIP

We use PLIP (https://github.com/ssalentin/plip/) to calculate non-covalent interactions and identify binding residues.

Example command:

        $ python plipcmd.py -f example_complex.pdb -t --name example_output

Output files:
Stored in:

        ./Data/plip_result_all_set/

Residue-level binding site labels are saved in:

        ./Data/site/site_labels.txt

## 6. Graph Construction for Model Input

All scripts are located in:

        $ cd ./data_process/graph_construction

For binding occurrence prediction and binding site prediction, separate molecular graph files for proteins and drugs are required (e.g., protein.pt and drug.pt, such as 1a0q.pt and DM5.pt).  The input files for each sample are saved separately.

For binding affinity prediction, all necessary inputs (protein graphs, drug graphs, heterographs, and labels) for the training, validation, and test sets should be saved as three separate . pkl files.  Each . pkl file contains all samples for the respective set (i.e., one . pkl file for the training set, one for the validation set, and one for the test set).

### (1) Drug Graphs (for occurrence or site prediction)
Two construction options are provided. Input requirement: provide either drug SDF files or SMILES strings.

From SMILES(occurrence prediction)

        $ python drug_gra.py

From SDF files

        $ python drug_graph.py

### (2) Protein Graphs (for occurrence or site prediction)

Protein graphs include surface features, pretrained embeddings, and geometric edge features.

Without labels (for occurrence or site prediction)
Required inputs:
Protein PDB files
Surface feature .pkl files (from 4. Protein Surface Feature Extraction)
Pretrained embedding .h5 files (from ProtTrans)

        $ python protein_graph_no_label.py

With site-level labels (for training site prediction model)
Required inputs:

Protein PDB files
Residue-level binding site labels .txt
Surface feature .pkl files
Pretrained embedding .h5 files

        $ python protein_graph.py

### (3)Complex Graphs (for affinity prediction)

Each sample includes:Drug graph, Protein pocket graph, Pocket–ligand heterogeneous graph, Binding affinity label

To construct these graphs, the following inputs are required:
Protein pocket PDB files
Ligand SDF files(Consistent with the coordinate system in the protein pdb file)
Pretrained protein embeddings (.h5 files from ProtTrans)
Protein surface features (.pkl files)
Run the construction script:

        $ python construct_graph_hetero.py

After this, the dataset is ready for model training and evaluation.
