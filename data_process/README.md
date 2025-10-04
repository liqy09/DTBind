# Data Preparation

This section describes the data preparation process for the three prediction tasks used in DTBind, including dataset partitioning, feature extraction, and graph construction.

## 1. Binding Occurrence Prediction

Data source: BioSnap database
Reference: MolTrans dataset format

Data files:

../Data/Biosnap/train.csv  
../Data/Biosnap/val.csv  
../Data/Biosnap/test.csv  

Each file contains: Drug SMILES string, Protein sequence and UniProt ID, Binary binding label (0/1)

Supporting files:

../Data/Biosnap/biosnap_uniprotid.txt   # Protein UniProt IDs
../Data/Biosnap/drug_smiles.tsv         # DrugBank IDs and SMILES strings

Protein structures:Proteins are downloaded from AlphaFoldDB using their UniProt IDs.

Example command:

        wget -i ./Data/BioSnap/wget_biosnap.txt -P ./pdb_files/biosnap_pdb

## 2. Binding Site & Binding Affinity Datasets
### (1) PDBBind (v2020)

Download from: https://www.pdbbind-plus.org.cn/ 
Save files in:
./Data/pdbbind_files/
./Data/pdbbind_index/

Each complex includes protein and ligand .pdb files (and defined binding pocket).

### (2) PDB Data

Create folder:

        mkdir -p ./Data/pdb_files/

Download protein and ligand PDBs:

        wget -i ./Data/PDBBind/pdbbind_wget_complex.txt -P ./pdb_files/complex
        wget -i ./Data/PDBBind/pdbbind_wget_ligand.txt -P ./pdb_files/ligand

### (3) Dataset Splits

Each task uses predefined train/val/test splits, located at:

./Data/PDBBind/bindingsite_dataset/
./Data/PDBBind/affinity_dataset/

## 3. Protein Pretrained Embedding Extraction

Protein sequences are extracted as follows:

Task	FASTA File
Binding Occurrence	./Data/BioSnap/biosnap_protein_seq.fasta
Binding Site Prediction	./Data/PDBBind/bindingsite_dataset/pdbbind_protein.fasta
Binding Affinity	./Data/PDBBind/affinity_dataset/pdbbind_pocket_seq.fasta

We use ProtTrans pretrained protein language models (e.g., ProtT5-XL-UniRef50) to generate residue-level embeddings.
Reference implementation: https://github.com/agemagician/ProtTrans
Example:

        python extract_prottrans_embedding.py \
            --input_fasta ./Data/BioSnap/biosnap_protein_seq.fasta \
            --model prot_t5_xl_uniref50 \
            --output ./Data/BioSnap/biosnap_embeddings.h5

The resulting .h5 files store per-residue embeddings for each protein.

## 4. Protein Surface Feature Extraction (dMaSIF)

Protein surface features are extracted following the dMaSIF framework.

For affinity prediction, surface features are extracted from the full protein structure but later mapped to pocket residues.

Move to the directory:

        cd ./data_process/surface_feature_extraction

Then run the following steps:
Surface mesh generation (using MSMS):

        python 1_extract_msms.py

Surface geometry computation:

        python 2_compute.py

Feature packaging:

        # For full-protein features:
        python 3_surface_feature.py

        # For pocket-level features (affinity task):
        python 3_pocket_surface_feature.py

All residue-level surface features are stored in .pkl format.

## 5. Binding Site Label Extraction via PLIP

We use PLIP (https://github.com/ssalentin/plip/) to calculate non-covalent interactions and identify binding residues.

Example command:

        python plipcmd.py -f example_complex.pdb -t --name example_output

Output files:
Stored in:

./Data/plip_result_all_set/

Residue-level binding site labels are saved in:

./Data/PDBBind/bindingsite_dataset/site_labels.txt

## 6. Graph Construction for Model Input

All scripts are located in:

        cd ./data_process/graph_construction

### (1) Drug Graphs (for occurrence or site prediction)

Two construction options are provided:

From SMILES

        python drug_gra.py

From SDF files

        python drug_graph.py

### (2) Protein Graphs

Protein graphs include surface features, pretrained embeddings, and geometric edge features.

Without labels (for occurrence or site prediction)

        python prepare_no_label.py

With site-level labels

        python protein_graph.py

### (3) Complex Graphs (for affinity prediction)

Each sample includes:Drug graph, Protein pocket graph, Pocket–ligand heterogeneous graph, Binding affinity label

        python construct_graph_hetero.py

Now the dataset is ready for model training and evaluation.
