# SpecEncoder

Source code for __SpecEncoder: Deep Metric Learning for Accurate Peptide Identification in Proteomics__. Link to paper: [https://academic.oup.com/bioinformatics/article/40/Supplement_1/i257/7700866](https://academic.oup.com/bioinformatics/article/40/Supplement_1/i257/7700866)

## __Error in Code and model weights have been FIXED__

Free for academic uses. Licensed under LGPL.

__Visit [https://predfull.com/](https://predfull.com/) to check related works__

## Update History

* 2024.07.14: Error in Code and model weights have been FIXED.
* 2024.03.07: Second version.
* 2023.10.28: First version.

## Method

Based on the structure of the residual convolutional networks.

![model](imgs/model.png)

Different workflows:

![workflow](imgs/workflow.png)

### Required Packages

Recommend to install dependency via [Anaconda](https://www.anaconda.com/distribution/)

* Python >= 3.7
* Tensorflow >= 2.5.0
* Pandas >= 0.20
* pyteomics
* numba
* Tensorﬂow-addons

### Usage

__After clone this project, you should download the pre-trained model (`encoder.h5`) from [zenodo.org](https://zenodo.org/records/12742432) and place it into SpecEncoder's folder.__

You can find model files for both charge 2+ and 3+. Note that charge 3+ model files is larger as we used 2x layers on charge 3+ for better performance.

#### Convert Spectra into vectors

First we convert query into vectors:

`python encode.py --query query.mgf --model encoder.h5 --output query.pkl`

Then we convert target library and decoy library into vectors:

`python encode.py --query library.mgf --model encoder.h5 --output library.pkl`

`python encode.py --query decoy.mgf --model encoder.h5 --output decoy.pkl`

Typical running speed: convert around 700 spectra in 1 second on a NVIDIA A6000 GPU.

#### Generate theoretical spectra

If we have sequences that don't have experimental spectra, we can predict theoretical spectra using [predfull](https://predfull.com/) (you can find `pm.h5` [here](https://drive.google.com/drive/folders/1Ca3HdV-w8TZPRa9KhPBbjrTtGSmtEIsn), also, note that the `predfull.py` here is modified to suit this project):

`python predfull.py --input example_db.tsv --model pm.h5 --output db_predicted.mgf --decoy decoy_predicted.mgf`

Note that this script will also generate DECOY database using reversed peptides.

#### Searching

We can do final search in 3 approaches:

1. Spectral library search
2. Database search
3. Mixed search

In mixed search spectral library will overwrite database spectra when share a same peptide.

Then we can perform searching:

`python search.py --query query.pkl --mode 1 --lib_vec library.pkl --lib_decoy decoy.pkl --output result.tsv`

## Train this model

__Work in progress__

See `train.py` for sample training codes
