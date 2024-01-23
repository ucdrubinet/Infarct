# Infarct Screening (under construction)
This is the implementation codes for the journal manuscript: 


## Authors / Contributors
* Luca Cerny Oliveira
* Joohi Chauhan
* Zhengfeng Lai

If you have any questions/suggestions or find any bugs,
please submit a GitHub issue.

## Prerequisites
The list of prerequisites for building and running this repository is described
below. Verify the environment.yml file to replicate the Conda environment used in this work.

More requisites on machine specifications to come.

## Setup Instructions
`conda env create -f environment.yml`  
You should be able to replicate our environment. If there are issues with specific packages and/or their interactions, refer to conda documentation and try installing those packages individually.

## Instructions for replicating work
* [preprocessing.py/] - First script to be executed. It will perform tiling and preprocessing raw WSI files. Requirements:
  * Raw WSI files (to be shared once the paper is released)
  * [/utils/] scripts for WSI handling
  * Annotation XML files for .svs WSIs (to be shared once the paper is released)
  * [../gt.csv/] file with WSI-level ground-truth
  * BrainSec trained model for background segmentation (available [here](https://drive.google.com/file/d/1LfK_xfBdqoptYRFPx6qVMXPhfEO3HpBI/view?usp=sharing))
* [training.py/] - Second script to be executed. It will train a patch-based model and save learning curves. Requirements:
  * Labeled patches: an output of the previous script
* [heatmap.py/] - Third script to be executed. It will generate the heatmaps for the trained patch-based model. Requirements:
  * Path to trained patch-based model: an output of the previous script
* [results.py/] - Fourth and last script to be executed. It will generate post-processed detection and screening results. It also compares with ground truth to generate WSI-level screening and individual infarct detection results. Requirements:
  * WSI directory: an output of the [preprocessing.py/] script
  * Binary Mask directory: an output of the [preprocessing.py/] script
  * [../gt_plus_wmr_pvs.csv/] file with WSI-level infarct ground-truth and white matter rarefaction scoring.


#### Note
In this repo, all Python scripts that are meant to be run as top-level scripts
support [`argparse`](https://docs.python.org/3/library/argparse.html).  
That is, input arguments can be specified to change the default behavior.

## Credit
* [plaquebox-paper](https://github.com/keiserlab/plaquebox-paper)
* [BrainSec](https://github.com/ucdrubinet/BrainSec/tree/master)
* [pyvips](https://libvips.github.io/pyvips/): Image processing for large
TIFF-based image.
