# TSAPE
## The code for the paper: Temporal-Series-Aware Adaptive Positional Encoding for Transformer-based Sequential Recommendation
We thank the **RecBole framework** ([RUCAIBox/RecBole: A unified, comprehensive and efficient recommendation library](https://github.com/RUCAIBox/RecBole )) for providing the foundational architecture for our experiments. To reproduce our work, please focus on the following key folders:

#### Folder Structure Overview

##### `config/`  

This folder contains parameter configurations for different model experiments and datasets. Each configuration file defines hyperparameters specific to a certain model-dataset combination.

##### `model/`  

This folder contains the model implementations used in the experiments. (TME* -> TSAPE+backbones)

##### `dataset/`  

This folder includes the processed datasets used in our experiments or links to publicly available large-scale datasets. (For the two datasets with relatively large file sizes, which are inconvenient to upload to GitHub, we have also created an anonymous cloud storage sharing link: https://www.dropbox.com/scl/fo/ef52k9tgihrgi2y8xioz5/AJHS5LLCrsOcBTXZdPL5N8E?rlkey=1txqpjhmr98h9q05qv3nwliyt&st=xwfyxjsz&dl=0)

##### Root Directory (`./`)  

The root directory contains the main entry files for running experiments.
