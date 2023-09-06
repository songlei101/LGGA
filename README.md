# LGGA
-LGGA-
## Introduction
This is the source code and dataset for the following paper: 
**LGGA: Local Geometry-guided Graph Attention for molecular property prediction**

## **Environment**
  - deepchem                  2.6.1
  - dgl-cuda11.3              0.9.0
  - python                    3.9.13   
  - pytorch                   1.12.1
  - rdkit                     2022.3.5
  - sklearn                   0.0

## Args:
  - dataset : Dataset
  - pretrain : Whether to load the pre-trained model (default: True)
  - load_saved : Whether to load the saved model (default: False)
  - save_model : Whether to save the model (default: False)
  - num_epochs : Maximum number of epochs for training(default: 300)
  - split : Splitting method(random or scaffold)

### Run code
- For pre-training:  
    ```
    python pretrain.py
    ```
- For predicting:  
    ```
    python main.py
    ```
    
