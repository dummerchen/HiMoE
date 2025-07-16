# Hierarchical Mixture-of-Experts Adapters for DeepFake Detection 
This repository contains the reproduction of the experiments presented in the paper "Hierarchical Mixture-of-Experts Adapters for DeepFake Detection".

## Usage 
### Prepare the Datasets
Please follow the DeepfakeBench guide and regenerate the JSON file containing the path to the dataset.
### Training the Model:
```python
python3 train.py -cf config_file_path -bs 32 -r -1 -ts 32000
```
### Evaluate the Model:
``` python
python3 eval.py -cf config_file_path -td test_dataset_name -r checkpoint_path
```

