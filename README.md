# Minor/adult classification from face images using an ensemble of shallow multilayer perceptrons

This repository contains source code, experiments and examples for the [Neural
Networks](https://www.fer.unizg.hr/en/course/neunet_a) course (Faculty of Electrical Engineering and Computing, University of
Zagreb) project.

Team members:
1. Filip Karlo Došilović
2. Antonio Gauta
3. Lara Lokin
4. Domagoj Pavlović
5. Ana Puljčan

## Installation

**Conda** installation:

```bash
conda create --name <env> --file requirements.txt
```

**Pip** installation:

```bash
pip install -r requirements.txt
```


## Running the experiments

For all experiments we recommend that you are positioned in the root of the
project directory.

### Hyperparameter search

To run **hyperparameter search** use:

```bash
python src/<param-search-script>.py <param-search-yaml>.yaml
```

for example

```bash
python src/adasyn_bagging_mlp_param_search.py adasyn_bagging_mlp_param_search.yaml
```

### Train and test

To run **train and test** scripts use:

```bash
python src/<train-test-script>.py <train-test-yaml>.yaml
```

for example

```bash
python src/balanced_bagging_mlp_train_test.py balanced_bagging_mlp_train_test_32x32_a.yaml
```

### Main script for inference on video files

After training and testing your chosen model, you can run it on a file of your
choice.

Parameters for `main.py` script are:

- **--model** - a trained ensemble model for classification (should return 1 for adults and 0 for minors), default value is provided
- **--pca-model** - a PCA trained on the training set for dimensionality reduction, default value is provided
- **--image-size** - an image size for the PCA, default value (32) is provided
- **--video-file** - a video file
- **--blur** - whether to blur the faces of minors or not, default value ("yes") is provided (other option is "no")

Examples:

```bash
python src/main.py --video-file=<absolute-path-to-video-file>
python src/main.py --video-file=<absolute-path-to-video-file> --pca-model=pca_48x48.npy --model=adasyn_bagging_mlp_48x48_b.jl --image-size=48
```
