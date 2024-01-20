# DRANetSplicer

## 1. Environment setup
All experiments were implemented using TensorFlow and Keras on the Linux system with a single NVIDIA GA102 [GeForce RTX 3090 Ti] 24GB GPU, and if you would like to reproduce our models, we recommend that you use [Anaconda](https://docs.anaconda.com/anaconda/install/linux/) to build a python virtual environment.

#### 1.1 Create and activate a new virtual environment

```
conda create -n DRANetSplicer python=3.8
conda activate DRANetSplicer
```

#### 1.2 Clone the project and install requirements

```
git clone https://github.com/XueyanLiu-creator/DRANetSplicer
cd DRANetSplicer
python3 -m pip install --editable .
```

#### 1.3 View the help file for running the program.

```
python run_code.py -h
```

## 2. Re-train (Skip this section if you just want to use trained models)

#### 2.1 Data processing

Please see the sample data at `/data/dna_sequences/`. If you are trying to re-train DRANetSplicer with your own data, please process you data into the same format as it. Our program automatically performs one-hot encoding of the data under `/data/dna_sequences/` and saves it in `data/encode_datasets/`.

#### 2.2 Model Training

```
python run_code.py \
    --organism_name oryza \
    --splice_site_type donor \
    --train \
    --num_train_epochs 20 \
    --batch_size 64 \
    --learning_rate 0.01 \
    --verbose 1 \
    --report
```

## 3. Prediction

You can run the following code to get predictions using the model we provide or a model you have re-trained. If you want to use your re-trained models to predict the data, remove the `--use_our_trained_models` option.

```
python run_code.py \
    --organism_name oryza \
    --splice_site_type donor \
    --use_our_trained_models DRANetSplicer1 \
    --test \
    --batch_size 64 \
    --verbose 1 \
    --report
```

With the above command, the trained or re-trained DRANetSplicer model will be loaded from `models/trained_models/` , and makes prediction on the `*_test.npz` file that saved in `data/encode_datasets/` and save the prediction result at `resluts/`.

## 4. Visualization

Visualiazation of DRANetSplicer consists of 2 steps. Calculate contribution scores and Plot. Calculate with only one model (For example, DRANetSplicer1). If you want to use your re-trained models for visualization, remove the `--use_our_trained_models` option. The method of visualization can be either `deep_shap` or `grad_cam`.

```
python run_code.py \
    --organism_name oryza \
    --splice_site_type donor \
    --use_our_trained_models DRANetSplicer1 \
    --visualization deep_shap
```

With the above command, the trained or re-trained DRANetSplicer model will be loaded from `models/trained_models/` , and calculates contribution scores on the `*_test.npz` file that saved in `data/encode_datasets/` and save the result at `visualization/figure/`.
