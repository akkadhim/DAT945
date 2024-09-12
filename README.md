# IMDb Dataset Preparation, Knowledge Collection, and Data Augmentation

This repository provides code to prepare the IMDb dataset for training, collect knowledge using a TM autoencoder, and apply data augmentation using EDA. You have the option to either:

1. **Use the provided pre-generated knowledge files** along with the `Report.ipynb` notebook to save time, or
2. **Build everything from scratch** by following the steps outlined below.

---

## Files in the Repository

- **`prepare.py`**: Prepares the IMDb dataset for training.
- **`collect.py`**: Trains a TM autoencoder and collects knowledge for all tokens in the IMDb dataset.
- **`eda.py`**: Performs data augmentation using EDA (Easy Data Augmentation) and knowledge-based synonym replacement for the IMDb dataset.
- **`Report.ipynb`**: A Jupyter notebook that demonstrates the entire process, including dataset preparation, knowledge collection, and data augmentation. You can use this with pre-existing IMDb knowledge files to skip the lengthy knowledge generation process.
- **Pre-generated IMDb knowledge files**: Files included in the repository that you can use to skip the time-consuming knowledge generation process.

---

## Quick Start Options

### Option 1: Use Pre-generated IMDb Knowledge Files

If you want to avoid the time-consuming process of generating knowledge files, you can directly use the `Report.ipynb` notebook with the pre-existing IMDb knowledge files. These files are already included in the repository for a specific setup.

- Simply open `Report.ipynb` and follow the steps.
  
### Option 2: Build Everything from Scratch

If you prefer to generate everything from scratch, follow the steps below to prepare the IMDb dataset, perform data augmentation, and train the TM autoencoder.

---

## Steps to Use the Repository
### Requirements

Make sure you have the following libraries installed before running the scripts:

```bash
pip install git+https://github.com/cair/tmu.git
```

---

### Step 1: Prepare the IMDb Dataset
First, you need to run `prepare.py` to process and prepare the IMDb dataset for training. This script performs the following tasks:

- It processes the IMDb dataset.
- It generates a vectorizer and saves it as a pickle file (`vectorizer_X.pickle`).
- It saves the training and testing datasets as `.npy` files:
  - `X_train.npy`: The vectorized training data.
  - `y_train.npy`: The training labels.
  - `X_test.npy`: The vectorized test data.
  - `y_test.npy`: The test labels.

#### To Run:

```bash
python prepare.py
```

#### Outputs:

- `vectorizer_X.pickle`: A pickle file containing the vectorizer for the IMDb dataset.
- `X_train.npy`: The vectorized feature matrix for the training dataset.
- `y_train.npy`: The labels for the training dataset.
- `X_test.npy`: The vectorized feature matrix for the test dataset.
- `y_test.npy`: The labels for the test dataset.

Make sure all these files are in the same directory when proceeding to the next step.

---

### Step 2: Train and Collect Knowledge

Once the dataset is prepared, you can proceed to train the model and collect knowledge for all tokens in the IMDb dataset. To do this, run `collect.py`.

Before running this script, ensure the following conditions are met:

- The files `vectorizer_X.pickle`, `X_train.npy`, `y_train.npy`, `X_test.npy`, and `y_test.npy` must be in the same directory as the `collect.py` script.
  
`collect.py` will:

- Train a TM autoencoder on the IMDb dataset using the files generated in the previous step.
- Collect knowledge for each token in the vocabulary.
- Store the knowledge in individual pickle files for each token.

#### To Run:

```bash
python collect.py
```

#### Outputs:

- A directory called `IMDbKnowledge` will be created.
- For each token in the vocabulary, a corresponding pickle file will be generated and stored in the `IMDbKnowledge` folder.

---
### Step 3: Data Augmentation with EDA and Knowledge
Use Report.ipynb for data augmentation and synonym replacement using EDA (Easy Data Augmentation). This step generates an augmented text file based on the input file.

---
### Folder Structure

After running the above steps, your folder should look like this:

```
.
├── Report.ipynb
├── prepare.py
├── collect.py
├── eda.py
├── IMDbKnowledge/
│   ├── 1.pickle
│   ├── 2.pickle
│   ├── ...
├── vectorizer_X.pickle
├── X_train.npy
├── y_train.npy
├── X_test.npy
├── y_test.npy
├── input_text.txt
├── output_text.txt
```
