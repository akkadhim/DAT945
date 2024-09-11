# IMDb Dataset Preparation and Knowledge Collection using Tsetlin Machine Autoencoder

This repository contains code to prepare the IMDb dataset for training and to collect knowledge using a TM autoencoder. Below is a step-by-step guide on how to use the scripts in this repository.

---

### Requirements

Make sure you have the following libraries installed before running the scripts:

```bash
pip install git+https://github.com/cair/tmu.git
```

---

## Files in the Repository

- **`Prepare.py`**: This script prepares the IMDb dataset for training.
- **`CollectKnowledge.py`**: This script uses a TM autoencoder to train the model and collect knowledge for each token in the IMDb dataset.

## Steps to Use the Repository

### Step 1: Prepare the IMDb Dataset

First, you need to run `Prepare.py` to process and prepare the IMDb dataset for training. This script performs the following tasks:

- It processes the IMDb dataset.
- It generates a vectorizer and saves it as a pickle file (`vectorizer_X.pickle`).
- It saves the training and testing datasets as `.npy` files:
  - `X_train.npy`: The vectorized training data.
  - `y_train.npy`: The training labels.
  - `X_test.npy`: The vectorized test data.
  - `y_test.npy`: The test labels.

#### To Run:

```bash
python Prepare.py
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

Once the dataset is prepared, you can proceed to train the model and collect knowledge for all tokens in the IMDb dataset. To do this, run `CollectKnowledge.py`.

Before running this script, ensure the following conditions are met:

- The files `vectorizer_X.pickle`, `X_train.npy`, `y_train.npy`, `X_test.npy`, and `y_test.npy` must be in the same directory as the `CollectKnowledge.py` script.
  
`CollectKnowledge.py` will:

- Train a TM autoencoder on the IMDb dataset using the files generated in the previous step.
- Collect knowledge for each token in the vocabulary.
- Store the knowledge in individual pickle files for each token.

#### To Run:

```bash
python CollectKnowledge.py
```

#### Outputs:

- A directory called `IMDbKnowledge` will be created.
- For each token in the vocabulary, a corresponding pickle file will be generated and stored in the `IMDbKnowledge` folder.

---

### Folder Structure

After running the above steps, your folder should look like this:

```
.
├── CollectKnowledge.py
├── Prepare.py
├── IMDbKnowledge/
│   ├── 1.pickle
│   ├── 2.pickle
│   ├── ...
├── vectorizer_X.pickle
├── X_train.npy
├── y_train.npy
├── X_test.npy
├── y_test.npy
```
