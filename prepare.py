import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.text import text_to_word_sequence

# Parameters
NUM_WORDS = 20000  # Example number of features
BINARY = True

# Preprocessing function
def preprocess_text(text):
    return ' '.join(text_to_word_sequence(text))

# Load IMDb data
def load_imdb_data(num_words=NUM_WORDS):
    # Load the IMDb dataset
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=num_words)
    
    # Reverse the word index dictionary to get the word for each index
    word_index = imdb.get_word_index()
    reverse_word_index = {index + 3: word for word, index in word_index.items()}
    reverse_word_index[0] = ' '
    reverse_word_index[1] = '[UNK]'
    reverse_word_index[2] = '[START]'
    reverse_word_index[3] = '[UNUSED]'
    
    # Decode the reviews back to text
    def decode_review(text):
        return ' '.join([reverse_word_index.get(i, '?') for i in text])

    X_train = [decode_review(x) for x in X_train]
    X_test = [decode_review(x) for x in X_test]

    return X_train, y_train, X_test, y_test

# Load data
X_train, y_train, X_test, y_test = load_imdb_data(num_words=NUM_WORDS)

# Vectorization
vectorizer_X = CountVectorizer(preprocessor=preprocess_text, max_features=NUM_WORDS, binary=BINARY)
X_train_vec = vectorizer_X.fit_transform(X_train)
X_test_vec = vectorizer_X.transform(X_test)

# Save data
f_vectorizer_X = open("vectorizer_X.pickle", "wb")
pickle.dump(vectorizer_X, f_vectorizer_X, protocol=4)
f_vectorizer_X.close()

np.save('X_train.npy', X_train_vec.toarray())
np.save('y_train.npy', y_train)
np.save('X_test.npy', X_test_vec.toarray())
np.save('y_test.npy', y_test)

print("Data preparation complete and files saved.")