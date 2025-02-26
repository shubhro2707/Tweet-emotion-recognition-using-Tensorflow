import pandas as pd

# Load dataset while skipping problematic rows
file_path = "/content/smile-annotations-final.csv"
df = pd.read_csv(file_path, sep=';', on_bad_lines='skip')

# Display the first few rows
print(df.head())

# Check the structure of the dataset
print(df.info())

import pandas as pd

# Load dataset with proper column separation
file_path = "/content/smile-annotations-final.csv"

df = pd.read_csv(file_path, sep=',', names=['id', 'text', 'category'], header=None, on_bad_lines='skip')

# Display the first few rows
print(df.head())
print(df.info())

import pandas as pd
import numpy as np
import tensorflow as tf
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Load dataset
file_path = "/content/smile-annotations-final.csv"
df = pd.read_csv(file_path, names=["id", "text", "category"], sep=",", skiprows=1)

# Drop "nocode" rows
df = df[df["category"] != "nocode"]

# Extract text and labels
texts = df["text"].values
labels = df["category"].values

# Encode labels as numbers
unique_labels = list(set(labels))
label_to_index = {label: i for i, label in enumerate(unique_labels)}
index_to_label = {i: label for label, i in label_to_index.items()}
labels = np.array([label_to_index[label] for label in labels])

# One-hot encode labels
num_classes = len(unique_labels)
labels = to_categorical(labels, num_classes=num_classes)

# Text preprocessing function
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r"@\w+|\#", "", text)  # Remove mentions and hashtags
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuations
    return text

# Apply text cleaning
texts = [clean_text(text) for text in texts]

# Tokenization
max_words = 10000  # Vocabulary size
max_len = 50  # Max tweet length

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(max_words, 64, input_length=max_len),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation="softmax")
])

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Plot training performance
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Evaluate on test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

import pandas as pd
import numpy as np
import tensorflow as tf
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Load dataset
file_path = "/content/smile-annotations-final.csv"  # Ensure correct path
df = pd.read_csv(file_path, names=["id", "text", "category"], sep=",", skiprows=1)

# Drop "nocode" rows (tweets without emotion labels)
df = df[df["category"] != "nocode"]

# Extract text and labels
texts = df["text"].values
labels = df["category"].values

# Encode labels as numbers
unique_labels = list(set(labels))
label_to_index = {label: i for i, label in enumerate(unique_labels)}
index_to_label = {i: label for label, i in label_to_index.items()}
labels = np.array([label_to_index[label] for label in labels])

# One-hot encode labels
num_classes = len(unique_labels)
labels = to_categorical(labels, num_classes=num_classes)

# Text preprocessing function
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r"@\w+|\#", "", text)  # Remove mentions and hashtags
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuations
    return text.strip()

# Apply text cleaning
texts = [clean_text(text) for text in texts]

# Tokenization
max_words = 10000  # Vocabulary size
max_len = 50  # Max tweet length

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

# Adjusted Model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(max_words, 100, input_length=max_len),
    tf.keras.layers.SpatialDropout1D(0.3),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True, kernel_regularizer=l2(0.01))),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, kernel_regularizer=l2(0.01))),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=l2(0.01)),
    tf.keras.layers.Dense(num_classes, activation="softmax")
])

# Adjust learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

# Compile
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# ReduceLROnPlateau callback to dynamically lower learning rate when validation loss plateaus
lr_scheduler = ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5, verbose=1)

# Train the model
history = model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test), callbacks=[lr_scheduler])

# Plot training performance
plt.figure(figsize=(12, 5))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.show()

# Evaluate on test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Function to make predictions
def predict_emotion(text):
    text = clean_text(text)
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding="post", truncating="post")
    prediction = model.predict(padded_sequence)
    emotion = index_to_label[np.argmax(prediction)]
    return emotion

# Example prediction
sample_tweet = "I am so excited for this opportunity!"
print(f"Predicted Emotion: {predict_emotion(sample_tweet)}")
