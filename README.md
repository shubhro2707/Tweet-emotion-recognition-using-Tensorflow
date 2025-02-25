Tweet Sentiment Classification

Purpose
This project classifies tweets into different sentiment categories using a deep learning model. It utilizes NLP techniques and LSTM-based neural networks to analyze tweet text and predict sentiment.

Architecture

Data Preprocessing:
Cleaning text (removing URLs, mentions, punctuation)
Tokenization and padding
One-hot encoding sentiment labels

Model Architecture:
Embedding Layer with SpatialDropout1D
Bidirectional LSTM layers with L2 regularization
Fully connected Dense layers
Dropout layers to prevent overfitting
Adam optimizer with ReduceLROnPlateau for adaptive learning

Training & Evaluation:
Model trained on processed tweet data
Plots accuracy and loss trends
Evaluates performance on a test set

Tools & Frameworks Used
Python
TensorFlow / Keras (Deep Learning)
NLTK / re (Text Preprocessing)
Scikit-Learn (Train-Test Split)
Matplotlib (Data Visualization)
Pandas / NumPy (Data Handling)
