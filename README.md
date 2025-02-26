# Tweet Sentiment Classification  

## **Purpose**  
This project classifies tweets into different sentiment categories using deep learning. It leverages NLP techniques and LSTM-based neural networks to analyze tweet text and predict sentiment.  

---  

## **Architecture**  

### **1. Data Preprocessing**  
- Cleaning text (removing URLs, mentions, punctuation)  
- Tokenization and padding  
- One-hot encoding sentiment labels  

### **2. Model Architecture**  
- **Embedding Layer** with SpatialDropout1D  
- **Bidirectional LSTM** layers with L2 regularization  
- **Fully Connected Dense Layers**  
- **Dropout Layers** to prevent overfitting  
- **Adam Optimizer** with ReduceLROnPlateau for adaptive learning  

### **3. Training & Evaluation**  
- Model trained on processed tweet data  
- Accuracy and loss trends plotted  
- Performance evaluated on a test set  

---  

## **Tools & Frameworks Used**  
- **Programming**: Python  
- **Deep Learning**: TensorFlow, Keras  
- **Text Preprocessing**: NLTK, Regex (`re`)  
- **Data Handling**: Pandas, NumPy  
- **Train-Test Split**: Scikit-Learn  
- **Data Visualization**: Matplotlib  
