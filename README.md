# **Tweet Sentiment Classification**  

## **Purpose**  
In this project, I built a deep learning model to **classify tweets** into different **sentiment categories**. The model leverages **NLP techniques** and an **LSTM-based neural network** to analyze tweet text and predict sentiment effectively.  

---  

## **Architecture**  

### **1. Data Preprocessing**  
I prepared the data by performing the following steps:  
- **Cleaning text** (removing URLs, mentions, punctuation, and special characters)  
- **Tokenization** and **padding** to ensure uniform input length  
- **One-hot encoding** of sentiment labels for training  

### **2. Model Architecture**  
The neural network architecture consists of:  
- **Embedding Layer** with **SpatialDropout1D** for word representation  
- **Bidirectional LSTM** layers with **L2 regularization** for sequential learning  
- **Fully Connected Dense Layers** for classification  
- **Dropout Layers** to prevent overfitting  
- **Adam Optimizer** with **ReduceLROnPlateau** for adaptive learning rate adjustments  

### **3. Training & Evaluation**  
- Trained the model on a **processed tweet dataset**  
- Plotted **accuracy** and **loss** trends during training  
- Evaluated performance on a **test set** to ensure generalization  

---  

## **Tools & Frameworks Used**  

### **Programming Language:**  
- **Python**  

### **Libraries & Frameworks:**  
- **Deep Learning:** TensorFlow, Keras  
- **Text Preprocessing:** NLTK, Regex (`re`)  
- **Data Handling:** Pandas, NumPy  
- **Train-Test Split:** Scikit-Learn  
- **Data Visualization:** Matplotlib  

This model effectively classifies tweets into sentiment categories, providing valuable insights for social media analysis and opinion mining.
  ![image](https://github.com/user-attachments/assets/e916aa43-0a5c-44d1-a26b-3546fadf4085)

