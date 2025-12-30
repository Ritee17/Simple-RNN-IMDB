# 🎬 IMDB Movie Review Sentiment Analysis with Simple RNN

A Deep Learning project that performs sentiment analysis on movie reviews using the IMDB dataset. This project utilizes a **Simple RNN** (Recurrent Neural Network) built with TensorFlow/Keras and features a user-friendly web interface deployed using **Streamlit**.

## 🚀 Project Overview

The goal of this project is to classify movie reviews as either **Positive** or **Negative**. The model processes natural language text, converting words into integer sequences, and uses a Recurrent Neural Network to understand the context and sentiment of the review.

### ✨ Key Features
* **Deep Learning Model:** Built using a Simple RNN architecture suitable for sequential text data.
* **Dataset:** Trained on the standard IMDB dataset (25,000 training reviews).
* **Text Preprocessing:** Implements tokenization and sequence padding to handle varying review lengths.
* **Visualization:** Utilizes **TensorBoard** to track training metrics (loss and accuracy) over epochs.
* **Deployment:** Includes a **Streamlit** web application for real-time, interactive sentiment prediction.

## 🛠️ Tech Stack
* **Python**
* **TensorFlow / Keras** (Model building & training)
* **NumPy** (Data manipulation)
* **Streamlit** (Web interface)
* **TensorBoard** (Performance visualization)

## 📂 Project Structure
* `IMDB_rnn.ipynb`: Jupyter Notebook for loading data, building the RNN model, training with callbacks (EarlyStopping, TensorBoard), and saving the model.
* `main.py`: The Streamlit application script that loads the saved model and creates the web interface.
* `prediction.ipynb`: A testing notebook to load the model and run predictions on raw text examples.
* `simple_rnn_imdb.h5`: The saved pre-trained model file.
* `logs/`: Directory containing TensorBoard logs.

## 📊 Model Architecture
1.  **Embedding Layer:** Converts integer-encoded words into dense vectors of fixed size.
2.  **SimpleRNN Layer:** A recurrent layer with 128 units (ReLU activation) to capture sequential patterns.
3.  **Dense Layer:** A single output unit with Sigmoid activation to output a probability score (0 to 1).

## 💻 How to Run

### 1. Clone the repository
```bash
git clone [https://github.com/yourusername/imdb-sentiment-analysis.git](https://github.com/yourusername/imdb-sentiment-analysis.git)
cd imdb-sentiment-analysis
2. Install Dependencies
Ensure you have the necessary libraries installed:

Bash

pip install tensorflow numpy streamlit
3. Train the Model (Optional)
If you want to retrain the model from scratch, run the IMDB_rnn.ipynb notebook. This will generate the simple_rnn_imdb.h5 file and TensorBoard logs.

To view TensorBoard training progress:

Bash

tensorboard --logdir=logs/
4. Run the Streamlit App
Launch the web interface to test the model with your own reviews:

Bash

streamlit run main.py
'''
📈 Results
The model achieves a validation accuracy of approximately 83% on the IMDB dataset, demonstrating the effectiveness of RNNs for basic natural language processing tasks.
