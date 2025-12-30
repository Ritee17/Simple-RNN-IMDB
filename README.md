# 🎬 IMDB Movie Review Sentiment Analysis with Simple RNN

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)

**🔴 Live Demo:** [Click here to try the App](https://simple-rnn-imdb-a5nqxlcmzz3jtxbjgdzqgw.streamlit.app/)

A Deep Learning project that performs sentiment analysis on movie reviews using the IMDB dataset. This project utilizes a **Simple RNN** (Recurrent Neural Network) built with TensorFlow/Keras and features a user-friendly web interface deployed using **Streamlit**.

## 🚀 Project Overview

The goal of this project is to classify movie reviews as either **Positive** or **Negative**. The model processes natural language text, converting words into integer sequences, and uses a Recurrent Neural Network to understand the context and sentiment of the review.

### ✨ Key Features
* **Deep Learning Model:** Built using a Simple RNN architecture suitable for sequential text data.
* **Dataset:** Trained on the standard IMDB dataset (25,000 training reviews).
* **Text Preprocessing:** Implements tokenization, decoding, and sequence padding to handle varying review lengths.
* **Visualization:** Utilizes **TensorBoard** to track training metrics (loss and accuracy) over epochs.
* **Deployment:** Includes a **Streamlit** web application for real-time, interactive sentiment prediction.

## 🛠️ Tech Stack
* **Python**
* **TensorFlow / Keras** (Model building & training)
* **NumPy** (Data manipulation)
* **Streamlit** (Web interface)
* **TensorBoard** (Performance visualization)

## 📂 Project Structure
| File Name | Description |
| :--- | :--- |
| `IMDB_rnn.ipynb` | Jupyter Notebook for data loading, preprocessing, model training, and saving the model. |
| `main.py` | The Streamlit application script that loads the trained model and creates the web interface. |
| `prediction.ipynb` | A testing notebook to load the model and run predictions on raw text examples. |
| `simple_rnn_imdb.h5` | The saved pre-trained model file generated after training. |
| `requirements.txt` | List of dependencies required to run the project. |
| `logs/` | Directory containing TensorBoard logs for visualization. |

## 💻 Installation & Setup

### 1. Clone the repository
```bash
git clone [https://github.com/yourusername/imdb-sentiment-analysis.git](https://github.com/yourusername/imdb-sentiment-analysis.git)
cd imdb-sentiment-analysis

```

### 2. Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

```

### 3. Install Dependencies

Install the required libraries using the requirements file:

```bash
pip install -r requirements.txt

```

*Note: Your `requirements.txt` should typically include:*

* `numpy`
* `tensorflow`
* `streamlit`

## 🚀 How to Run

### 1. Train the Model (Optional)

If you want to retrain the model from scratch, run the `IMDB_rnn.ipynb` notebook. This will generate the `simple_rnn_imdb.h5` file and TensorBoard logs.

### 2. Visualize Training with TensorBoard

To view the training graphs (Accuracy vs. Loss):

```bash
tensorboard --logdir=logs/fit

```

### 3. Launch the Web App

Run the Streamlit application to use the model interactively:

```bash
streamlit run main.py

```

Once the app is running, enter any movie review in the text box and click **Classify** to see if the sentiment is Positive or Negative.

## 📊 Model Architecture

The model consists of the following layers:

1. **Embedding Layer:** Converts integer-encoded words into dense vectors of fixed size (128).
2. **SimpleRNN Layer:** A recurrent layer with 128 units (ReLU activation) to capture sequential patterns.
3. **Dense Layer:** A single output unit with Sigmoid activation to output a probability score (0 to 1).

## 📈 Results

The model achieves a validation accuracy of approximately **83%** on the IMDB dataset.


*Created by Ritee*

