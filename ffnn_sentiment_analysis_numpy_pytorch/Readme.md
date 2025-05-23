# 💬 Sentiment Classification using FFNNs: NumPy vs PyTorch

This project compares two Feed-Forward Neural Network (FFNN) models for classifying airline tweets into **positive** or **negative** sentiment:
- A custom FFNN built from **scratch using NumPy**
- A modern FFNN built using **PyTorch**

🧪 Completed for AIT 726 – NLP with Deep Learning at George Mason University.

---

## 🧠 Project Overview

This project explores two distinct approaches to sentiment classification using deep learning:
- Both models process raw airline tweets into TF-IDF vectors
- Preprocessing includes optional stemming using NLTK's SnowballStemmer
- Evaluated both **with** and **without** stemming

---

## ✈️ Dataset Summary

- 4183 tweets for training  
- 4182 tweets for testing  
- Tweets are labeled as `positive` or `negative`
- Dataset split across folders (train/test, pos/neg)

---

## 🧪 Experiments Conducted

| Model     | Stemming | Accuracy | Confusion Matrix         |
|-----------|----------|----------|--------------------------|
| NumPy     | ✅ Yes    | 71.74%   | [[3000, 0], [1182, 0]]   |
| NumPy     | ❌ No     | 71.74%   | [[3000, 0], [1182, 0]]   |
| PyTorch   | ✅ Yes    | 75.49%   | [[3000, 0], [1025, 157]] |
| PyTorch   | ❌ No     | **76.71%** | [[2999, 1], [973, 209]]  |

➡️ **PyTorch without stemming performed best**, correctly predicting over 200 positive tweets and showing balanced generalization.

---

## 📉 Training Loss Curves

![Training Loss](loss_curves.png)

- **NumPy models** converged with slightly lower loss but failed to generalize
- **PyTorch models** demonstrated more adaptive training (Adam optimizer)

---

## ⚙️ Model Configurations

| Parameter       | Value         |
|-----------------|---------------|
| Hidden Units    | 20            |
| Epochs          | 50            |
| Batch Size      | 32            |
| Optimizer       | SGD (NumPy), Adam (PyTorch) |
| Loss Function   | Mean Squared Error (MSE) |
| Activation      | Sigmoid (NumPy), ReLU+Sigmoid (PyTorch) |

---

## 🔍 Key Learnings

- **PyTorch outperformed NumPy** due to better optimizers, stability, and tensor operations
- **Stemming reduced vocabulary** size (by ~16%) but also slightly hurt performance
- **NumPy model learned to predict majority class** only (negative), failing on positive sentiment
- **Confusion matrix analysis revealed true model effectiveness**, not just raw accuracy

---

## 📂 Files in This Repository

| File | Description |
|------|-------------|
| `Assignment_2_v2_0.ipynb` | Full implementation notebook |
| `Airline Tweet Sentiment Classification Report.docx` | Final report |
| `results_comparison.txt` | Accuracy and confusion matrix |
| `model_analysis.txt` | Detailed model comparison and evaluation |
| `ffnn_sentiment_classification_20250228_221701.log` | Experiment logs |
| `loss_curves.png` | Training loss plot |

---

## 📚 Technologies Used

```bash
NumPy
PyTorch
NLTK
Matplotlib

