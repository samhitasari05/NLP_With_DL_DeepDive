# 🧠 Word2Vec Skip-Gram: Custom Embeddings with Positive & Negative Sampling in PyTorch

This project presents a **custom implementation of the Word2Vec Skip-Gram model** using both **positive and negative sampling** from scratch in **PyTorch**. The objective was to build meaningful word embeddings and compare the semantic capture capabilities of different neural architectures.

---

## 💡 Project Summary

- Preprocessed a small real-world corpus by cleaning, tokenizing, and generating a vocabulary.
- Created **word-context pairs** using a fixed-size context window.
- Implemented **positive sampling** for true word-context pairs and **negative sampling** by randomly selecting unrelated words.
- Built and trained two models from scratch using PyTorch:
  - ✅ **Logistic Regression**
  - ✅ **Feedforward Neural Network (FFNN)**
- Extracted learned word embeddings and visualized them using **T-SNE**.
- Evaluated performance and compared outputs across models using classification metrics.

---

## 🧰 Techniques Implemented

| Component              | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| Skip-Gram Architecture | Center word predicts context words                                          |
| Positive Sampling      | Real context word pairs (label = 1)                                         |
| Negative Sampling      | Random non-context word pairs (label = 0)                                   |
| Binary Classification  | Logistic and FFNN trained on combined samples                               |
| Embedding Extraction   | Trained weights used as dense vector representations of words               |
| Visualization          | T-SNE used to project high-dimensional embeddings into 2D for clustering    |

---

## 🧪 Model Evaluation

You trained both models and evaluated their ability to classify word-context pairs and learn semantic embeddings.

### ✅ Logistic Regression Results:
- **Precision:** 0.92  
- **Recall:** 0.86  
- **F1-Score:** 0.88  

💬 *Interpretation:*  
The logistic regression model showed strong performance with fewer parameters, offering a solid baseline. It was fast to train and accurately identified most true relationships, though it lacked the ability to capture subtle semantics.

---

### ✅ Feedforward Neural Network (FFNN) Results:
- **Precision:** 0.91  
- **Recall:** 0.92  
- **F1-Score:** 0.91  

💬 *Interpretation:*  
The FFNN outperformed logistic regression in capturing nuanced semantic patterns. With an additional hidden layer and ReLU activation, it achieved a better balance of precision and recall. The embeddings learned were also more separable in T-SNE plots, showing improved clustering.

---

## 📊 Embedding Visualization

T-SNE was used to reduce the dimensionality of word embeddings for visual inspection.

- Embeddings from both models were plotted.
- **Clusters emerged** showing semantically similar words grouped together (e.g., “airline”, “flight”, “trip”).
- FFNN-based embeddings displayed tighter, more meaningful groupings.

---

## 📁 Project Structure

| File                                | Description                                         |
|-------------------------------------|-----------------------------------------------------|
| `Word2Vec.ipynb`               | Full PyTorch code: preprocessing, training, results |
| `logistic_regression_results_new.txt` | Output predictions from Logistic Regression         |
| `ffnn_results_new.txt`             | Output predictions from Feedforward Neural Net      |
| `Word2Vec_samhitasarikonda.html` | HTML export of notebook for easy preview            |
---

## 📚 Key Takeaways

- Understood the intuition behind Skip-Gram and negative sampling.
- Practiced manual data pair generation and batch construction.
- Implemented PyTorch training loops and custom loss computation.
- Observed trade-offs between shallow and deep neural architectures.
- Gained confidence in interpreting vector embeddings using T-SNE.
---

## 🔗 Connect with Me

If you're interested in embeddings, NLP, or deep learning research, feel free to connect and collaborate!

- 📘 [LinkedIn](https://linkedin.com/in/samhita-sarikonda)  
