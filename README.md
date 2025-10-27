# Sentiment Analysis of Twitter and Reddit using Reinforcement Learning

## 🧩 Problem Statement
This project aims to perform **sentiment analysis** (positive, neutral, negative) on text data from **Twitter and Reddit** using **Reinforcement Learning (RL)** to enhance the consistency and generalization of sentiment classification models.

---

## 📊 Dataset Details
- **Source:** `/kaggle/input/twitter-and-reddit-sentimental-analysis-dataset/`
- **Content:** Combined Reddit and Twitter comments labeled with sentiment categories:  
  - `-1` → Negative  
  - `0` → Neutral  
  - `1` → Positive  
- **Total Samples:** 162,969  
- **Split:**  
  - Training set: 130,375 samples (80%)  
  - Holdout test set: 32,594 samples (20%)  

### Sampling Strategies Implemented
1. **Balanced Undersampling** – Equal samples from each sentiment class.  
2. **Stratified Sampling** – Preserves original class distribution.  
3. **Hybrid Sampling** – Oversamples minority and undersamples majority classes.

---

## ⚙️ Methodology

### 1. Data Preprocessing
- Removal of special characters and stopwords.  
- Normalization and tokenization of text.  
- Label encoding and dataset balancing using the three sampling strategies.  

### 2. Feature Engineering
- Sentence vectorization using probabilistic embeddings.  
- Reinforcement signal created from model prediction confidence.  

### 3. Reinforcement Learning Framework
- RL reward based on prediction correctness.  
- Exploration–exploitation mechanism (similar to multi-armed bandit).  
- Adaptive sampling weights adjusted dynamically based on model feedback.  

### 4. Model Training & Evaluation
- Models trained and validated on each sampling strategy.  
- Validation performed on 80% of data; holdout test on 20%.  
- Performance compared across accuracy, precision, recall, and F1-score.

---

## 🧠 Results

| Sampling Strategy | Validation Accuracy | Holdout Accuracy | Drop (%) | Remarks |
|--------------------|--------------------|------------------|----------|----------|
| Balanced Undersampling | 83.46% | 79.18% | -5.1% | Slight overfit, decent stability |
| Stratified (Original) | 77.10% | 77.01% | -0.1% | Most stable but lowest performance |
| **Hybrid Sampling** | **87.17%** | **82.87%** | **-4.9%** | **Best generalization and performance** |

### 📈 Additional Insights
- **Validation Accuracy (80% data):**  
  - Balanced: 0.8346  
  - Stratified: 0.7710  
  - Hybrid: 0.8717  
- **Holdout Test Accuracy (20% unseen):**  
  - Balanced: 0.7918  
  - Stratified: 0.7701  
  - Hybrid: 0.8287  

🏆 **Best Model:** Hybrid Sampling  
✅ **Final Holdout Accuracy:** 82.87%  
✅ **Generalizes well across unseen data**

---

## 🧾 Conclusion
Hybrid sampling provides the optimal balance between data diversity and class balance. Reinforcement-driven resampling and weight updates improved stability and overall model performance compared to conventional supervised learning baselines.

---

## 📚 Technologies Used
- Python, Pandas, NumPy, Scikit-learn  
- NLTK for text preprocessing  
- Reinforcement Learning (custom bandit-style reward updates)  
- Matplotlib / Seaborn for visualization  
- Jupyter Notebook (Kaggle Environment)

---

## ✨ Author
Developed by **Akshat**  
> “Harnessing conscious learning through reinforcement.”
