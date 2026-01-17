# 🎬 ML-IMDB-Dataset: Sentiment Analysis on Movie Reviews

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/ML-Sentiment%20Analysis-brightgreen)
![Status](https://img.shields.io/badge/Status-Complete-success)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📖 Overview

This project performs **sentiment analysis** on the IMDB movie reviews dataset using machine learning techniques. The goal is to classify movie reviews as **positive** or **negative** based on their textual content. The project leverages natural language processing (NLP) techniques including text preprocessing, feature extraction using **Bag of Words (BoW)**, and classification using **K-Nearest Neighbors (KNN)**.

**Problem:** Movie reviews contain valuable sentiment information, but manually classifying thousands of reviews is impractical.

**Solution:** Build a machine learning pipeline that automatically classifies reviews as positive or negative with high accuracy using text vectorization and KNN classification.

---

## 📂 Folder Structure

```
ML-IMDB-Dataset/
│
├── notebooks/                                    # 📓 Jupyter notebooks with analysis
│   └── Sentiment_Analysis_knn_pipeline.ipynb   # Main sentiment analysis notebook
│
├── IMDB Dataset.csv                             # 📊 Original IMDB reviews dataset
├── preprocessed_imdb_reviews.csv               # 🧹 Cleaned and preprocessed data
├── requirements.txt                             # 📦 Python dependencies
├── .gitignore                                   # 🚫 Files to exclude from Git
├── LICENSE                                      # 📜 MIT License
└── README.md                                    # 📖 Project documentation (this file)
```

---

## 📊 Dataset

- **Name:** IMDB Movie Reviews Dataset
- **Source:** [IMDB Dataset on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Size:** 50,000 movie reviews (25,000 positive, 25,000 negative)
- **Target Variable:** `sentiment` (positive/negative)
- **Features:** Movie review text

The dataset contains balanced classes with reviews labeled as either positive or negative sentiment.

---

## 🎯 Key Results

- **Model Used:** K-Nearest Neighbors (KNN) with Bag of Words (BoW) vectorization
- **Best Parameters:** n_neighbors=10, p=2 (Euclidean distance)
- **Training F1-Score:** ~0.85+
- **Test F1-Score:** ~0.80+
- **Text Preprocessing:** HTML tag removal, lowercasing, stopword removal, stemming
- **Feature Extraction:** Bag of Words (BoW) with CountVectorizer
- **Hyperparameter Tuning:** GridSearchCV for optimal KNN parameters

The model successfully classifies movie reviews with strong performance, demonstrating effective text preprocessing and feature engineering techniques.

---

## 🚀 Installation & Usage

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/TasinAhmed2508/ML-IMDB-Dataset.git
cd ML-IMDB-Dataset
```

### 2️⃣ Install Dependencies

Make sure you have Python 3.8+ installed, then install the required libraries:

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Notebook

Navigate to the `notebooks/` folder and open the Jupyter notebook:

```bash
jupyter notebook notebooks/Sentiment_Analysis_knn_pipeline.ipynb
```

Or use VS Code with the Jupyter extension to run the notebook directly.

### 4️⃣ Execute the Pipeline

Run all cells in the notebook to:
1. Load and explore the IMDB dataset
2. Preprocess text data (HTML removal, stemming, stopword removal)
3. Perform exploratory data analysis (EDA)
4. Extract features using Bag of Words
5. Train KNN classifier with hyperparameter tuning
6. Evaluate model performance with classification metrics

---

## 🛠️ Technologies Used

- **Python 3.8+**
- **pandas** - Data manipulation
- **NumPy** - Numerical computing
- **scikit-learn** - Machine learning models and pipelines
- **matplotlib & seaborn** - Data visualization
- **NLTK/BeautifulSoup** - Text preprocessing
- **Jupyter Notebook** - Interactive development

---

## 📈 Project Workflow

1. **Data Loading:** Import IMDB dataset with 50K reviews
2. **EDA:** Analyze distribution, review lengths, and class balance
3. **Text Preprocessing:** Clean HTML, remove stopwords, apply stemming
4. **Feature Engineering:** Convert text to numerical features using BoW
5. **Model Training:** Train KNN classifier with GridSearchCV
6. **Evaluation:** Assess performance using F1-score, confusion matrix, and classification report

---

## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

**Copyright © 2026 Tasin Ahmed**

---

## 👤 Author

**Tasin Ahmed**

### Connect with Me:

[![GitHub](https://img.shields.io/badge/GitHub-TasinAhmed2508-181717?logo=github)](https://github.com/TasinAhmed2508)

---

⭐ **If you find this project helpful, please consider giving it a star!**
