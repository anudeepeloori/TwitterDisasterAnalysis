# Twitter Disaster Analysis: Comparative Study of Machine Learning and Deep Learning Approaches

## Project Overview

This repository presents the full implementation of our study on classifying disaster-related tweets using a range of machine learning and deep learning models. We evaluated:

* **Traditional Machine Learning:** Naive Bayes, Support Vector Machines (SVM), Random Forest, Gradient Boosting, XGBoost
* **Deep Learning:** Convolutional Neural Networks (CNN), Generative Adversarial Networks (GAN), and Transformers (BERT-based)

Our aim is to benchmark these methods and identify the most effective approach for reliable, real-time disaster tweet classification.

## Files

* **TwitterDisasterAnalysis.ipynb**:
  The complete notebook containing:

  * Data preprocessing
  * Model training (all traditional and deep learning models)
  * Evaluation metrics
  * Visualizations (confusion matrices, ROC-AUC curves)
  * Final analysis and discussion

* **README.md**:
  You are here!

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/anudeepeloori/twitter-disaster-analysis.git
cd twitter-disaster-analysis
```

### 2. Install Dependencies

Main dependencies include:

* pandas
* scikit-learn
* tensorflow / keras
* transformers (Hugging Face)
* matplotlib / seaborn

### 3. Download the Dataset

Get the **Twitter Disaster Dataset** from Kaggle
Place `train.csv` in the working directory.

Alternatively, download using Kaggle API:

```bash
kaggle datasets download -d "dataset"
unzip dataset.zip
```

## How to Run

Simply open the notebook and follow along step by step:

```bash
jupyter notebook TwitterDisasterAnalysis.ipynb
```



## Reproducibility

This notebook is designed for **full reproducibility**. It includes:

* Preprocessing routines for cleaning and tokenizing the tweet data
* Training procedures for all machine learning and deep learning models
* Evaluation routines with precise metrics matching the results reported in our study
* Visualizations to verify performance (confusion matrices, ROC-AUC)


## Results Snapshot

| Model                     | Accuracy | Precision | Recall | F1-Score |
| ------------------------- | -------- | --------- | ------ | -------- |
| Linear SVM                | 74.3%    | 73.8%     | 75.1%  | 77.1%    |
| Multinomial Naive Bayes   | 76.9%    | 76.5%     | 78.2%  | 78.3%    |
| Random Forest             | 73%      | 72.2%     | 76.0%  | 71%      |
| Gradient Boosting         | 76%      | 75.8%     | 77.3%  | 79.0%    |
| XGBoost                   | 78%      | 78.2%     | 73.5%  | 76.0%    |
| CNN                       | 71%      | 79.5%     | 71.1%  | 73%      |
| GAN                       | 68%      | 72.6%     | 65.4%  | 72%      |
| Transformers (BERT-based) | 82%      | 84.7%     | 81.2%  | 84%      |



## Contact

For any questions or collaborations:

* **Sahanya Pogaku:** [sahanyapogaku@usf.edu](mailto:sahanyapogaku@usf.edu)
* **Anudeep Eloori:** [anudeepeloori@usf.edu](mailto:anudeepeloori@usf.edu)
* **Sushma Gandham:** [sushmag@usf.edu](mailto:sushmag@usf.edu)
* **Karthik Eloori:** [karthikeloori@usf.edu](mailto:karthikeloori@usf.edu)

---
