# Project: Otto Group Product Classification Challenge

### Overview
This project focuses on building a robust machine learning model to classify products into multiple categories based on their features. The project was inspired by the Kaggle competition "Otto Group Product Classification Challenge," and aims to tackle real-world challenges such as feature selection, data preprocessing, model evaluation, and deployment preparation.

---

### Objectives
1. Develop a machine learning pipeline for multi-class classification.
2. Experiment with feature selection techniques.
3. Address data preprocessing and imbalanced datasets.
4. Evaluate model performance using various metrics.
5. Prepare the project for sharing and reproducibility.

---

### Steps Followed

#### **1. Data Loading and Preprocessing**
- **Dataset**: The raw data was loaded from `../data/raw/train.csv`.
- **Target Variable**: The target variable was addressed separately for each class. A binary classification approach was applied where each class was defined as "class x," and all other classes were treated as "not class x."
- **Preprocessing**: Since no missing values were found in the dataset, no imputation was necessary. However, the data was processed individually for each class, with specific attention given to each class separately.
- **Train-Test Split**: Data was split into 80% training and 20% testing using a random state for reproducibility.

#### **2. Feature Selection**
- **Technique**: Random Forest-based feature importance was used to select the top `n` features.
- **Outcome**: Only the most significant features were retained for model building, improving both efficiency and interpretability.

#### **3. Handling Imbalanced Data**
- **Sampling Methods**: Different oversampling and undersampling techniques (e.g., SMOTE) were tested.
- **Challenges**: Balancing class distributions while avoiding overfitting.

#### **4. Model Development**
- **Base Models**:
  - Random Forest
  - XGBoostClassifier
- **Ensemble Model**:
  - Developed a custom ensemble model combining predictions from multiple classifiers.
  - Implemented a `ClassFinalClassifier` to unify predictions.

#### **5. Model Evaluation**
- Metrics used:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
- Additional analysis:
  - Confusion Matrix
  - Cross-validation with 5 folds to ensure model stability.

#### **6. Addressing Overfitting**
- Compared training and testing accuracy to identify overfitting.
- Cross-validation results were used to confirm the generalizability of the model.

#### **7. Automation and Reporting**
- A `generate_report` function was created to generate textual evaluation reports.
- Reports included:
  - Model metrics
  - Confusion Matrix
  - Classification Report
  
#### **8. Saving Predictions**
- Final predictions were saved in a DataFrame for easy analysis and comparison.

---

### Challenges and Solutions
1. **Imbalanced Data**:
   - Challenge: Certain classes were underrepresented, affecting model performance.
   - Solution: SMOTE oversampling improved balance and overall accuracy.

2. **Feature Selection**:
   - Challenge: Too many features increased model complexity.
   - Solution: Random Forest feature importance helped reduce the feature set.

3. **Overfitting**:
   - Challenge: Models performed well on training data but poorly on test data.
   - Solution: Cross-validation and hyperparameter tuning mitigated this issue.

4. **Environment Management**:
   - Challenge: Ensuring reproducibility across systems.
   - Solution: Created `environment.yml` and `requirements.txt` files.

---

### Tools and Libraries Used
- **Programming Language**: Python
- **Libraries**:
  - pandas, numpy (Data manipulation)
  - scikit-learn (Modeling and evaluation)
  - matplotlib, seaborn (Visualization)
  - imbalanced-learn (Handling imbalanced datasets)

---

### File Structure
```plaintext
Project/
├── data/
│   ├── raw/
│   ├── processed/
├── src/
│   ├── models/
│   ├── utils/
│   ├── main.py
├── reports/
├── README.md
├── environment.yml
├── requirements.txt
```

---

### Installation
To set up the project environment:

Using Conda:
```bash
conda env create -f environment.yml
conda activate your_project_name
```

Using Pip:
```bash
pip install -r requirements.txt
```

---

### Conclusion
This project provided valuable insights into handling real-world classification problems. By focusing on data preprocessing, feature selection, and robust evaluation methods, we achieved a scalable and interpretable machine learning pipeline. Future work could include hyperparameter tuning and testing more advanced ensemble methods.

---

### Acknowledgements
Special thanks to Kaggle for providing the dataset and the inspiration for this challenge.

---

### My Thoughts and Insights
This project helped me tremendously in expanding my knowledge of data science and machine learning. One of the key elements was the development of the custom ClassFinalClassifier, which played a pivotal role in improving the overall performance of the models.

Additionally, this experience allowed me to explore different techniques such as handling imbalanced data with SMOTE, improving feature selection with Random Forest importance, and addressing overfitting with cross-validation. Although the project was inspired by a Kaggle challenge, my primary goal was to develop my skills further, rather than simply competing for a ranking.

By tackling the challenges in data preprocessing, model building, and evaluation, I was able to deepen my understanding of how to construct end-to-end machine learning pipelines.

---
