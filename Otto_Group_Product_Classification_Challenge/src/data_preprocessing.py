import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore", category=FutureWarning)


def load_and_preprocess_data(file_path, target_class, test_size=0.2, random_state=42):
    """
    This function loads the data file, splits into independent and dependent variables,
    splits into training and testing datasets, and processes the target class.
    
    Parameters:
    - file_path: Path to the data file
    - target_class: Target class for binary classification
    - test_size: Proportion of data to be used for testing (default: 0.2)
    - random_state: Random state for reproducibility (default: 42)
    
    Returns:
    - data_raw: Raw data (Pandas DataFrame)
    - X: Independent variables
    - y: Dependent variable
    - X_train: Training data (independent variables)
    - X_test: Test data (independent variables)
    - y_train: Training data (dependent variable)
    - y_test: Test data (dependent variable)
    """
    # Load the data
    data_raw = pd.read_csv(file_path)

    # Independent and dependent variables
    X = data_raw.drop(columns=['target', 'id'])  # Drop columns 'id' and 'target'
    y = (data_raw['target'] == target_class).astype(int)  # target_class: 1, others: 0

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=test_size, random_state=random_state)
    
    return data_raw, X, y, X_train, X_test, y_train, y_test


def create_specific_dataframe(train_data, target_class, target_column='target'):
    """
    Creates a binary target variable based on a specific class.
    
    Args:
        train_data (pd.DataFrame): Input dataset.
        target_class (Any): Target class value.
        target_column (str): Name of the column to be used for classification. Default is 'target'.
    
    Returns:
        pd.Series: Binary target variable (1: matching class, 0: others).
    """
    if target_column not in train_data.columns:
        raise ValueError(f"Column '{target_column}' not found in dataset.")
    
    if target_class not in train_data[target_column].unique():
        raise ValueError(f"Target class '{target_class}' not found in '{target_column}' column.")
    
    y = (train_data[target_column] == target_class).astype(int)
    return y


def plot_class_distribution(class_counts, figsize=(16, 8), fmt='.2f', cbar=True):
    """
    Visualizes class distribution using both pie chart and bar chart.
    
    Parameters:
        class_counts (dict or pd.Series): Class distribution (key: class, value: count).
        figsize (tuple): Figure size.
    """
    # Create figure and subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Pie Chart
    axes[0].pie(
        class_counts.values(),
        labels=class_counts.keys(),
        autopct='%1.1f%%',
        startangle=140,
        colors=plt.cm.Paired.colors
    )
    axes[0].set_title('Class Distribution (Pie Chart)', fontsize=14)

    # Bar Chart
    sns.barplot(
        x=list(class_counts.keys()),
        y=list(class_counts.values()),
        palette='viridis',
        ax=axes[1]
    )
    axes[1].set_title('Class Distribution (Bar Chart)', fontsize=14)
    axes[1].set_xlabel('Classes')
    axes[1].set_ylabel('Count')
    axes[1].tick_params(axis='x', rotation=45)

    # Show the plots
    plt.tight_layout()
    plt.show()


def plot_spearman_correlation(data, figsize=(20, 16), cmap='coolwarm'):
    """
    Computes and visualizes the Spearman correlation between variables in the dataset.

    Parameters:
        data (pd.DataFrame): The dataset for correlation analysis.
        figsize (tuple): Heatmap size. Default is (20, 16).
        cmap (str): Color map. Default is 'coolwarm'.

    Returns:
        spearman_corr (pd.DataFrame): Spearman correlation matrix.
    """
    # Compute Spearman correlation
    spearman_corr = data.corr(method='spearman')

    # Visualize the correlation
    plt.figure(figsize=figsize)
    sns.heatmap(spearman_corr, annot=True, cmap=cmap, fmt='.2f', cbar=True)
    plt.title('Spearman Correlation Heatmap', fontsize=16)
    plt.show()

    return spearman_corr


def random_forest_feature_selection(X, y, n_features_to_select='all', random_state=None):
    """
    Performs feature selection using Random Forest.

    Parameters:
    X (pd.DataFrame): Feature matrix.
    y (pd.Series): Target variable.
    n_features_to_select (int or 'all'): Number of features to select. 
        If 'all' is provided, all features will be used.
    random_state (int): Controls randomness for reproducibility.
    
    Returns:
    selected_features (list): Names of the selected features.
    X_selected (pd.DataFrame): New dataset with selected features.
    """
    # Create RandomForestClassifier model
    clf = RandomForestClassifier(random_state=random_state)
    
    # Train the model
    clf.fit(X, y)
    
    # Get feature importances
    feature_importances = clf.feature_importances_
    
    # Sort features by importance
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)
    
    print("Feature Importances:")
    print(feature_importance_df)
    
    # Select the specified number of features
    if n_features_to_select == 'all':
        selected_features = X.columns
    else:
        selected_features = feature_importance_df['Feature'][:n_features_to_select].values
    
    # Create a new dataset with selected features
    X_selected = X[selected_features]
        
    plt.figure(figsize=(20, 12))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.title('Feature Importances for the Target Class')
    plt.gca().invert_yaxis()
    plt.show()
    
    return selected_features, X_selected


def calculate_vif(X):
    """
    Calculates the Variance Inflation Factor (VIF) for each feature in the dataset.
    
    Parameters:
    - X: Feature matrix (independent variables)
    
    Returns:
    - vif_data: DataFrame containing VIF values for each feature
    """
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    print(vif_data)
    
    return vif_data
