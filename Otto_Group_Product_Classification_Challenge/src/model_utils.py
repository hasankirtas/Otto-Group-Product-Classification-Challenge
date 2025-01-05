import os
import zipfile
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc


def calculate_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def calculate_precision(y_true, y_pred, pos_label=1):
    return precision_score(y_true, y_pred, pos_label=pos_label)

def calculate_recall(y_true, y_pred, pos_label=1):
    return recall_score(y_true, y_pred, pos_label=pos_label)

def calculate_f1(y_true, y_pred, pos_label=1):
    return f1_score(y_true, y_pred, pos_label=pos_label)

def calculate_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)

def calculate_classification_report(y_true, y_pred):
    return classification_report(y_true, y_pred)

def generate_report(y_true, y_pred, model_name):
    """
    Generates a textual report with various evaluation metrics.
    """
    accuracy = calculate_accuracy(y_true, y_pred)
    precision = calculate_precision(y_true, y_pred)
    recall = calculate_recall(y_true, y_pred)
    f1 = calculate_f1(y_true, y_pred)
    conf_matrix = calculate_confusion_matrix(y_true, y_pred)
    class_report = calculate_classification_report(y_true, y_pred)

    # Write the report to a file
    report_text = f"""
    Model Evaluation Report for {model_name}:
    -------------------------------------------
    Accuracy: {accuracy}
    Precision: {precision}
    Recall: {recall}
    F1 Score: {f1}

    Confusion Matrix:
    {conf_matrix}

    Classification Report:
    {class_report}
    """
    
    # Save the report to a file
    report_filepath = f"{model_name}_report.txt"
    with open(report_filepath, 'w') as f:
        f.write(report_text)

def plot_metrics(y_true, y_pred, model_name):
    """
    Plots confusion matrix and ROC curve and saves them as image files.
    """
    # Confusion Matrix
    conf_matrix = calculate_confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['True Negative', 'True Positive'])
    plt.title(f'{model_name} - Confusion Matrix')
    cm_filepath = f"{model_name}_confusion_matrix.png"
    plt.savefig(cm_filepath)
    plt.close()

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} - Receiver Operating Characteristic')
    roc_filepath = f"{model_name}_roc_curve.png"
    plt.savefig(roc_filepath)
    plt.close()

def save_reports(y_true, y_pred, model_name, output_dir='../reports'):
    """
    Generates evaluation reports and visualizations, 
    saves them to the specified output directory, and then creates a ZIP file containing them.

    Parameters:
    - y_true: The true labels
    - y_pred: The predicted labels
    - model_name: The name of the model used for generating the reports
    - output_dir: The directory where the reports and ZIP file will be saved (default is '../reports')

    Returns:
    - None
    """

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate the report and plots
    generate_report(y_true, y_pred, model_name)
    plot_metrics(y_true, y_pred, model_name)
    
    # Define file paths for the report and visualizations
    report_filepath = os.path.join(output_dir, f"{model_name}_report.txt")
    cm_filepath = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
    roc_filepath = os.path.join(output_dir, f"{model_name}_roc_curve.png")
    
    # Add the files to a ZIP archive
    with zipfile.ZipFile(os.path.join(output_dir, f"{model_name}_reports.zip"), 'w') as zipf:
        zipf.write(report_filepath, os.path.basename(report_filepath))
        zipf.write(cm_filepath, os.path.basename(cm_filepath))
