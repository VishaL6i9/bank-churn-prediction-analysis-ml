import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import joblib

warnings.filterwarnings('ignore')

# Load given dataset 
data = pd.read_csv("bankchurn.csv")
df = data.dropna()
print(df.columns)

# Remove unnecessary columns
df.drop(columns=["RowNumber", "CustomerId", "Surname"], inplace=True)

# Encode categorical variables
var_mod = ['Geography', 'Gender']
le = LabelEncoder()
for i in var_mod:
    df[i] = le.fit_transform(df[i]).astype(int)

# Define features and target variable
X = df.drop(columns=['Exited'])
y = df['Exited']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# Random Forest Classifier Model
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
predictR = rfc.predict(X_test)

print("\nClassification report of Random Forest Classifier Results:\n")
print(classification_report(y_test, predictR))

cm = confusion_matrix(y_test, predictR)
print('\nConfusion Matrix result of Random Forest Classifier is:\n', cm)

# Sensitivity and Specificity Calculation
sensitivity = cm[0,0] / (cm[0,0] + cm[0,1])
specificity = cm[1,1] / (cm[1,0] + cm[1,1])

print('\nSensitivity:', sensitivity)
print('\nSpecificity:', specificity)

# Cross-validation accuracy
accuracy = cross_val_score(rfc, X, y, scoring='accuracy')
print('\nCross validation test results of accuracy:\n', accuracy)
print('\nAccuracy result of Random Forest Classifier is:', accuracy.mean() * 100)
RF = accuracy.mean() * 100

# Accuracy Comparison Graph
def graph():
    plt.figure(figsize=(5,5))
    plt.bar(['Random Forest Classifier'], [RF], color='b')
    plt.title("Accuracy comparison of Bank customer churn", fontsize=15)
    plt.show()

graph()

# Confusion Matrix Breakdown
TP = cm[0,0]
FP = cm[1,0]
FN = cm[1,1]
TN = cm[0,1]

print("True Positive:", TP)
print("True Negative:", TN)
print("False Positive:", FP)
print("False Negative:", FN)

# Rates Calculation
TPR = TP / (TP + FN)
TNR = TN / (TN + FP)
FPR = FP / (FP + TN)
FNR = FN / (TP + FN)

print("\nTrue Positive Rate:", TPR)
print("True Negative Rate:", TNR)
print("False Positive Rate:", FPR)
print("False Negative Rate:", FNR)

# Predictive Values
PPV = TP / (TP + FP)
NPV = TN / (TN + FN)

print("\nPositive Predictive Value:", PPV)
print("Negative Predictive Value:", NPV)

# Confusion Matrix Plot
def plot_confusion_matrix(cm2, title='Confusion matrix - Random Forest Classifier', cmap=plt.cm.Blues):
    plt.figure(figsize=(6,6))
    sns.heatmap(cm2 / np.sum(cm2), annot=True, fmt='.2%', cmap=cmap)
    plt.title(title)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()

cm2 = confusion_matrix(y_test, predictR)
print('\nConfusion matrix - Random Forest Classifier:\n', cm2)
plot_confusion_matrix(cm2)

# Save the model
joblib.dump(rfc, "model.pkl")
