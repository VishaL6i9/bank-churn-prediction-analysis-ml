import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings

warnings.filterwarnings('ignore')

data = pd.read_csv("bankchurn.csv")
df = data.dropna()
print(df.columns)

print(pd.crosstab(df.Gender, df.Exited))
print(pd.crosstab(df.Balance, df.Exited))
print(pd.crosstab(df.IsActiveMember, df.Exited))

# Histogram Plot of Customer Balance 
plt.figure()
plt.hist(df['Balance'], bins=30, edgecolor='k')
plt.title("Balance of the Customers")
plt.xlabel("Balance")
plt.ylabel("No of Customers")
plt.show()

# Histogram Plot of Credit Score 
plt.figure()
plt.hist(df['CreditScore'], bins=30, edgecolor='k')
plt.title("Credit Score Distribution")
plt.xlabel("Credit Score")
plt.ylabel("No of Customers")
plt.show()

# Function to plot categorical variable distribution
def PropByVar(df, variable):
    dataframe_pie = df[variable].value_counts()
    ax = dataframe_pie.plot.pie(figsize=(10,10), autopct='%1.2f%%', fontsize=12)
    ax.set_title(variable + ' Distribution \n', fontsize=15)
    plt.ylabel('')  # Hide default ylabel
    plt.show()
    return np.round(dataframe_pie / df.shape[0] * 100, 2)

# Example usage of PropByVar
print(PropByVar(df, 'Geography'))

# Boxplot for Balance
fig, ax = plt.subplots(figsize=(15,6))
sns.boxplot(x=df['Balance'], ax=ax)
plt.title("Balance")
plt.show()

# Pairplot
sns.pairplot(df)
plt.show()

# Heatmap plot diagram 
fig, ax = plt.subplots(figsize=(15,10))
sns.heatmap(df.corr(), ax=ax, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.show()

# Preprocessing, split test and train dataset
X = df.drop(columns=['Exited'])
y = df['Exited']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Number of training dataset: ", len(X_train))
print("Number of test dataset: ", len(X_test))

# Function to create bar plot
def qul_No_qul_bar_plot(df, bygroup):
    dataframe_by_Group = pd.crosstab(df[bygroup], df["Exited"], normalize='index')
    dataframe_by_Group = np.round((dataframe_by_Group * 100), decimals=2)
    ax = dataframe_by_Group.plot.bar(figsize=(15,7))

    vals = ax.get_yticks()
    ax.set_yticklabels(['{:3.0f}%'.format(x) for x in vals])
    ax.set_xticklabels(dataframe_by_Group.index, rotation=0, fontsize=15)
    ax.set_title(f'Bank Customer Exit Status by {bygroup} (%)\n', fontsize=15)
    ax.set_xlabel(bygroup, fontsize=12)
    ax.set_ylabel("(%)", fontsize=12)
    ax.legend(title="Exited", loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=12)

    for rect in ax.patches:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height + 2, f'{height:.1f}%', ha='center', va='bottom', fontsize=12)

    plt.show()
    return dataframe_by_Group

# Example usage of qul_No_qul_bar_plot
print(qul_No_qul_bar_plot(df, 'IsActiveMember'))