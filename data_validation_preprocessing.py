import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# Load the given dataset
data = pd.read_csv("bankchurn.csv")

# Before dropping NaN values
print("Before dropping NaN values:")
print(data.head(10))
print("Shape:", data.shape)

# After dropping NaN values
df = data.dropna()
print("After dropping NaN values:")
print(df.head())
print("Shape:", df.shape)

# Checking data types and dataset information
print("\nDataset Information:")
print(df.info())

# Unique values in various columns
print("\nUnique values in columns:")
print("Age:", df.Age.unique())
print("IsActiveMember:", df.IsActiveMember.unique())
print("Gender:", df.Gender.unique())
print("Geography:", df.Geography.unique())
print("Surname:", df.Surname.unique())
print("HasCrCard:", df.HasCrCard.unique())
print("NumOfProducts:", df.NumOfProducts.unique())
print("Exited:", df.Exited.unique())

# Correlation matrix
print("\nCorrelation Matrix:")
print(df.corr())

# Before Pre-Processing
print("\nBefore Pre-Processing:")
print(df.head())

# Dropping unnecessary columns
df1 = df.drop(['RowNumber', 'Surname'], axis=1)
print("\nAfter Pre-Processing:")
print(df1.head())

# Encoding categorical variables
from sklearn.preprocessing import LabelEncoder

var_mod = ['Geography', 'Gender']
le = LabelEncoder()
for i in var_mod:
    df1[i] = le.fit_transform(df1[i]).astype(int)

print("\nData after Label Encoding:")
print(df1.head())
