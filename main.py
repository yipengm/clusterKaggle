import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


file_path = './data/data.csv'
content = pd.read_csv(file_path)
content.info()
content.describe()
content.isnull().sum()
feature = content.drop(columns="ID")
featureSelected = pd.DataFrame(content, columns=["RRmean","PTdis","STdis"])
df_new = pd.DataFrame(content, columns=["ID"])
# Step 1: Select only numerical columns (int64, float64)
numerical_df = feature.select_dtypes(include=['float64', 'int64'])

# Step 2: Calculate the z-scores for the numerical columns
z_scores = zscore(numerical_df)

mask = np.abs(z_scores) < 3  # Absolute value to capture both positive and negative outliers
# Step 4: Apply the mask to the original dataframe to filter out outliers
# We use .all(axis=1) to make sure all columns in a row pass the condition

# Step 2: Identify columns with outliers
outlier_columns = []

# Check if any value in a column has a z-score greater than the threshold (e.g., 3)
for i, column in enumerate(numerical_df.columns):
    if np.any(mask[:, i] == False):  # threshold = 3 for outliers
        outlier_columns.append(column)



#df_filtered = feature.drop(columns=outlier_columns)
df_filtered = featureSelected
# Step 5: Update the original dataframe with the filtered data

corr_matrix = df_filtered.corr()
high_corr_var = []  # Initialize an empty list to store high correlation columns
for column in corr_matrix.columns:
    # Check if there is any correlation greater than 0.9 in the column
    for other_column in corr_matrix.columns:
        if column != other_column and corr_matrix[column][other_column] > 0.9:
            high_corr_var.append(column)

#df_filtered = df_filtered.drop(columns=high_corr_var)


df_numerical = df_filtered.select_dtypes(include=['float64', 'int64'])

means = df_numerical.mean()
stds = df_numerical.std()

df_standardized = (df_numerical - means) / stds
df_filtered[df_standardized.columns] = df_standardized
cov_matrix = df_standardized.cov()

# Step 3: Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Step 4: Sort eigenvalues in descending order and get their corresponding eigenvectors
sorted_indices = np.argsort(eigenvalues)[::-1]  # Sort in descending order
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

# Step 5: Display eigenvalues
print("Eigenvalues (variance explained by each component):")
print(sorted_eigenvalues)

pca = PCA(n_components=2)
feature_pca = pd.DataFrame(pca.fit_transform(df_filtered))



kmeans = KMeans(n_clusters=3, random_state=42)  # Specify the number of clusters (n_clusters)
kmeans.fit(feature_pca)
df_new['Category'] = kmeans.labels_

df_new.to_csv('outputStandardFeature3PCA3.csv', index=False)
print("o")