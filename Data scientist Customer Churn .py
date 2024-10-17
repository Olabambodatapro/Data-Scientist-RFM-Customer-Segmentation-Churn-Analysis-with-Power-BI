#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


# Load the dataset (assuming it's in the same directory as the Jupyter notebook)
file_path = 'Online_retail_data.xlsx'
retail_data = pd.read_excel(file_path)

# Preview the first few rows of the dataset
retail_data.head()


# In[8]:


# Remove duplicate rows, if any, and make an explicit copy of the cleaned data
retail_data_cleaned = retail_data.drop_duplicates().copy()

# Convert 'InvoiceDate' to datetime using .loc[] (safe operation)
retail_data_cleaned['InvoiceDate'] = pd.to_datetime(retail_data_cleaned['InvoiceDate'])

# Create a 'TotalPrice' column (Quantity * UnitPrice) using .loc[] (safe operation)
retail_data_cleaned['TotalPrice'] = retail_data_cleaned['Quantity'] * retail_data_cleaned['UnitPrice']

# Create a reference date for Recency (e.g., the last day in the dataset)
latest_date = retail_data_cleaned['InvoiceDate'].max()

# Calculate Recency, Frequency, and Monetary values
rfm = retail_data_cleaned.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (latest_date - x.max()).days,  # Recency
    'InvoiceNo': 'nunique',  # Frequency (number of unique invoices)
    'TotalPrice': 'sum'  # Monetary (total amount spent)
})

# Rename the columns
rfm.columns = ['Recency', 'Frequency', 'Monetary']

# Display the RFM table
print(rfm.head())

# Define churn based on recency (e.g., if Recency > 180 days, customer is churned)
churn_threshold = 180  # You can adjust this threshold
rfm['Churned'] = rfm['Recency'].apply(lambda x: 1 if x > churn_threshold else 0)

# Display the first few rows with the Churn label
print(rfm[['Recency', 'Churned']].head())



# In[9]:


#Data Normalization
from sklearn.preprocessing import MinMaxScaler

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Normalize Recency, Frequency, and Monetary columns
rfm[['Recency', 'Frequency', 'Monetary']] = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# Check the scaled RFM table
rfm.head()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Split the data into features (RFM) and target (Churned)
X = rfm[['Recency', 'Frequency', 'Monetary']]
y = rfm['Churned']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train a Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))


# In[10]:


# Check feature importance in the Random Forest model
importances = rf_model.feature_importances_
features = ['Recency', 'Frequency', 'Monetary']

# Display feature importance
for feature, importance in zip(features, importances):
    print(f'{feature}: {importance:.4f}')


# In[14]:


#Plotting the Distribution of Recency, Frequency, and Monetary
#Recency, Frequency, and Monetary Distributions: Uses histograms with kernel density estimation (KDE) to visualize the distribution of each RFM metric.
# Set the style for Seaborn plots
sns.set(style='whitegrid')

# Create a figure with subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Recency distribution
sns.histplot(rfm['Recency'], bins=30, kde=True, color='blue', ax=axes[0])
axes[0].set_title('Recency Distribution')
axes[0].set_xlabel('Recency (days)')
axes[0].set_ylabel('Frequency')

# Frequency distribution
sns.histplot(rfm['Frequency'], bins=30, kde=True, color='green', ax=axes[1])
axes[1].set_title('Frequency Distribution')
axes[1].set_xlabel('Frequency (Number of Purchases)')
axes[1].set_ylabel('Frequency')

# Monetary distribution
sns.histplot(rfm['Monetary'], bins=30, kde=True, color='orange', ax=axes[2])
axes[2].set_title('Monetary Distribution')
axes[2].set_xlabel('Monetary (Total Spending)')
axes[2].set_ylabel('Frequency')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()


# In[15]:


#Visualizing the Churn Distribution
#Churn Distribution: A simple count plot shows how many customers have churned and how many haven't, based on the threshold we defined earlier.
# Plot the churn distribution using a bar plot
plt.figure(figsize=(8, 5))
sns.countplot(x='Churned', data=rfm, palette='Set2')
plt.title('Churn Distribution')
plt.xlabel('Churn Status')
plt.ylabel('Customer Count')
plt.xticks([0, 1], ['Not Churned', 'Churned'])
plt.show()


# In[ ]:





# In[ ]:




