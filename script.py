# %% [markdown]
# # **BUSINESS UNDERSTANDING**

# %% [markdown]
# **Background**

# %% [markdown]
# Breast cancer is one of the most common types of cancer affecting women worldwide. Early detection plays a crucial role in improving survival rates. Machine learning can assist healthcare professionals in diagnosing breast cancer more accurately by distinguishing between benign (non-cancerous) and malignant (cancerous) tumors

# %% [markdown]
# **Problem Statement**

# %% [markdown]
# Despite advancements in technology, misdiagnosis of breast cancer remains a significant challenge. The goal is to develop a predictive model that can classify tumors as benign or malignant based on specific features derived from a breast cancer dataset.

# %% [markdown]
# # Main Obective: #

# %% [markdown]
# To develop a machine learning model that accurately predicts whether a breast tumor is benign or malignant, thereby aiding in early cancer diagnosis and improving patient outcomes.

# %% [markdown]
# # Specific Objectives:

# %% [markdown]
# * To explore and analyze the breast cancer dataset to understand the key features affecting tumor classification.
#
# * To preprocess the data, including handling missing values, normalizing the features, and performing feature selection.
#
# * To build, train, and evaluate machine learning models (e.g., logistic regression, random forest, K-Nearest Neighbors) to classify breast cancer tumors.

# %% [markdown]
# # **1.IMPORTING LIBRARIES AND WARNINGS**
#
#
#

# %%
import os
import joblib
import pickle
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
warnings.filterwarnings('ignore')
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# %% [markdown]
# # 2. Data Understanding

# %% [markdown]
# **2.1 Loading Data**

# %% [markdown]
# We'll load our data and analyse it

# %%
df = pd.read_csv("data/Wisconsin Breast Cancer Data Set.csv")
df

# %%
df[["diagnosis", "radius_mean", "concavity_mean", "smoothness_mean", "texture_mean"]].head(30)

# %%
#Looking at each columns data types
df.info()

# %%
#Summary statistics
df.describe()

# %%
# Checking the number of rows and columns
df.shape

# %%
#checking the columns in the dataset
df.columns

# %%
# Checking the unique diagnosis types,
df["diagnosis"].unique()

# %% [markdown]
# # **3. DATA PREPARATION**

# %% [markdown]
# 3.1 Check if there are any missing values

# %%
# Checking if there are any null values in our dataset
df.isnull().sum()

# %% [markdown]
# 3.2 Handling Missing values

# %% [markdown]
# 3.2.1 Unnamed column

# %%
# Checking the number of null values in the unnamed column
unnamed_null = df["Unnamed: 32"].isnull().sum()
unnamed_null

# %%
# Checking the total number of values in the column
unnamed_total = df["Unnamed: 32"].value_counts().sum()
unnamed_total
# The result means the whole column has null values

# %%
#Dropping the whole column
df.drop("Unnamed: 32", axis=1, inplace=True)

# %%
#checking whether the column has been succesfully dropped.
df.shape
# We had 33 columns and now we have 32, so the column has been succesfully dropped

# %%
df.isnull().sum() # Checking whether there are any more missing values

# %% [markdown]
# **3.3 Checking for any Duplicates**

# %%
df.duplicated().sum()
#There are no duplicates.

# %% [markdown]
#

# %% [markdown]
# 3.4 Checking for any Whitespace in the columns

# %%
df_whitespace = [col for col in df.columns if col.strip() != col]
print("Columns with whitespace:", df_whitespace)

#The result shows there are no whitespaces in  any of the columns

# %% [markdown]
# 3.5 Converting categorical data to Numeric

# %% [markdown]
# We are going to convert the diagnosis column into numerical by mapping malignant to 1 and Benign to 0

# %%
print(df["diagnosis"].unique())
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
df["diagnosis"].unique()



# %% [markdown]
# # **4. EXPLORATORY DATA ANALYSIS(EDA)**

# %% [markdown]
# 4.1 UNIVARIATE

# %% [markdown]
# We are going to visualize the selected variables in order to understand their distributions

# %% [markdown]
# 4.1.1 Diagnosis

# %%
#Plotting the figure
plt.figure(figsize=(10,6))

#countplot
sns.countplot(x="diagnosis", data=df,hue="diagnosis")

#Labels
plt.title("Diagnosis distribution")
plt.xlabel("Diagnosis")
plt.ylabel("count")
plt.legend()

#Display the visual
plt.show()



# %% [markdown]
# The counplot clearly shows that benign counts are greater than Malignant. showing an imbalance

# %%
df.columns

# %% [markdown]
# 4.1.2 Visualizing the radius_mean

# %%
#plotting the figure
plt.figure(figsize=(10,6))

#hist plot
sns.histplot(df["radius_mean"],kde=True,bins=30)

#labels
plt.title("Distribution of radius_mean")
plt.xlabel("radius_mean")
plt.ylabel("count")

#Display the visual
plt.show()


# %% [markdown]
#
# Our data is right skewed,most of the radius_mean range between 10-15 and a couple of them above that

# %% [markdown]
# 4.1.3 Visualizing the area_mean

# %%
plt.figure(figsize=(10,6))
sns.histplot(data=df,x="area_mean",kde=True,bins=30)
plt.title("Distribution of area_mean")
plt.xlabel("area_mean")
plt.ylabel("count")
plt.show()



# %% [markdown]
# Area_mean is also right skewed

# %% [markdown]
# **Checking  skewness of area and radius**

# %% [markdown]
# From the graphs above, radius_mean and area_mean we can clearly see the data is right skewed which can affect how our model predicts

# %%
print("Skewness (area_mean):", df['area_mean'].skew())
print("Skewness (radius_mean):", df['radius_mean'].skew())


# %% [markdown]
# Transforming area_mean since it's heavily skewed compared to radius_mean

# %%
#Applying log transformation to the 'area_mean' column to reduce skewness and handle large variations.
df['area_mean_log'] = np.log(df['area_mean'] + 1) # adding 1 to avoid log(o)

#visualizing using a histplot to see the effect of skewness on our transformed data
sns.histplot(df['area_mean_log'], kde=True)

#labels
plt.title("Log-Transformed area_mean")

#display the visual
plt.show()


# %% [markdown]
# our histplot is now less right skewed

# %% [markdown]
# OUTLIERS ON COMPACTNESS

# %% [markdown]
# compactness refers to how closely packed the cells are within a tumour

# %%
#Plot the figure
plt.figure(figsize=(10,6))

#Create a boxplot to visualize the distribution of compactness mean
sns.boxplot(data=df,x="compactness_mean",orient="h")

#labels
plt.title("Distribution of compactness_mean")
plt.xlabel("compactness_mean")

#display the plot
plt.show()



# %% [markdown]
# according to our boxplot there are several outliers which can signify malignant tumours

# %% [markdown]
# **BIVARIATE**

# %% [markdown]
# * Diagnosis vs radius_mean

# %% [markdown]
# we are going to visualize the relationship between diagnosis and radius mean

# %%
#boxplot
sns.boxplot(data=df,x="diagnosis",y="radius_mean")

#labels
plt.title("Diagnosis vs radius_mean")
plt.xlabel("Diagnosis")
plt.ylabel("radius_mean")

#display the plot
plt.show()



# %% [markdown]
# For Malignant tumours, their cells tend to have higher radius compared to benign tumours

# %% [markdown]
# **MULTIVARIATE**

# %% [markdown]
# **Radius mean vs Area mean**
#
#
#

# %% [markdown]
# We are going to visualize the relationship between radius mean and area mean

# %%
#Scatter plot
sns.scatterplot(data=df,x="radius_mean",y="area_mean",hue="diagnosis")

#labels
plt.title("Radius vs Mean")
plt.xlabel("Radius_mean")
plt.ylabel("Area_mean")
plt.legend()

#Display the plot
plt.show()


# %% [markdown]
# From the above scatterplot, we see that area_mean and radius_mean are correlated
# There is a noticeable seperation between the two classes which implies that these features might be of help while creating our model

# %% [markdown]
# We are going to visualize relationships between various features

# %%
sns.pairplot(df[['radius_mean', 'texture_mean', 'area_mean', 'perimeter_mean', 'diagnosis']], hue='diagnosis')
plt.show()

# %% [markdown]
# From the pairplot above there is a noticeable distinction between the two classes especially along the radius_mean, area_mean and also perimeter mean
#
# There are strong linear relationships between some features such as radius_mean, area_mean and also perimeter_mean which show positive correlations
#
# The texture mean shows a less pronounced seperation betweenbenign and malignant unlike the other features

# %% [markdown]
# **Interaction Between Radius Mean and Concavity Mean**

# %%
#Scatterplot with a regression line
sns.lmplot(x='radius_mean', y='concavity_mean', hue='diagnosis', data=df, aspect=1.5)

#labels
plt.title("Radius Mean vs concavity Mean")

#display the plot
plt.show()


# %% [markdown]
#  Radius mean and concavity mean are positively correlated which means as radius mean increases also the concavity mean tends to increase
# For the malignant tumours, the steep regression line implies that malignant tumors generally have higher concavity as radius increases compared to benign

# %% [markdown]
# **Interaction between texture mean and radius mean**

# %%
#Scatterplot with a regression line
sns.lmplot(data=df,x="radius_mean",y="texture_mean",hue="diagnosis")

#labels
plt.title("Texture mean vs Radius mean")

#Display the plot
plt.show()

# %% [markdown]
# in the above plot, There is an overlap between the 2 classes especially for small radius which implies that texture_mean may alone may not be helpful in distinguishing benign and malignant

# %% [markdown]
# Interaction between smoothness_mean and compactness_mean

# %%
#Lineplot
sns.lmplot(data=df,x="smoothness_mean",y="compactness_mean",hue="diagnosis")

#labels
plt.title("Smoothness mean vs compactness mean")

#display the visual
plt.show()

# %% [markdown]
# positive correlation between the two features
# Malignant tumours tend to have higher values for both compactness and smoothness
# There is also an overlap between the two features which implies using just smoothness and compactness to distinguish the two classes might be challenging hence need to be combined with other features
#

# %%
df["diagnosis"].value_counts()
#There's imbalance in our diagnosis groups we'll handle that during the model selection and training

# %% [markdown]
# **TTEST ON Radius_mean**

# %% [markdown]
# statistically test whether there is a difference between the means of the two groups

# %%
#calculate benign mean
benign = df[df['diagnosis'] == 0]['radius_mean']

#calculate radius mean
malignant = df[df['diagnosis'] == 1]['radius_mean']

#carry out the ttest
t_statistic, p_value = ttest_ind(benign, malignant)

#Print out the values
print("T-statistic:", t_statistic)
print("P-value:", p_value)

#The small p value shows there's a very large difference between the diagnosis groups hence its very unlikely to have occured by chance

# %%
#Visualizing the two groups using a kdeplot
sns.kdeplot(benign, label='Benign', shade=True)
sns.kdeplot(malignant, label='Malignant', shade=True)
plt.title('Density Plot of Radius Mean')
plt.legend()
plt.show()


# %% [markdown]
# above plot confirms that the 2 groups are different
# also gives other implications due to the overlap that at some cases it might be hard to classify the tumours using the radius mean only...especially when the radius is small

# %% [markdown]
# Exploring other features(area mean, compactness mean, concavity mean)

# %%
features = ['area_mean_log', 'compactness_mean', 'concavity_mean',"smoothness_mean","texture_mean"]
for feature in features:
    benign = df[df['diagnosis'] == 0][feature]
    malignant = df[df['diagnosis'] == 1][feature]
    t_statistic, p_value = ttest_ind(benign, malignant)
    print(f"Feature: {feature}")
    print(f"T-statistic: {t_statistic}")
    print(f"P-value: {p_value}")




#The p values are extremely small which mean that these values are not just by random chance theres a significant difference between the two groups

# %%

p_value_dict = {
    'area_mean_log': 5.18e-95,
    'compactness_mean': 3.94e-56,
    'concavity_mean': 9.97e-84,
    'smoothness_mean': 1.05e-18,
    'texture_mean': 4.06e-25
}


plt.figure(figsize=(8, 6))
plt.bar(p_value_dict.keys(), p_value_dict.values(), color='skyblue')
plt.axhline(y=0.05, color='red', linestyle='--', label='Significance Threshold (0.05)')  # Red line for significance
plt.xlabel('Features')
plt.ylabel('P-Value')
plt.yscale('log')  # Log scale to visualize small p-values better
plt.title('P-Values from T-Tests (Log Scale)')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.legend()

plt.show()

# %% [markdown]
# # FEATURE SELECTION

# %%
#Dictionary storing the p values of various features
p_values_dict = {
     'radius_mean': 8.47e-96,
    'area_mean_log': 5.18e-95,
    'compactness_mean': 3.94e-56,
    'concavity_mean': 9.97e-84,
    'smoothness_mean': 1.05e-18,
    'texture_mean': 4.06e-25

}
#defining the significance level threshold
threshold = 0.05

#selecting features whose p value is less that the threshold
selected_features = [feature for feature, p_value in p_values_dict.items() if p_value < threshold]

print("Selected Features:", selected_features)

#All have a p value thats less than the set threshold which means they can be used for modelling

# %%
#Calculating the correlation matrix of the selected features
correlation_matrix = df[selected_features].corr()

#visualizing the correlations using a heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')

#Labels
plt.title('Correlation Heatmap')

#display the visual
plt.show()

#area_mean_log and radius mean are highly correlated which may lead to multicollinearity..redundant info. so we're going to drop on from our selected features
#also compactness and concavity..


# %% [markdown]
# area_mean_log and radius mean are highly correlated which may lead to multicollinearity..redundant info. so we're going to drop on from our selected features
# also compactness and concavity..

# %% [markdown]
# Defining which features to select

# %%
final_features = ['radius_mean', 'concavity_mean', 'smoothness_mean', 'texture_mean']

# %% [markdown]
# # TRAINING OUR MODEL
#

# %%
#Define the features x
x = df[final_features]

#Define the target variable
y = df["diagnosis"]

#Split the dataset into training and testing sets where 80% is used to train and 205 to test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#We are going to use SMOTE to balance our dataset, remember we saw that the counts of benign were a bit higher than those of malignant.
smote = SMOTE(random_state=42)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

# %%
# Creating a directory to save models
os.makedirs("models", exist_ok=True)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(random_state=42)
}

# Looping through the models
for name, model in models.items():
    print(f"\n{name}")  # Printing the name of the model

    # Fitting the model to the training data
    model.fit(x_train_resampled, y_train_resampled)

    # Predicting the target variable on x_test
    y_pred = model.predict(x_test)

    # Print the accuracy score, classification report, and confusion matrix
    print("Accuracy Score:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Perform cross-validation and print mean accuracy
    cv_scores = cross_val_score(model, x_train_resampled, y_train_resampled, cv=5, scoring='accuracy')
    print(f"\nCross-Validation Scores: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")

    # Save the trained model
    model_filename = f"models/{name.replace(' ', '_')}.pkl"
    joblib.dump(model, model_filename)
    print(f"Model saved as {model_filename}")

# %% [markdown]
# # EVALUATION

# %% [markdown]
# 3 models were evaluated on their ability to classify the breast tumours as benign or malignant based on various features from the dataset
# According to their perfomance, we decided to choose Random Forest classifier, reasons:
#
#
# *   It has a strong perfomance in terms of accuracy, recall and also precision
# *   High cross validation which indicates that the model will work well on unseen data
# *   It is less affected by outliers
#
#
#
#
#
#
#
#
#
#
#

# %% [markdown]
# # CONCLUSIONS

# %% [markdown]
#
#
# *   Malignant tumors have higher radius and area values which aligns with their nature and faster growth
# *   Some features such as compactness mean had outliers which may signify aggressive tumors or unique cases that require further investigation
# *  Applyong SMOTE addressed the class imbalance in the dataset ensuring that both benign and malignant cases were adequately represented in the training data
# *  Scaling the features helped improve model perfomance by ensuring that features like KNN were not affected
#
#

# %% [markdown]
# # RECOMMENDATIONS


