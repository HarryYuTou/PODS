#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import silhouette_score

import random
RNG = 13994237
random.seed(RNG)  



# Load the dataset
file_path = 'spotify52kData.csv'
spotify_data = pd.read_csv(file_path)
sample_data = spotify_data.head()
sample_data



# 1.
# Selecting the 10 song features for analysis
features = ['duration', 'danceability', 'energy', 'loudness', 'speechiness', 
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

# Defining the plotting function to include p-values from the Shapiro-Wilk test
def plot_with_p_values(data, features, sample_size=5000):
    plt.figure(figsize=(20, 10))

    # Dictionary to store p-values
    p_values = {}

    # Calculating p-values using Shapiro-Wilk test for each feature
    for feature in features:
        _, p_value = stats.shapiro(data[feature].sample(sample_size))
        p_values[feature] = p_value

    # Plotting histograms with p-values
    for i, feature in enumerate(features, 1):
        plt.subplot(2, 5, i)
        sns.histplot(data[feature], kde=True, bins=30)
        plt.title(f"{feature}\nSW p-value: {p_values[feature]:.2e}")  # Displaying p-value in scientific notation

    plt.tight_layout()
    plt.show()

# Plotting the histograms with p-values
plot_with_p_values(spotify_data, features)




# 2.
# Selecting variables for the analysis: 'duration' and 'popularity'
data_for_analysis = spotify_data[['duration', 'popularity']]

# Data Cleaning: Checking for any missing values or outliers in the selected variables
missing_values = data_for_analysis.isnull().sum()
outliers = data_for_analysis.describe()

# Calculating the correlation coefficient between 'duration' and 'popularity'
correlation = data_for_analysis.corr().loc['duration', 'popularity']

# Plotting the scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='duration', y='popularity', data=data_for_analysis)
plt.title(f"Song Duration vs Popularity\nCorrelation: {correlation:.2f}")
plt.xlabel("Duration (ms)")
plt.ylabel("Popularity")
plt.show()




# 3.
# Selecting variables: 'explicit' and 'popularity'
explicit_data = spotify_data[['explicit', 'popularity']]

# Splitting the data into two groups: Explicit and Non-Explicit
explicit_songs = explicit_data[explicit_data['explicit'] == True]['popularity']
non_explicit_songs = explicit_data[explicit_data['explicit'] == False]['popularity']

# Mann-Whitney U test: Non-parametric test used as the data might not be normally distributed
u_statistic, p_value = mannwhitneyu(explicit_songs, non_explicit_songs)

# Preparing the results for display
test_results = {
    "Test Method": "Mann-Whitney U",
    "Null Hypothesis": "No difference in popularity between explicit and non-explicit songs",
    "Alternative Hypothesis": "Difference in popularity between explicit and non-explicit songs",
    "Significance Level": 0.05,
    "Test Statistic (U)": u_statistic,
    "p-Value": p_value
}
print(test_results)




# Plotting the histogram plot
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
sns.histplot(x='popularity', data=explicit_data[explicit_data['explicit'] == True])
plt.title(f"Histogram of popularity (explicit = True)")
plt.xlabel("Popularity")
plt.ylabel("Frequency")
plt.subplot(2, 1, 2)
sns.histplot(x='popularity', data=explicit_data[explicit_data['explicit'] == False])
plt.title(f"Histogram of popularity (explicit = False)")
plt.xlabel("Popularity")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()




# 4.
# Selecting variables: 'mode' (1 for major, 0 for minor) and 'popularity'
mode_data = spotify_data[['mode', 'popularity']]

# Splitting the data into two groups: Major Key and Minor Key
major_key_songs = mode_data[mode_data['mode'] == 1]['popularity']
minor_key_songs = mode_data[mode_data['mode'] == 0]['popularity']

# Mann-Whitney U test: Non-parametric test used as the data might not be normally distributed
u_statistic_mode, p_value_mode = mannwhitneyu(major_key_songs, minor_key_songs)

# Preparing the results for display
test_results_mode = {
    "Test Method": "Mann-Whitney U",
    "Null Hypothesis": "No difference in popularity between songs in major key and minor key",
    "Alternative Hypothesis": "Difference in popularity between songs in major key and minor key",
    "Significance Level": 0.05,
    "Test Statistic (U)": u_statistic_mode,
    "p-Value": p_value_mode
}

print(test_results_mode)




# Plotting the histogram plot
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
sns.histplot(x='popularity', data=mode_data[mode_data['mode'] == 1])
plt.title(f"Histogram of popularity (mode = 1)")
plt.xlabel("Popularity")
plt.ylabel("Frequency")
plt.subplot(2, 1, 2)
sns.histplot(x='popularity', data=mode_data[mode_data['mode'] == 0])
plt.title(f"Histogram of popularity (mode = 0)")
plt.xlabel("Popularity")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()



# 5.
# Selecting variables for analysis: 'energy' and 'loudness'
energy_loudness_data = spotify_data[['energy', 'loudness']]


# Calculating the correlation coefficient between 'energy' and 'loudness'
correlation_energy_loudness = energy_loudness_data.corr().loc['energy', 'loudness']

# Plotting the scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='energy', y='loudness', data=energy_loudness_data)
plt.title(f"Energy vs Loudness\nCorrelation: {correlation_energy_loudness:.2f}")
plt.xlabel("Energy")
plt.ylabel("Loudness (dB)")
plt.show()



# 6.
# Selecting the features and the target variable 'popularity'
features_for_prediction = ['duration', 'danceability', 'energy', 'loudness', 
                           'speechiness', 'acousticness', 'instrumentalness', 
                           'liveness', 'valence', 'tempo']

# Create an empty dictionary for storing R^2 values
r2_values = {}

for feature in features_for_prediction:
    x = spotify_data[feature].values.reshape(-1,1)
    y = spotify_data['popularity'].values
    
    # Splitting the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state = RNG)
    
    # Fit a linear regression model
    model = LinearRegression()
    model.fit(x_train, y_train)
    
    # Predicting on the test set
    y_pred = model.predict(x_test)
    
    # Calculating the R^2 values
    r2 = r2_score(y_test, y_pred)
    r2_values[feature] = r2
print(r2_values)

#Plot the results
features_plot = list(r2_values.keys())
r2_plot = list(r2_values.values())
plt.figure(figsize=(10,6))
plt.barh(features_plot,r2_plot)
plt.xlabel('R^2 value')
plt.ylabel('Features')
plt.title('R^2 values for Features predicting Popularity')
plt.show()
    
    
# 7.
# Plotting the histogram plot   
X = spotify_data[features_for_prediction]
y = spotify_data['popularity']

# Create an empty dictionary for storing R^2 values
r2_values2 = {}

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = RNG)

# Creating and training the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predicting on the test set
y_pred = model.predict(X_test)

# Calculating R2 score
r2_all = r2_score(y_test, y_pred)

# Fill in the values for comparison
r2_values2['All Features'] = r2_all
r2_values2['Instrumentalness'] = r2_values['instrumentalness']
print(r2_values2)

# Plot the results 
features_plot2 = list(r2_values2.keys())
r2_plot2 = list(r2_values2.values())
plt.figure(figsize=(10,6))
plt.barh(features_plot2,r2_plot2)
plt.xlabel('R^2 value')
plt.title('R^2 values for Instrumentalness predicting Popularity VS All Features predicting Popularity')
plt.show()



# 8.
# Selecting the 10 song features for PCA
X_pca = spotify_data[features_for_prediction]

# Data Cleaning: Standardizing the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pca)

# Principal Component Analysis (PCA)
pca = PCA(n_components=len(features_for_prediction))
X_pca_transformed = pca.fit_transform(X_scaled)

# Proportion of Variance Explained by each Principal Component
variance_explained = pca.explained_variance_ratio_

# Cumulative Variance Explained
cumulative_variance = np.cumsum(variance_explained)

# Determining the number of meaningful principal components
# Rule of thumb: look for components that add up to a cumulative variance of about 0.7-0.8
n_components_meaningful = np.argmax(cumulative_variance >= 0.8) + 1  # +1 because index starts at 0

# Plotting the explained variance and cumulative variance
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(variance_explained) + 1), variance_explained, alpha=0.5, label='Individual explained variance')
plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', label='Cumulative explained variance')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.legend(loc='best')
plt.title('Explained Variance by Principal Components')
plt.show()



# KMeans Clustering using the principal components
# Using the Elbow Method to determine the number of clusters
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=RNG)
    kmeans.fit(X_pca_transformed[:, :n_components_meaningful])
    sse.append(kmeans.inertia_)

# Plotting the Elbow Method result
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), sse, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.title('Elbow Method for Determining Optimal Number of Clusters')
plt.show()

# Results of PCA and clustering
pca_results = {
    "Number of Meaningful Components": n_components_meaningful,
    "Proportion of Variance Explained": cumulative_variance[n_components_meaningful - 1]
}

print(pca_results, variance_explained, cumulative_variance)



# Using Silhouette Score to determine the number of clusters
silhouette_scores = []
for k in range(2, 11):  # Starts from 2 as silhouette score can't be computed with a single cluster
    kmeans = KMeans(n_clusters=k, random_state=RNG)
    kmeans.fit(X_pca_transformed[:, :n_components_meaningful])
    score = silhouette_score(X_pca_transformed[:, :n_components_meaningful], kmeans.labels_)
    silhouette_scores.append(score)
    
# Plotting the Silhouette Scores result
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Determining Optimal Number of Clusters')
plt.show()




# Selecting the optimal number of clusters
optimal_clusters = range(2, 11)[np.argmax(silhouette_scores)] + 1

# Results of PCA and clustering with silhouette score
pca_results_with_silhouette = {
    "Number of Meaningful Components": n_components_meaningful,
    "Proportion of Variance Explained": cumulative_variance[n_components_meaningful - 1],
    "Optimal Number of Clusters": optimal_clusters
}

print(pca_results_with_silhouette)



# 9.
# Selecting variables: 'valence' as predictor and 'mode' as target
X_valence = spotify_data[['valence']]
y_mode = spotify_data['mode']

# Splitting the data into training and testing sets
X_train_valence, X_test_valence, y_train_mode, y_test_mode = train_test_split(X_valence, y_mode, test_size=0.3, random_state=RNG)

# Logistic Regression Model
logreg = LogisticRegression()
logreg.fit(X_train_valence, y_train_mode)

# Predicting on the test set
y_pred_mode = logreg.predict(X_test_valence)

y_pred_proba_mode = logreg.predict_proba(X_test_valence)[:, 1]  # probabilities for the positive class


# Evaluating the model
auc_score = roc_auc_score(y_test_mode, y_pred_proba_mode)
accuracy = accuracy_score(y_test_mode, y_pred_mode)
classification_rep = classification_report(y_test_mode, y_pred_mode)

# Checking if there's a better predictor among the other features
other_features = ['duration', 'danceability', 'energy', 'loudness', 
                  'speechiness', 'acousticness', 'instrumentalness', 
                  'liveness', 'tempo', 'popularity']

best_feature = None
best_accuracy = accuracy
best_feature_auc = None
best_auc = auc_score

for feature in other_features:
    X_feature = spotify_data[[feature]]
    X_train_feature, X_test_feature, y_train_feature, y_test_feature = train_test_split(X_feature, y_mode, test_size=0.3, random_state=RNG)

    logreg_feature = LogisticRegression()
    logreg_feature.fit(X_train_feature, y_train_feature)
    y_pred_feature = logreg_feature.predict(X_test_feature)
    y_pred_proba_feature = logreg_feature.predict_proba(X_test_feature)[:, 1]

    feature_accuracy = accuracy_score(y_test_feature, y_pred_feature)
    feature_auc = roc_auc_score(y_test_feature, y_pred_proba_feature)

    if feature_accuracy > best_accuracy:
        best_accuracy = feature_accuracy
        best_feature = feature
    
    if feature_auc > best_auc:
        best_auc = feature_auc
        best_feature_auc = feature

# Results
model_results = {
    "Predictor": "valence",
    "Accuracy": accuracy,
    "AUC": auc_score,
    "Best Predictor": best_feature if best_feature else "valence",
    "Best Accuracy": best_accuracy,
    "Best Predictor (AUC)": best_feature_auc if best_feature_auc else "valence",
    "Best AUC": best_auc
}

print(model_results)




y_pred_proba_mode = logreg.predict_proba(X_test_valence)[:, 1]  # probabilities for the positive class
auc_score = roc_auc_score(y_test_mode, y_pred_proba_mode)

# Generating and plotting confusion matrix for the training set
y_train_pred = logreg.predict(X_train_valence)
cm_train = confusion_matrix(y_train_mode, y_train_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_train, annot=True, fmt="d")
plt.title("Confusion Matrix (Training Set)")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()



# Generating and plotting confusion matrix for the test set
cm_test = confusion_matrix(y_test_mode, y_pred_mode)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_test, annot=True, fmt="d")
plt.title("Confusion Matrix (Test Set)")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()




# 10.
# Selecting variables: 10 song features and 'track_genre' as the target
X_genre = spotify_data[features_for_prediction]
y_genre = spotify_data['track_genre']

# Encoding the 'track_genre' labels to numerical labels
le = LabelEncoder()
y_genre_encoded = le.fit_transform(y_genre)

# Splitting the data into training and testing sets
X_train_genre, X_test_genre, y_train_genre, y_test_genre = train_test_split(X_genre, y_genre_encoded, test_size=0.3, random_state=42)

# Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train_genre, y_train_genre)

# Evaluating the model
accuracy_genre = dt_classifier.score(X_test_genre, y_test_genre)

# Plotting the decision tree
plt.figure(figsize=(12, 8))
plot_tree(dt_classifier, max_depth=2, feature_names=features_for_prediction, class_names=le.classes_, filled=True, fontsize=9)
plt.show()

# Results
genre_prediction_results = {
    "Accuracy": accuracy_genre,
    "Model": "Decision Tree Classifier"
}

print(genre_prediction_results)



# Extra credit
from scipy.stats import f_oneway
import statsmodels.api as sm
from statsmodels.formula.api import ols


# ANOVA for RQ1: Popularity by Key
anova_data = spotify_data[['key', 'popularity']]
anova_results = ols('popularity ~ C(key)', data=anova_data).fit()
anova_table = sm.stats.anova_lm(anova_results, typ=2)
print(anova_table)

# Visualizing the relationship between song key and popularity (RQ1)
plt.figure(figsize=(12, 6))
sns.boxplot(x='key', y='popularity', data=spotify_data)
plt.title('Popularity by Musical Key')
plt.xlabel('Key')
plt.ylabel('Popularity')
plt.show()


# Linear Regression for RQ2: Time Signature and Danceability
regression_data = spotify_data[['time_signature', 'danceability']]
model = sm.OLS(regression_data['danceability'], sm.add_constant(regression_data['time_signature'])).fit()
regression_summary = model.summary()
print(regression_summary)


# Visualizing the relationship between time signature and danceability (RQ2)
plt.figure(figsize=(12, 6))
sns.scatterplot(x='time_signature', y='danceability', data=spotify_data)
plt.title('Danceability by Time Signature')
plt.xlabel('Time Signature')
plt.ylabel('Danceability')
plt.show()









