# Capstone Project

## Overview
This project analyzes a dataset of Spotify songs, focusing on various musical features and their relationships with song popularity. It employs multiple statistical methods and machine learning techniques to uncover insights and predict song popularity. 

## Project Structure

### 1. **Data Loading and Preparation**
   - The dataset is loaded from `spotify52kData.csv`.
   - The dataset contains features such as `duration`, `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, and `tempo`.
   - Initial data cleaning is performed to handle any missing values and outliers.

### 2. **Exploratory Data Analysis**
   - **Histograms with Shapiro-Wilk Test**:
     - Histograms of the features are plotted.
     - A Shapiro-Wilk test is performed to assess the normality of each feature's distribution.
   - **Correlation Analysis**:
     - The correlation between `duration` and `popularity` is calculated and visualized using a scatter plot.

### 3. **Statistical Hypothesis Testing**
   - **Mann-Whitney U Test**:
     - The popularity of explicit vs. non-explicit songs is compared.
     - The popularity of songs in major vs. minor keys is compared.
   - **ANOVA**:
     - The relationship between a song's key and its popularity is examined using Analysis of Variance (ANOVA).

### 4. **Machine Learning Models**
   - **Linear Regression**:
     - Linear regression models are built to predict song popularity based on individual features and a combination of all features.
   - **Principal Component Analysis (PCA)**:
     - PCA is used to reduce the dimensionality of the feature space.
     - The number of meaningful principal components is determined based on the explained variance.
   - **K-Means Clustering**:
     - K-Means clustering is applied to the principal components to identify clusters within the data.
     - The optimal number of clusters is determined using the Elbow Method and Silhouette Scores.
   - **Logistic Regression**:
     - Logistic regression is performed to predict the mode (major or minor key) of a song based on various features.
     - The accuracy and AUC of the logistic regression models are compared across different predictor variables.
   - **Decision Tree Classifier**:
     - A decision tree classifier is built to predict the genre of a song based on the selected features.

### 5. **Visualization**
   - Various visualizations including histograms, scatter plots, bar charts, and decision trees are generated to illustrate the results of the analysis.

## Files
- **Capstone Project code.py**: The main Python script that contains all the code for data analysis, statistical testing, and model building.
- **Capstone Project.pdf**: The detailed report explaining the methods, results, and interpretations of the analyses performed in this project.

## How to Run
1. Ensure all required libraries are installed: pandas, numpy, matplotlib, seaborn, scipy, sklearn, statsmodels.
2. Place the `spotify52kData.csv` file in the same directory as the Python script.
3. Run the `Capstone Project code.py` script in your Python environment.

## Results Summary
- No feature was found to have a normal distribution according to the Shapiro-Wilk test.
- A very weak negative correlation was observed between song duration and popularity.
- Significant differences in popularity were found between explicit vs. non-explicit songs and between songs in major vs. minor keys.
- Linear regression models showed that `instrumentalness` was the best individual predictor of popularity, although the overall predictive power was low.
- PCA revealed that 6 principal components explain about 84% of the variance.
- K-Means clustering suggested 2 optimal clusters.
- Logistic regression models showed that `valence` was the best predictor for the mode, with an accuracy of 62%.
- The decision tree classifier did not achieve high accuracy in predicting song genres, indicating the need for additional features.
