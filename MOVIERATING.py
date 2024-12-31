import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv("C:\\Users\\Vanitha_V\\Documents\\IMDb Movies India.csv", encoding='latin1')
features = ['Name', 'Year', 'Duration', 'Genre', 'Votes', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']
target = 'Rating'

# Analyze missing data
print("Missing values in each column:")
print(data.isna().sum())

# Clean the 'Year' column
data['Year'] = data['Year'].str.extract(r'(\d{4})').astype(float)  # Extract numeric part

# Clean the 'Duration' column
data['Duration'] = data['Duration'].str.replace(r'\D+', '', regex=True).astype(float)  # Remove non-numeric characters

# Replace missing numeric values with the median
numeric_columns = ['Year', 'Duration', 'Votes', 'Rating']
for column in numeric_columns:
    if column in data.columns:
        data[column] = pd.to_numeric(data[column], errors='coerce')  # Ensure numeric
        data[column] = data[column].fillna(data[column].median())
# Replace missing categorical values with a placeholder
categorical_columns = ['Name', 'Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']
for column in categorical_columns:
    if column in data.columns:
        data[column] = data[column].fillna('Unknown')

# Verify missing values are handled
print("Missing values after rectification:")
print(data.isna().sum())

# Define features and target
X = data[features]
y = data[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing for categorical and numerical data
categorical_features = ['Name', 'Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']
numerical_features = ['Year', 'Duration', 'Votes']

categorical_transformer = OneHotEncoder(handle_unknown='ignore')
numerical_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Define the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Create a pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

# Train the model
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
import matplotlib.pyplot as plt
import seaborn as sns

# Visualize the distribution of the target (Rating)
plt.figure(figsize=(8, 6))
sns.histplot(data['Rating'], kde=True, bins=20, color='blue')
plt.title('Distribution of Movie Ratings', fontsize=16)
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")

# Example of predicting a new movie's rating
new_movie = pd.DataFrame({
    'Name': ['A Flat'],
    'Year': [2010],
    'Duration': [103],
    'Genre': ['Drama, Horror, Romance'],
    'Votes': [202],
    'Director': ['Hemant Madhukar'],
    'Actor 1': ['Jimmy Sheirgill'],
    'Actor 2': ['Sanjay Suri'],
    'Actor 3': ['Hazel Croney']
})

predicted_rating = pipeline.predict(new_movie)
print(f"Predicted Rating: {predicted_rating[0]}")

