import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load data
data = pd.read_csv("C:\\Users\\Vanitha_V\\Downloads\\Titanic-Dataset.csv")
# Data preprocessing
# Fill missing Age values with the median
data['Age']=data['Age'].fillna(data['Age'].median())

# Fill missing Embarked values with the mode
data['Embarked']=data['Embarked'].fillna(data['Embarked'].mode()[0])

# Drop the Cabin column (too many missing values)
data=data.drop(columns=['Cabin'])

# Convert categorical features to numeric
sex_mapping = {'male': 0, 'female': 1}
data['Sex'] = data['Sex'].map(sex_mapping)

embarked_mapping = {'C': 0, 'Q': 1, 'S': 2}
data['Embarked'] = data['Embarked'].map(embarked_mapping)

# Features and target
X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = data['Survived']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Feature Importance
importances = model.feature_importances_
feature_names = X.columns
feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importances.sort_values(by='Importance', ascending=False, inplace=True)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title('Feature Importance')
plt.show()


# Testing the model with a single test sample
sample_data = pd.DataFrame([[2,'female',14,1,0,30.0708,'C']], columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']) 
sample_data['Sex'] = sample_data['Sex'].map({'male': 0, 'female': 1})
sample_data['Embarked'] = sample_data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
sample_prediction = model.predict(sample_data)
print(f"Sample Prediction: {'Survived' if sample_prediction[0] == 1 else 'Not Survived'}")
