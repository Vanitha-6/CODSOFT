#importing libraries

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd

#load the dataset

iris = datasets.load_iris()
X = iris.data 
y = iris.target  
target_names = ['Iris-setosa','Iris-versicolor','Iris-virginica']

#scatter plot

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
plt.figure(figsize=(10, 6))
plt.scatter(df['sepal length (cm)'], df['sepal width (cm)'], c=df['species'].cat.codes, cmap='viridis', edgecolors='k', s=100)
plt.title('Sepal Length vs Sepal Width')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.colorbar(label='Species')
plt.show()

#KNN classifier model training

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

#accuracy prediction

accuracy = accuracy_score(y_test, y_pred)

confusion = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
for row, species in zip(confusion, target_names):
    print(f"{species}: {row}")

print("\nPredicted Values   Actual Values")
for pred, actual in zip(y_pred, y_test):
    print(f"{pred}                {actual}")

print(f"\nAccuracy of the K-Nearest Neighbors classifier: {accuracy * 100:.2f}%")
confusion = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
plt.imshow(confusion, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(target_names))
plt.xticks(tick_marks, target_names, rotation=45)
plt.yticks(tick_marks, target_names)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
for i in range(len(target_names)):
    for j in range(len(target_names)):
        plt.text(j, i, f'{confusion[i, j]}', ha='center', va='center', color='white')
plt.show()

#testing the model

print(classification_report(y_test, y_pred))
test_sample = np.array([[6,2,4,1]]) 
prediction = knn.predict(test_sample)
species_names = iris.target_names  # ['setosa', 'versicolor', 'virginica']
predicted_species = species_names[prediction]

print(f"Predicted class: {predicted_species[0]}")


