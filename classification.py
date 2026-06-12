from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

print("=== AI Classification Project ===")
print("Dataset Used: Iris Dataset")

# Loading the iris flower dataset
iris = load_iris()

# Features and target values
X = iris.data
y = iris.target

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Creating the Decision Tree model
model = DecisionTreeClassifier()

# Training the model
model.fit(X_train, y_train)

# Making predictions on test data
predictions = model.predict(X_test)

# Evaluating model performance
accuracy = accuracy_score(y_test, predictions)

print("\nProject Completed Successfully!")
print("Number of Training Samples:", len(X_train))
print("Number of Testing Samples:", len(X_test))
print("Model Accuracy:", round(accuracy * 100, 2), "%")