import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Create a DataFrame with the given data
data = {
    'x1': ['T', 'I', 'T', 'I', 'T', 'I', 'T', 'I', 'T', 'I'],
    'x2': ['M', 'M', 'P', 'P', 'P', 'M', 'M', 'P', 'P', 'P'],
    'x3': ['S', 'S', 'S', 'C', 'C', 'C', 'S', 'S', 'C', 'C'],
    'y': ['W', 'W', 'W', 'W', 'L', 'L', 'L', 'L', 'L', 'L']
}

df = pd.DataFrame(data)

# Convert categorical variables to numerical using one-hot encoding
df_encoded = pd.get_dummies(df, columns=['x1', 'x2', 'x3'])

# Split data into features (X) and target (y)
X = df_encoded.drop('y', axis=1)
y = df_encoded['y']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the decision tree classifier
decision_tree = DecisionTreeClassifier(max_depth=1, criterion='entropy', random_state=42)
decision_tree.fit(X_train, y_train)

# Predict using the trained classifier
predictions = decision_tree.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

print("Optimal Decision Tree Depth 1 Accuracy:", accuracy)
