import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Example classifier
from sklearn.metrics import accuracy_score
import joblib  # For saving the trained model
import os

# Load the data
data = pd.read_excel("match_data.xlsx", sheet_name="Sheet1")

# Basic data preprocessing and feature engineering
# (Adjust based on your actual data structure)
data['Outcome'] = data.apply(lambda x: 'HomeWin' if x['HomeGoals'] > x['AwayGoals'] else 
                             ('AwayWin' if x['AwayGoals'] > x['HomeGoals'] else 'Draw'), axis=1)

# Select features and labels
features = data[['HomeTeam', 'AwayTeam']]  # Replace with actual features used for training
labels = data['Outcome']

# Encode categorical features
features = pd.get_dummies(features, columns=['HomeTeam', 'AwayTeam'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")

# Save the trained model
joblib.dump(model, r"D:\streamlit\match_outcome_predictor.pkl")

print("Model saved as match_outcome_predictor.pkl")

print("Current working directory:", os.getcwd())
