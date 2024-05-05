import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv("Visa_Predection_Dataset.csv")

# Preprocess the data
X = df.drop('case_status', axis=1)
y = df['case_status']
df.dropna(inplace=True)
X = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose a machine learning algorithm (Random Forest Classifier)
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Streamlit app
st.title("Visa Prediction App")

# Sidebar for user input
st.sidebar.header("Input Parameters")
# Add input widgets for user to enter values
# Example:
# user_input = st.sidebar.number_input("Enter a value", min_value=0, max_value=10)

# Make predictions and display result
# Example:
# prediction = model.predict([[user_input]])
# st.write(f"Prediction: {prediction}")

# Evaluate the model (optional)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: {accuracy:.2f}")

# Display classification report (optional)
classification_rep = classification_report(y_test, y_pred)
st.write("Classification Report:")
st.write(classification_rep)
