import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Function to preprocess the dataset
@st.cache_data()
def preprocess_data(df):
    # Drop the 'case_id' column
    df = df.drop(columns=['case_id'])
    
    # Handle missing values
    imputer = SimpleImputer(strategy='most_frequent')
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    # Encode categorical variables
    label_encoder = LabelEncoder()
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = label_encoder.fit_transform(df[column])
    
    return df, label_encoder  # Return the LabelEncoder object along with the preprocessed data

# Function to train the model and make predictions
@st.cache_data()
def train_and_predict(X_train, y_train, X_test):
    # Initialize XGBoost classifier
    model = XGBClassifier()
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    return y_pred

# Main function to run the Streamlit app
def main():
    # Title and file upload
    st.title("Visa Prediction App")
    st.write("Upload your dataset to predict visa status.")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Load the dataset
        df = pd.read_csv(uploaded_file)
        
        # Preprocess the data
        visa_data, label_encoder = preprocess_data(df)
        
        # Splitting the dataset into features and target
        X = visa_data.drop(columns=['case_status'])  # Features
        y = visa_data['case_status']  # Target
        
        # Splitting the dataset into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model and make predictions
        y_pred = train_and_predict(X_train, y_train, X_test)
        
        # Print the case IDs along with predicted visa status
        predicted_labels = label_encoder.inverse_transform(y_pred)  # Corrected line
        results = pd.DataFrame({'case_id': df['case_id'].iloc[X_test.index], 'predicted_status': predicted_labels})
        st.write("Predicted Visa Status:")
        st.write(results)
        
        # Evaluate model
        accuracy = accuracy_score(y_test, y_pred)
        st.write("Accuracy:", accuracy)

# Run the main function to start the Streamlit app
if __name__ == '__main__':
    main()
