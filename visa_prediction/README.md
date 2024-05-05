
                                                      Visa Prediction App

This Streamlit app predicts visa acceptance using a Random Forest Classifier model. Users can input parameters related to visa applications to receive predictions from the trained model.

Installation:

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/visa-prediction-app.git
   ```

2. Navigate to the project directory:
   ```bash
   cd visa-prediction-app
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

 Usage:

1. Place your dataset in CSV format and name it `"Visa_Predection_Dataset.csv"`. Ensure the dataset contains relevant features and the target variable ("case_status").

2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

3. Access the app in your browser at http://localhost:8501.

4. Use the sidebar to input parameters related to visa applications.

5. The app will display the model's accuracy and classification report based on the test data.

Customization:

- Modify the input widgets in the sidebar (`app.py`) to include additional parameters for prediction.
- Adjust the machine learning model (`RandomForestClassifier`) or use a different algorithm if needed.

 Dataset:

Ensure your dataset is preprocessed and cleaned before using it with the app. It should include features that influence visa acceptance and the corresponding visa status ("case_status").

Acknowledgements:

- This app uses the Streamlit library for creating interactive web apps.
- The machine learning model is implemented using scikit-learn.

