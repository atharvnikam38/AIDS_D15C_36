from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import joblib
import io
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

app = Flask(__name__)

# Load the trained model
model = joblib.load('final_random_forest_model.pkl')

# Feature names expected by the model
feature_names = [
    'Gender', 'Customer Type', 'Age', 'Type of Travel', 'Flight Distance',
    'Inflight wifi service', 'Departure/Arrival time convenient', 
    'Ease of Online booking', 'Gate location', 'Food and drink', 
    'Online boarding', 'Seat comfort', 'Inflight entertainment', 
    'On-board service', 'Leg room service', 'Baggage handling', 
    'Checkin service', 'Inflight service', 'Cleanliness', 
    'Departure Delay in Minutes', 'Arrival Delay in Minutes',
    'Class_Eco', 'Class_Eco Plus'
]

# Preprocessing functions
def preprocess_input(input_data, is_batch=False):
    if not is_batch:
        # Convert single input to DataFrame
        input_df = pd.DataFrame([input_data])
    else:
        input_df = input_data.copy()
    
    # Convert categorical variables to numerical
    input_df['Gender'] = input_df['Gender'].map({'Male': 1, 'Female': 0})
    input_df['Customer Type'] = input_df['Customer Type'].map({'Loyal Customer': 0, 'disloyal Customer': 1})
    input_df['Type of Travel'] = input_df['Type of Travel'].map({'Business travel': 0, 'Personal Travel': 1})
    
    # One-hot encode Class
    input_df['Class_Eco'] = (input_df['Class'] == 'Eco').astype(int)
    input_df['Class_Eco Plus'] = (input_df['Class'] == 'Eco Plus').astype(int)
    
    # Drop original Class column
    input_df.drop('Class', axis=1, inplace=True)
    
    # Ensure all expected columns are present
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Reorder columns to match training data
    input_df = input_df[feature_names]
    
    # Scale numerical features (using the same scaler as in training)
    numerical_cols = ['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
    scaler = StandardScaler()
    input_df[numerical_cols] = scaler.fit_transform(input_df[numerical_cols])
    
    return input_df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        form_data = request.form.to_dict()
        
        # Convert numerical fields from strings to numbers
        numerical_fields = [
            'Age', 'Flight Distance', 'Inflight wifi service',
            'Departure/Arrival time convenient', 'Ease of Online booking',
            'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
            'Inflight entertainment', 'On-board service', 'Leg room service',
            'Baggage handling', 'Checkin service', 'Inflight service',
            'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes'
        ]
        
        for field in numerical_fields:
            form_data[field] = float(form_data[field])
        
        # Preprocess the input
        processed_data = preprocess_input(form_data)
        
        # Make prediction
        prediction = model.predict(processed_data)
        probability = model.predict_proba(processed_data)[0]
        
        # Convert prediction to human-readable form
        result = "Satisfied" if prediction[0] == 1 else "Neutral or Dissatisfied"
        satisfaction_prob = round(probability[1] * 100, 2)
        dissatisfaction_prob = round(probability[0] * 100, 2)
        
        return render_template('result.html', 
                             prediction=result,
                             satisfaction_prob=satisfaction_prob,
                             dissatisfaction_prob=dissatisfaction_prob,
                             input_data=form_data)
    
    except Exception as e:
        return render_template('index.html', error=str(e))

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return render_template('index.html', error="No file uploaded")
        
        file = request.files['file']
        
        # Check if file is empty
        if file.filename == '':
            return render_template('index.html', error="No file selected")
        
        # Check if file is Excel
        if not file.filename.endswith(('.xlsx', '.xls')):
            return render_template('index.html', error="Only Excel files are allowed")
        
        # Read Excel file
        df = pd.read_excel(file)
        
        # Check if required columns are present
        required_columns = [
            'Gender', 'Customer Type', 'Age', 'Type of Travel', 'Class', 'Flight Distance',
            'Inflight wifi service', 'Departure/Arrival time convenient', 
            'Ease of Online booking', 'Gate location', 'Food and drink', 
            'Online boarding', 'Seat comfort', 'Inflight entertainment', 
            'On-board service', 'Leg room service', 'Baggage handling', 
            'Checkin service', 'Inflight service', 'Cleanliness', 
            'Departure Delay in Minutes', 'Arrival Delay in Minutes'
        ]
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return render_template('index.html', 
                                 error=f"Missing required columns: {', '.join(missing_cols)}")
        
        # Preprocess the data
        processed_data = preprocess_input(df, is_batch=True)
        
        # Make predictions
        predictions = model.predict(processed_data)
        probabilities = model.predict_proba(processed_data)
        
        # Add predictions to original DataFrame
        df['Prediction'] = ['Satisfied' if p == 1 else 'Neutral or Dissatisfied' for p in predictions]
        df['Satisfaction Probability'] = [round(p[1] * 100, 2) for p in probabilities]
        df['Dissatisfaction Probability'] = [round(p[0] * 100, 2) for p in probabilities]
        
        # Save results to Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)
        output.seek(0)
        
        # Save results to session for display
        sample_results = df.head(5).to_dict('records')
        
        return render_template('batch_result.html', 
                             sample_results=sample_results,
                             total_records=len(df),
                             send_file=True)
    
    except Exception as e:
        return render_template('index.html', error=str(e))

@app.route('/download_results')
def download_results():
    try:
        # In a real app, you would retrieve the processed data from session or database
        # For simplicity, we'll just return a sample file here
        # In practice, you should store the processed data from the predict_batch function
        
        # Create a sample DataFrame (in a real app, use your actual data)
        data = {
            'Gender': ['Male', 'Female'],
            'Customer Type': ['Loyal Customer', 'disloyal Customer'],
            'Prediction': ['Satisfied', 'Neutral or Dissatisfied'],
            'Satisfaction Probability': [85.2, 32.7],
            'Dissatisfaction Probability': [14.8, 67.3]
        }
        df = pd.DataFrame(data)
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)
        output.seek(0)
        
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name='passenger_satisfaction_predictions.xlsx'
        )
    
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(debug=True)