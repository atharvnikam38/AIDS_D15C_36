from flask import Flask, render_template, request, jsonify, send_file, session
import pandas as pd
import numpy as np
import joblib
import io
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
from collections import defaultdict

app = Flask(__name__)
app.secret_key = 'airline_satisfaction_predictor'  # needed for session

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

# Rating features for identifying key factors
rating_features = [
    'Inflight wifi service', 'Departure/Arrival time convenient', 
    'Ease of Online booking', 'Gate location', 'Food and drink', 
    'Online boarding', 'Seat comfort', 'Inflight entertainment', 
    'On-board service', 'Leg room service', 'Baggage handling', 
    'Checkin service', 'Inflight service', 'Cleanliness'
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
    
    # Store the original data before dropping Class
    original_data = input_df.copy()
    
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
    
    return input_df, original_data

def identify_key_factors(input_data):
    """Identify top positive and negative factors based on user input"""
    # Rating factors (1-5 scale)
    rating_factors = {}
    for feature in rating_features:
        if feature in input_data:
            rating_factors[feature] = float(input_data[feature])
    
    # Sort factors by rating
    sorted_factors = sorted(rating_factors.items(), key=lambda x: x[1])
    
    # Get top 3 lowest and highest rated factors
    negative_factors = []
    positive_factors = []
    
    # Bottom 3 factors (negative)
    for factor, rating in sorted_factors[:3]:
        # Make factor name more user-friendly
        factor_name = factor.replace('_', ' ').title()
        negative_factors.append(f"{factor_name} (Low Rating)")
    
    # Top 3 factors (positive)
    for factor, rating in sorted_factors[-3:]:
        # Make factor name more user-friendly
        factor_name = factor.replace('_', ' ').title()
        positive_factors.append(f"{factor_name} (High Rating)")
    
    # Check delay times
    if 'Departure Delay in Minutes' in input_data and float(input_data['Departure Delay in Minutes']) > 30:
        negative_factors.append("Departure Delay (Long)")
    elif 'Departure Delay in Minutes' in input_data and float(input_data['Departure Delay in Minutes']) < 15:
        positive_factors.append("Short Departure Delay")
        
    if 'Arrival Delay in Minutes' in input_data and float(input_data['Arrival Delay in Minutes']) > 30:
        negative_factors.append("Arrival Delay (Long)")
    elif 'Arrival Delay in Minutes' in input_data and float(input_data['Arrival Delay in Minutes']) < 15:
        positive_factors.append("Short Arrival Delay")
    
    # Ensure we have at most 3 factors for each category
    negative_factors = negative_factors[:3]
    positive_factors = positive_factors[:3]
    
    return positive_factors, negative_factors

def analyze_batch_results(df):
    """Analyze batch prediction results and generate insights"""
    total_records = len(df)
    satisfied_count = (df['Prediction'] == 'Satisfied').sum()
    satisfaction_rate = round((satisfied_count / total_records) * 100, 1)
    
    # Calculate average ratings for each service feature
    rating_averages = {}
    for feature in rating_features:
        if feature in df.columns:
            rating_averages[feature] = df[feature].mean()
    
    # Find top satisfying and dissatisfying factors
    sorted_ratings = sorted(rating_averages.items(), key=lambda x: x[1])
    
    # Bottom factor (most dissatisfying)
    bottom_factor = sorted_ratings[0]
    bottom_factor_name = bottom_factor[0].replace('_', ' ').title()
    bottom_factor_rating = round(bottom_factor[1], 1)
    
    # Top factor (most satisfying)
    top_factor = sorted_ratings[-1] 
    top_factor_name = top_factor[0].replace('_', ' ').title()
    top_factor_rating = round(top_factor[1], 1)
    
    # Satisfaction by demographic segments
    satisfaction_by_gender = df.groupby('Gender')['Prediction'].apply(
        lambda x: round((x == 'Satisfied').mean() * 100, 1)).to_dict()
    
    satisfaction_by_type = df.groupby('Customer Type')['Prediction'].apply(
        lambda x: round((x == 'Satisfied').mean() * 100, 1)).to_dict()
    
    satisfaction_by_class = df.groupby('Class')['Prediction'].apply(
        lambda x: round((x == 'Satisfied').mean() * 100, 1)).to_dict()
    
    satisfaction_by_travel = df.groupby('Type of Travel')['Prediction'].apply(
        lambda x: round((x == 'Satisfied').mean() * 100, 1)).to_dict()
    
    insights = {
        'total_records': total_records,
        'satisfied_count': satisfied_count,
        'satisfaction_rate': satisfaction_rate,
        'top_factor': {
            'name': top_factor_name,
            'rating': top_factor_rating
        },
        'bottom_factor': {
            'name': bottom_factor_name,
            'rating': bottom_factor_rating
        },
        'demographics': {
            'gender': satisfaction_by_gender,
            'customer_type': satisfaction_by_type,
            'class': satisfaction_by_class,
            'travel_type': satisfaction_by_travel
        }
    }
    
    return insights

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
        processed_data, original_data = preprocess_input(form_data)
        
        # Make prediction
        prediction = model.predict(processed_data)
        probability = model.predict_proba(processed_data)[0]
        
        # Convert prediction to human-readable form
        result = "Satisfied" if prediction[0] == 1 else "Neutral or Dissatisfied"
        satisfaction_prob = round(probability[1] * 100, 2)
        dissatisfaction_prob = round(probability[0] * 100, 2)
        
        # Identify key factors
        positive_factors, negative_factors = identify_key_factors(form_data)
        
        return render_template('result.html', 
                             prediction=result,
                             satisfaction_prob=satisfaction_prob,
                             dissatisfaction_prob=dissatisfaction_prob,
                             input_data=form_data,
                             positive_factors=positive_factors,
                             negative_factors=negative_factors)
    
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
        
        # Store original data
        original_df = df.copy()
        
        # Preprocess the data
        processed_data, _ = preprocess_input(df, is_batch=True)
        
        # Make predictions
        predictions = model.predict(processed_data)
        probabilities = model.predict_proba(processed_data)
        
        # Add predictions to original DataFrame
        original_df['Prediction'] = ['Satisfied' if p == 1 else 'Neutral or Dissatisfied' for p in predictions]
        original_df['Satisfaction Probability'] = [round(p[1] * 100, 2) for p in probabilities]
        original_df['Dissatisfaction Probability'] = [round(p[0] * 100, 2) for p in probabilities]
        
        # Analyze results
        insights = analyze_batch_results(original_df)
        
        # Save results to Excel for download
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            original_df.to_excel(writer, index=False)
        output.seek(0)
        
        # Store the processed data in session for download
        session['batch_results'] = output.getvalue()
        
        # Sample results for display
        sample_results = original_df.head(5).to_dict('records')
        
        return render_template('batch_result.html', 
                             sample_results=sample_results,
                             total_records=insights['total_records'],
                             satisfied_count=insights['satisfied_count'],
                             satisfaction_rate=insights['satisfaction_rate'],
                             top_factor=insights['top_factor'],
                             bottom_factor=insights['bottom_factor'],
                             demographics=insights['demographics'],
                             send_file=True)
    
    except Exception as e:
        return render_template('index.html', error=str(e))

@app.route('/download_results')
def download_results():
    try:
        # Get processed data from session
        if 'batch_results' not in session:
            return "No batch results found. Please process a file first.", 400
        
        output = io.BytesIO(session['batch_results'])
        
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