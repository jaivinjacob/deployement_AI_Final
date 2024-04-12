from flask import Flask, request, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Initialize the model and scaler as global variables
model = None
scaler = None

@app.route('/', methods=['GET', 'POST'])
def index():
    global model, scaler
    
    if request.method == 'POST':
        # Handle file upload
        file = request.files.get('file')
        if file:
            data = pd.read_csv(file)
            # Separate features and target
            X = data[['feature1', 'feature2']]
            y = data['target']
            
            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            
            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train a logistic regression model
            model = LogisticRegression()
            model.fit(X_train_scaled, y_train)
            
            # Predict and get the classification report
            predictions = model.predict(X_test_scaled)
            report = classification_report(y_test, predictions)
            
            return render_template('result.html', report=report)
        
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    global model, scaler
    
    # Check if model and scaler are properly initialized
    if model is None or scaler is None:
        return "Model and/or scaler not initialized. Please upload data and train the model first.", 400
    
    if request.method == 'POST':
        # Get user input values
        feature1 = float(request.form['feature1'])
        feature2 = float(request.form['feature2'])
        
        # Standardize the input data using the previously fitted scaler
        input_data = [[feature1, feature2]]
        input_data_scaled = scaler.transform(input_data)
        
        # Make a prediction using the trained model
        prediction = model.predict(input_data_scaled)
        
        return render_template('predict_result.html', prediction=prediction)
    
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
