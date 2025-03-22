from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the trained model
model_path = 'model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from form
        features = [float(x) for x in request.form.values()]
        final_features = [np.array(features)]

        # Make prediction
        prediction = model.predict(final_features)
        output = 'Smoker' if prediction[0] == 1 else 'Not Smoker'

        return render_template('index.html', prediction_text=f'Prediction: {output}')
    
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
