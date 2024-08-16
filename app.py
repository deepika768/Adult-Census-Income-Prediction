import numpy as np
from flask import Flask, request, render_template
import pickle
from urllib.parse import quote as url_quote

# Create Flask app
app = Flask(__name__)

# Load the model once globally
model = pickle.load(open('model.pkl', 'rb'))

# Prediction function
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, -1)
    result = model.predict(to_predict)
    return result[0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))  # Ensure all inputs are integers
        result = ValuePredictor(to_predict_list)
        
        if int(result) == 1:
            prediction = 'Income is >50K $'
        else:
            prediction = 'Income is <=50K $'
        
        return render_template('index.html', prediction_text=prediction)

if __name__ == "__main__":
    app.run(debug=True)
