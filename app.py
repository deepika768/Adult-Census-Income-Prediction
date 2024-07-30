import numpy as np
from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, Predict_pipeline

# Create Flask app
app = Flask(__name__)

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        data = CustomData(
            age=int(request.form.get('age')),
            workclass=request.form.get('workclass'),
            education=request.form.get('education'),
            marital_status=request.form.get('marital-status'),
            occupation=request.form.get('occupation'),
            relationship=request.form.get('relationship'),
            race=request.form.get('race'),
            sex=request.form.get('sex'),
            capital_gain=int(request.form.get('capital-gain')),
            capital_loss=int(request.form.get('capital-loss')),
            hours_per_week=int(request.form.get('hours-per-week')),
            country=request.form.get('country')
        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline = Predict_pipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("After Prediction")
        
        return render_template('index.html', results=results[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0")
