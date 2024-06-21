from flask import Flask, request, render_template
import joblib
import os

app = Flask(__name__, template_folder='Template')

# Load the model
model = joblib.load('svc_pipeline_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']
        prediction = model.predict([review])
        sentiment = 'Positive' if prediction == 1 else 'Negative'
        return render_template('result.html', review=review, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
