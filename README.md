# Sentiment Analysis Project with Flask

**🔗 To view the model, follow the Google Colab link in the description.**

## Overview
This project implements a sentiment analysis web application using Flask. Using machine learning models, the application analyzes restaurant reviews to classify them as positive or negative sentiments.

## Features
- **Web Application:** Allows users to input restaurant reviews and get real-time sentiment analysis results.
- **Machine Learning:** Trained models (SVC, MultinomialNB) for sentiment classification.
- **Visualization:** Uses Matplotlib for visualizing data insights.
- **Deployment:** Easily deployable with Flask on local machines.

## Requirements
Ensure you have Python 3. x installed on your system. Use pip to install the necessary Python libraries:

```bash
Setup
Clone the repository:
git clone https://github.com/sumeetpatil01/Sentiment-analysis.git
cd Sentiment-analysis
Install dependencies:
pip install -r requirements.txt
Run the application:
python app.py
Navigate to http://localhost:5000/ in your web browser.
## Project Structure
sentiment-analysis-flask/
│
├── app.py            # Main Flask application script
├── model_training.py # Script for training machine learning models
├── templates/        # HTML templates for web interface
│   ├── index.html    # Main page template
│   └── result.html   # Result page template
├── static/           # Directory for static assets (CSS, JS)
├── requirements.txt  # Python dependencies
└── README.md         # This README file
Output Screenshots
Enter a review
![Screenshot (72)](https://github.com/user-attachments/assets/a849a5a7-9d77-4301-91cf-d600a008130c)
Sentiment prediction by two different models based on their accuracy
![Screenshot (73)](https://github.com/user-attachments/assets/8bd7fb19-a495-40c5-9349-b3509d0b085f)

