# Sentiment Analysis Project with Flask
Overview
This project implements a sentiment analysis web application using Flask. The application analyzes restaurant reviews to classify them as positive or negative sentiments using machine learning models.

Features
Web Application: Allows users to input restaurant reviews and get real-time sentiment analysis results.
Machine Learning: Trained models (SVC, MultinomialNB) for sentiment classification.
Visualization: Uses matplotlib for visualizing data insights.
Deployment: Easily deployable with Flask on local machines.
Requirements
Ensure you have Python 3.x installed on your system. Use pip to install the necessary Python libraries:

bash
Copy code
pip install -r requirements.txt                                                                                                
The requirements.txt file contains:

Flask==2.0.2
pandas==1.3.3
scikit-learn==0.24.2
matplotlib==3.4.3
numpy==1.21.2
joblib==1.1.0
Setup
Clone the repository:

bash
Copy code
git clone https://github.com/sumeetpatil01/Sentiment-analysis.git                                                                                                 
cd sentiment-analysis                                                                                                                  
Install dependencies                                                                                                                                                                            
                                                                                                                                                     bash
Copy code:                                                                                                                                     
pip install -r requirements.txt                                                                                                                                  
Run the application:

bash
Copy code
python app.py
Navigate to http://localhost:5000/ in your web browser.

Project Structure
php
Copy code
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
# Output Screenshots                                                                                                                                 
Enter a review
![Screenshot (47)](https://github.com/user-attachments/assets/3e02759e-4462-41fe-b8d1-3b88ce35c722)
Sentiment prediction by two different models based on their accuracy
![Screenshot (46)](https://github.com/user-attachments/assets/96600618-44cb-4275-ae60-67e08cab14dd)






