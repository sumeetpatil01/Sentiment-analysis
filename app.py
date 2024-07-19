from flask import Flask, request, render_template
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__, template_folder='Template')

# Load the models
model_svc = joblib.load('svc_pipeline_model.pkl')
model_nb = joblib.load('multiNB_pipeline_model.pkl')

# Load the data to get the counts
df = pd.read_table("https://raw.githubusercontent.com/jayantsinghjs7/Resturant-Reviews/master/Restaurant_Reviews.tsv")
review_counts = df['Liked'].value_counts().sort_index().tolist()

# Calculate model accuracies
accuracy_svc = model_svc.score(df['Review'], df['Liked']) * 100
accuracy_nb = model_nb.score(df['Review'], df['Liked']) * 100

# Function to generate accuracy comparison plot
def plot_accuracy_comparison(accuracy_svc, accuracy_nb):
    labels = ['SVC', 'MultinomialNB']
    accuracies = [accuracy_svc, accuracy_nb]

    plt.figure(figsize=(8, 6))
    plt.bar(labels, accuracies, color=['blue', 'green'])
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Models')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)

    for i, accuracy in enumerate(accuracies):
        plt.text(i, accuracy + 2, f'{accuracy:.2f}%', ha='center', va='bottom')

    plt.tight_layout()

    # Save the figure to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return plot_url

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']

        # Predictions
        prediction_svc = model_svc.predict([review])
        prediction_nb = model_nb.predict([review])

        # Sentiment labels
        sentiment_svc = 'Positive' if prediction_svc == 1 else 'Negative'
        sentiment_nb = 'Positive' if prediction_nb == 1 else 'Negative'

        # Generate plots for predictions
        plot_url_svc = generate_sentiment_plot(sentiment_svc)
        plot_url_nb = generate_sentiment_plot(sentiment_nb)

        # Accuracy comparison plot
        plot_url_accuracy = plot_accuracy_comparison(accuracy_svc, accuracy_nb)

        return render_template('result.html', review=review,
                               plot_url_svc=plot_url_svc, plot_url_nb=plot_url_nb,
                               sentiment_svc=sentiment_svc, sentiment_nb=sentiment_nb,
                               accuracy_svc=accuracy_svc, accuracy_nb=accuracy_nb,
                               plot_url_accuracy=plot_url_accuracy)

def generate_sentiment_plot(sentiment):
    sizes = [60, 40] if sentiment == 'Positive' else [40, 60]
    colors = ['green', 'lightgrey'] if sentiment == 'Positive' else ['lightgrey', 'red']

    plt.figure(figsize=(4, 4))
    plt.pie(sizes, colors=colors, startangle=90, autopct='%1.1f%%', wedgeprops={'edgecolor': 'black'})
    plt.title(f'{sentiment} Review')

    # Save the figure to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return plot_url

if __name__ == '__main__':
    app.run(debug=True)
