import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import joblib

# Load the data
df = pd.read_table("https://raw.githubusercontent.com/jayantsinghjs7/Resturant-Reviews/master/Restaurant_Reviews.tsv")

# Data exploration
print(df.info())
print(df['Liked'].value_counts())

# Plotting the distribution of reviews
review = ['positive review', 'negative review']
numbers = df['Liked'].value_counts().sort_index().tolist()
colors = ['blue', 'red']

plt.bar(review, numbers, color=colors)
plt.title("Distribution of Reviews")
plt.xlabel("Review Type")
plt.ylabel("Number of Reviews")
plt.show()

# Data preparation
x = df['Review'].values
y = df['Liked'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, stratify=y)

# Vectorize the text data
vect = CountVectorizer(stop_words='english')
x_train_vect = vect.fit_transform(x_train)
x_test_vect = vect.transform(x_test)

# Model 1: SVC
model1 = SVC()
model1.fit(x_train_vect, y_train)
y_pred1 = model1.predict(x_test_vect)
print("SVC Accuracy:", accuracy_score(y_pred1, y_test))

# Model 2: SVC with Pipeline
model2 = make_pipeline(CountVectorizer(stop_words='english'), SVC())
model2.fit(x_train, y_train)
y_pred2 = model2.predict(x_test)
print("SVC Pipeline Accuracy:", accuracy_score(y_pred2, y_test))

# Model 3: MultinomialNB
model3 = MultinomialNB()
model3.fit(x_train_vect, y_train)
y_pred3 = model3.predict(x_test_vect)
print("MultinomialNB Accuracy:", accuracy_score(y_pred3, y_test))

# Model 4: MultinomialNB with Pipeline
model4 = make_pipeline(CountVectorizer(stop_words='english'), MultinomialNB())
model4.fit(x_train, y_train)
y_pred4 = model4.predict(x_test)
print("MultinomialNB Pipeline Accuracy:", accuracy_score(y_pred4, y_test))

# Save the best model
joblib.dump(model4, 'MultiNB_pipeline_model.pkl')
