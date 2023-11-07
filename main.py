pip install pandas
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tables
df = pd.read_table("https://raw.githubusercontent.com/jayantsinghjs7/Resturant-Reviews/master/Restaurant_Reviews.tsv") #data set url
df
df.info()
x = df['Review'].values
y = df['Liked'].values
df['Liked'].value_counts()                                          #0=negative review
                                                                    #1=positive review
review = ['positive review','neagtive review']
numbers = [500,500]
colour = ['blue','red']
plt.bar(review,numbers,color = colour)
plt.show()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(stop_words='english')
x_train_vect = vect.fit_transform(x_train)
x_test_vect = vect.transform(x_test)
x_train_vect.toarray()
from sklearn.svm import SVC
model1 = SVC()
model1.fit(x_train_vect,y_train)
y_pred1 = model1.predict(x_test_vect)
y_pred1
from sklearn.metrics import accuracy_score
accuracy_score(y_pred1,y_test)
from sklearn.pipeline import make_pipeline
model2 = make_pipeline(CountVectorizer(),SVC())
model2.fit(x_train,y_train)
y_pred2 = model2.predict(x_test)
y_pred2
from sklearn.metrics import accuracy_score
accuracy_score(y_pred2,y_test)
from sklearn.naive_bayes import MultinomialNB
model3 = MultinomialNB()
model3.fit(x_train_vect,y_train)
y_pred3 = model3.predict(x_test_vect)
y_pred3
from sklearn.metrics import accuracy_score
accuracy_score(y_pred3,y_test)
from sklearn.pipeline import make_pipeline
model4 = make_pipeline(CountVectorizer(),MultinomialNB())
model4.fit(x_train,y_train)
y_pred4 = model4.predict(x_test)
y_pred4
from sklearn.metrics import accuracy_score
accuracy_score(y_pred4,y_test)
import joblib
joblib.dump(model2,'0-1')
import joblib
text_model = joblib.load('0-1')
text_model
text_model.predict(["food was not there on the table"])   
 #Here you can give the input and the model will predict the review was positive or negative                         
           #0=negative review                       #1=positive review

