# Sentiment-analysis
Sentiment analysis performed on restaurant data set using machine learning with python

Sentiment analysis:
Sentiment analysis is the process of classifying whether a block of text is positive,negative, or, neutral. Sentiment analysis is contextual mining of words which indicates the social sentiment of a brand and also helps the business to determine whether the product which they are manufacturing is going to make a demand in the market or not. The goal which Sentiment analysis tries to gain is to analyze people’s opinion in a way that it can help the businesses expand. It focuses not only on polarity (positive, negative & neutral) but also on emotions (happy, sad, angry, etc.). It uses various Natural Language Processing algorithms such as Rule-based, Automatic, and Hybrid
# Execution of code
Just copy the main.py file and Run the code on google colab
# code Explanation
#Step 1:-
Install all the required libraries                                                                                                                                    
pip install pandas                                                                                                                                                    
import pandas as pd                                                                                                                                                   
import matplotlib.pyplot as plt                                                                                                                                       
import numpy as np                                                                                                                                                    
import tables                                                                                                                                                         
#Step 2:-                                                                                                                                                              
Reading the data and data preprocessing                                                                                                                               
df = pd.read_table("https://raw.githubusercontent.com/jayantsinghjs7/Resturant-Reviews/master/Restaurant_Reviews.tsv")                                                                                                                                                                                            
df
df.info()  #checking for null values                                                                                                                                 
#Initializing the columns present in the data to particular variables                                                                                                 
x = df['Review'].values                                                                                                                                               
y = df['Liked'].values                                                                                                                                             
 #counting the number of positvie and negative review  values present in liked column                                                    
df['Liked'].value_counts()                                                                                                                                         
#Ploting the bar graph                                                                                                                                             
review = ['positive review','neagtive review']                                                                                                                
numbers = [500,500]
colour = ['blue','red']                                                                                                                                      
plt.bar(review,numbers,color = colour)                                                                                                                       
plt.show()                                                                                                                                                       
#Step 3:-Training the model                                                                                                                                      
#Split arrays or matrices into random train and test subsets                                                                                                    
from sklearn.model_selection import train_test_split                                                                                                               
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0)                                                                                                
#CountVectorizer It is used to transform a given text into a vector on the basis of the frequency (count) of each word that occurs in the entire text.             
from sklearn.feature_extraction.text import CountVectorizer                                                                                                         
vect = CountVectorizer(stop_words='english')                                                                                                                       
x_train_vect = vect.fit_transform(x_train)                                                                                                                          
x_test_vect = vect.transform(x_test)                                                                                                                                
x_train_vect.toarray()  #converting from vector to array                                                                                                          
#Here we are going to train 4 models and checking there accuracy the model which has the highest score is selected for testing                                     
#Model 1 SVC                                                                                                                                                         
from sklearn.svm import SVC                                                                                                                                       
model1 = SVC()                                                                                                                                                     
model1.fit(x_train_vect,y_train) #Fitting the data in the model                                                                                                    
#predicting model 1                                                                                                                                              
y_pred1 = model1.predict(x_test_vect)                                                                                                                            
y_pred1                                                                                                                                                           
#Checking the accuracy score of model 1                                                                                                                             
from sklearn.metrics import accuracy_score                                                                                                                         
accuracy_score(y_pred1,y_test)                                                                                                                                     
#Training and predicting the model 2(SVC Pipeline)                                                                                                                  
from sklearn.pipeline import make_pipeline                                                                                                                         
model2 = make_pipeline(CountVectorizer(),SVC())                                                                                                                     
model2.fit(x_train,y_train)                                                                                                                                         y_pred2 = model2.predict(x_test)                                                                                                                                    
y_pred2                                                                                                                                                            
from sklearn.metrics import accuracy_score                                                                                              
accuracy_score(y_pred2,y_test)              #Accuracy score of model2                                                                                               
#Training and predicting the model 3(MultinomialNB)                                                                                                                
from sklearn.naive_bayes import MultinomialNB                                                                                                                      
model3 = MultinomialNB()                                                                                                                                            
model3.fit(x_train_vect,y_train)                                                                                                                                  
y_pred3 = model3.predict(x_test_vect)                                                                                                                              
y_pred3                                                                                                                                                            
from sklearn.metrics import accuracy_score                                                                                                                       
accuracy_score(y_pred3,y_test)      #Accuracy score of model 3                                                                                                     
#Training and predicting the model 4(MultinomialNB Pipeline)                                                                                                       
from sklearn.pipeline import make_pipeline                                                                                                                         
model4 = make_pipeline(CountVectorizer(),MultinomialNB())                                                                                                          
model4.fit(x_train,y_train)                                                                                                                                        
y_pred4 = model4.predict(x_test)                                                                                                                                   
y_pred4                                                                                                                                                            
from sklearn.metrics import accuracy_score                                                                                                                         
accuracy_score(y_pred4,y_test) #Accuracy score of model 4                                                                                                          
#You can train a sklearn models in parallel using the sklearn joblib interface. This allows sklearn to take full advantage of the multiple cores in your machine and speed up training.                                                                                                                                              
import joblib                                                                                                                                                       
joblib.dump(model2,'0-1')                                                                                                                                           
import joblib                                                                                                                                                       
text_model = joblib.load('0-1')                                                                                                                                     
text_model    
#Step 4:-Predicting by taking the input if it gives output as 1 then its a positive review or negative review i.e,0                                                
text_model.predict(["food was not there on the table"])                                                                                                             
# Output Screenshort                                                                                                                                                
![Screenshot (18)](https://github.com/sumeetpatil01/Sentiment-analysis/assets/136491586/0b57e724-2bcd-495d-813a-291b9e467f59)                                      
as you can see it is giving 0 as ouput then its a negative review                                                                                                
![Screenshot (19)](https://github.com/sumeetpatil01/Sentiment-analysis/assets/136491586/216b77f3-c97a-4e9b-962a-390b93b1a9ac)                                     
Now it is giving the output as 1 then its a positive review                                                                                                       
# Accuracy scores of all four models                                                                                                                                
#SVC                       - 0.72                                                                                                                                  
#SVC pipeline              - 0.792                                                                                                                                  
#MultinomialNB             - 0.744                                                                                                                                  
#MultinomialNB pipeline    - 0.784

