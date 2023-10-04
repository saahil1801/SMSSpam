# Import necessary libraries and load the pre-trained TF-IDF vectorizer and voting ensemble classifier using 'pickle'.
import streamlit as st
import pickle
import string
import nltk
import numpy as np
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# Define a function 'transform_text' to preprocess the input text by converting it to lowercase, tokenizing, removing non-alphanumeric characters,
# removing stopwords and punctuation, and applying stemming using the NLTK library.
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load the pre-trained TF-IDF vectorizer ('tfidf') and the voting ensemble classifier ('model') using 'pickle'.
tfidf = pickle.load(open("C:\\Users\\saahi\\SMSSpamproj\\vectorizertr.pkl",'rb'))
model = pickle.load(open("C:\\Users\\saahi\\SMSSpamproj\\modeltr1.pkl",'rb'))


# Create a Streamlit web application titled "SMS Spam Classifier."
st.title("SMS Spam Classifier")

# Create a Streamlit text input area for users to enter the SMS message.
input_sms = st.text_area("Enter the message")
# When the user clicks the "Predict" button:


if st.button('Predict'):

    transformed_sms = transform_text(input_sms) # - Preprocess the input SMS using the 'transform_text' function.
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms]).toarray() # - Vectorize the preprocessed SMS using the loaded TF-IDF vectorizer.
    # 3. predict
    result = model.predict(vector_input)[0] # - Make a prediction using the loaded ensemble classifier.
    # 4. Display
    if result == 1:  # - Display the prediction result as "Spam" or "Not Spam" based on the model's prediction.
        st.header("Spam")
    else:
        st.header("Not Spam")