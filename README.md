# SMS Spam Detection using Machine Learning & NLP

This project is a SMS spam detection system that leverages Natural Language Processing (NLP) and Machine Learning to identify and classify text messages as either spam or ham (non-spam). The project utilizes popular libraries such as NLTK, Streamlit, and TF-IDF, and it incorporates a machine learning model trained on a dataset from the UCI Machine Learning Repository.

Overview
In today's digital age, unwanted SMS spam is a common nuisance. This project aims to tackle this issue by providing an efficient tool to automatically detect and filter out spam messages. The system is designed to help users keep their SMS inboxes clean and free from unwanted content.

About 

SMS Classification: The core feature of this project is its ability to classify incoming SMS messages as spam or ham.
User-Friendly Interface: The project offers a user-friendly interface developed using Streamlit, making it easy for users to interact with the application.
Text Preprocessing: Text data undergoes preprocessing using NLTK for tasks like tokenization, stop-word removal, and stemming.
TF-IDF Vectorization: The text messages are converted into numerical vectors using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization, which is crucial for training the machine learning model.
Machine Learning Model: Machine learning models (e.g., Naive Bayes, Support Vector Machine, or others) is trained and testes on a dataset from the UCI Machine Learning Repository to classify SMS messages.
