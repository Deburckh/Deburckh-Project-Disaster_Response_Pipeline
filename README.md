# Disaster Response Pipeline Project

### Summary:
This project is part of the Udacity Data Scientis Nanodegree. The goal of this project is to apply the data engineering skills learned in the course to analyze disaster data to build a model that classifies disaster messages. 

The project is divided in three sections:
1. Data Processing: build an ETL (Extract, Transform, and Load) Pipeline to extract data from the given dataset, clean the data, and then store it in a SQLite database.
2. Machine Learning Pipeline: split the data into a training set and a test set. Then, create a machine learning pipeline that uses NLTK, as well as scikit-learn's Pipeline and GridSearchCV to output a final model that predicts a message classifications for the 36 categories (multi-output classification).
3. Web development develop a web application to show classify messages in real time.

### Libraries & Installations:
- Python Version 3.6.3
- numpy
- pandas
- nltk
- sklearn
- sqlalchemy

A full list of requirements can be found under the requirements.txt file. 
To install all Python packages written in the requirements.txt file run pip install -r requirements.txt.

### Files:
There are three main folders in this project:
- data
- models
- app

Data contains the process_data.py script which reads in the disaster_categories/messages.csv files, cleans them and stores them into a sqlite database called DisasterResponse.db.

Models contains the train_classifier.py script which use the DisasterResponse.db to load it´s data. Additionally it sets up a ML pipeline, fits and evaluates the model and saves it to a picke file in the end. 

App contains the run.py script which sets up the website, the visualizations and the server.

### Instructions:
- Execute the run.py script to start the application.

If this doesn´t work try the following steps.

1. Run the following commands in the project's root directory to set up your database and model.
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
2. Go to `app` directory: `cd app`
3. Run your web app: `python run.py`

