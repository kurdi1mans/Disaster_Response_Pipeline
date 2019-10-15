# Disaster_Response_Pipeline

This project is part of the Data Science Nano-Degree from Udacity.

### Table of Contents

1. [Project Motivation](#motivation)
2. [Installation](#installation)
3. [File Descriptions](#files)
4. [Instructions](#instructions)

## Project Motivation<a name="motivation"></a>

This project contains code for a machine-learning-based web app that can analyze the text of messages during natural or man-made disasters in order to provide categorizations of the messages that can be useful to emergency workers/agencies.

- The respository contains files for an ETL pipeline that processes the text messages dataset.
- It also contains files for a Machine learning pipeline that trains a model built with a Random Forest Classifier for the categorizations predictions.
- Furthermore, the respository contains code for a Flask web applications that provides an interface for message-entry and categorization output.


## Installation <a name="installation"></a>

The code in this repository requires requires a Jupyter Notebook (optional) and a python installation with the following libraries available: sys, pandas, sqlalchemy , numpy, nltk, sklearn, re, and pickle. The easiest way to get all of these is to install them through Anaconda https://www.anaconda.com/.

## File Descriptions <a name="files"></a>

- ETL pipeline: reads text messages and categorization from CSV files. Tokenizes the data and stores the results into a local database file.
	- ETL Pipeline Preparation.ipynb (ellaboration on how the pipeline works)
	- data folder (actual pipeline code)
- ML pipeline: reads the prepared data from the ETL pipeline output database and trains a model (Random Forest Classifier) on the data. The resulting model is save in pickle file.
	- ML Pipeline Preparation.ipynb (ellaboration on how the pipeline works)
	- models folder (actual pipeline code)
- Flask application: utilizes the trained model in a simple web app to predict the categories of a text message.
	- app folder

## Instructions <a name="instructions"></a>

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
