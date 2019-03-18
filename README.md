# Disaster Response Pipeline Project

### Description

This is a training project with the aim of analysing disaster data from Figure Eight and creating a model that classifies disaster messages. The project consists of 3 stages (ETL Pipeline, ML Pipeline, and web application) that upload and clean the initial data, classify it according to the task and then upload as an app.

Note: long training time

### Installation

The code contained in this repository was written in HTML and Python 3, and requires the following Python packages: json, plotly, pandas, nltk, flask, sklearn, sqlalchemy, sys, numpy, re, pickle, warnings

### Running Instructions

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Important files

#### process_data.py : ETL Pipeline 
    
Actions: load and parse datasets, cleanse the data and store the data in a SQLite database.
   
#### train_classifier.py : ML Pipeline 

Actions: load the data from SQLite database, splits the data into training and test sets, build a text processing and clssification pipeline, trains and tunes a model using GridSearchCV and exports the final model as a pickle file.
    
#### Run.py : Flask Web App 

Actions: display the visualization (the app accept messages from users and returns classification results for 36 categories of disaster events).

#### Limitations and possible improvement

The used datasets are very unbalanced, with very few positive examples for some message categories. This results in a low recall rate despite having high accuracy.

This app must not be used for actual pridiction unless more data is collected.

#### Screenshots

![Alt text](https://github.com/pol690/Disaster-Response-Pipeline-Project/blob/master/Screen1.png "Screenshot1")
![Alt text](https://github.com/pol690/Disaster-Response-Pipeline-Project/blob/master/Screen2.png "Screenshot2")
![Alt text](https://github.com/pol690/Disaster-Response-Pipeline-Project/blob/master/Screen3.png "Screenshot3")
#### Licensing and Acknowledgements

This app was developed as part of the Udacity Data Scientist Nanodegree.
