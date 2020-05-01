# Introduction

This project analyzes data provided by [Figure Eight](https://appen.com/datasets/combined-disaster-response-data/) containing thousands of raw texts following a natural disaster. The data was collected either from social media or direct messages to emergency responders. This project will have three parts:

1. Build an extract, transform, load (ETL) pipeline that loads the data from csv, cleans and prepares it using Pandas, and loads it back into an sqlite database.
2. Build a machine learning pipeline that reads from the sqlite database to create a mutli-output supervised learning model.
3. Build a web application that extracts data from the sqlite database to create data visualizations and use the model to classify new messages.

# File Descriptions

```
- app
  | - templates = contains html templates for the web app
  | | - master.html
  | | - go.html
  | - run.py = main flask file
- data
  | - categories.csv = raw categories data from Figure 8
  | - messages.csv = raw messages data from Figure 8
  | - process_data.py = python script to generate database from raw data
  | - DisasterResponse.db = output database from etl_pipeline.py (not stored)
- models
  | - train_classifier.py = script to create ML model
  | - classifier.pkl = output ML model from train_classifier.py (not stored)
- jupyter
  | - etl-preparation.ipynb = jupyter notebook for etl_pipeline.py
  | - ml-pipeline.ipynb = jupyteer notebook for train_classifier.py
- requirements.txt = python dependencies for project
- README.md
- LICENSE
```

# Build Instructions

This project must be run using Python3. Perform the following steps to deploy the web app:
1. Install all of the Python dependencies:
```
pip install -r requirements.txt
```
2. Run the ETL pipeline to create an sqlite database of the cleaned data:
```
python data/process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db
```
3. Run the ML pipeline to build a multi-output supervised ML model:
```
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```
4. Start the web app which will visualize the data and run the ML model to classify new messages:
```
python app/run.py
```
5. Navigate to the web app in your browser at `http://0.0.0.0:3001/`
