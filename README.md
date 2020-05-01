# Introduction

This project analyzes data provided by [Figure Eight](https://appen.com/datasets/combined-disaster-response-data/) containing thousands of raw texts following a natural disaster. The data was collected either from social media or direct messages to emergency responders. This project will have three parts:

1. Build an extract, transform, load (ETL) pipeline that loads the data from csv, cleans and prepares it using Pandas, and loads it back into an sqlite database.
2. Build a machine learning pipeline that reads from the sqlite database to create a mutli-output supervised learning model.
3. Build a web application that extracts data from the sqlite database to create data visualizations and use the model to classify new messages.

# File Descriptions

```
- app
  | - template
  | - master.html
  | - go.html
  | - run.py
- data
  | - disaster_categories.csv = raw data from Figure 8
  | - disaster_messages.csv = raw data from Figure 8
  | - etl_pipeline.py = ETL script to generate DisasterResponse.db from raw data
  | - DisasterResponse.db = output database from etl_pipeline.py
- models
  | - train.py = ML script to create classifier.pkl from DisasterResponse.db data
  | - classifier.pkl = output ML model from train.py
- jupyter
  | - etl-preparation.ipynb = jupyter notebook explaining etl_pipeline.py
  | - ml-pipeline.ipynb = jupyteer notebook explaining train.py
- README.md
```

# Build Instructions

- This project must be run using Python3 since the nltk library is only available for Python3.
- Install Python dependencies:
```
pip install -r requirements.txt
```
- To run the ETL pipeline creating the sqlite database of the cleaned data:
```
python data/etl_pipeline.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```
- To build the ML pipeline creatimg a multi-output supervised learning model:
```
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```
- To start the web app which will visualize the data and run the model to classify new messages:
```
python app/run.py
```
