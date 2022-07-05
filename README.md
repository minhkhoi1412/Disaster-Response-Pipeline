# **Disaster Response Pipeline Project**


## **Table of contents**

- [Environment Setup](#environment-setup)
- [Project Descriptions](#project-descriptions)
- [File Structure](#file-structure)
- [Usage](#usage)


## **Environment Setup**

**Environment**
- OS: Windows 10

- Interpreter: Visual Studio Code

- Python version: Python 3.8+

**Libraries**
- Install all packages using requirements.txt file. This is the command to install: `pip install -r requirements.txt`


## **Project Descriptions**

In this project, we will analyze disaster data set containing real messages that were sent during disaster events to build a model for an API that classifies disaster messages.


## **File Structure**

~~~~~~~
disaster_response_pipeline
    |-- app
        |-- templates
                |-- go.html
                |-- master.html
        |-- run.py
    |-- data
        |-- DisasterResponseETL.db
        |-- disaster_message.csv
        |-- disaster_categories.csv
        |-- process_data.py
    |-- models
        |-- DisasterResponseModel.pkl
        |-- train_classifier.py
    |-- Preparation
        |-- ETL Pipeline Preparation.ipynb
        |-- ML Pipeline Preparation.ipynb
    |-- README
    |-- requirements.txt
~~~~~~~


## **Usage**

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponseETL.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponseETL.db models/DisasterResponseModel.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage
