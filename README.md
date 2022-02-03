

# Disaster Response Pipeline Project
---
### Summary 

<p>This project is based on the disaster data from Appen (formally Figure 8). 
In this project, I have made a text classification model using the scikit-learn library and then made a web app using Flask to deploy the model as a web app. 
</p>

---
### Instructions:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

---
### Files

1. app
    * This directory has the files needed to make the model a web app (the Flask code).
2. data
    * This directory contains the data used to build the model and the code to clean the data. 
3. models
    * This directory has the file that trains and saves the model. 
