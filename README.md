

# Disaster Response Pipeline Project
---
### Summary 

<p>This project is based on the disaster data from Appen (formally Figure 8). 
In this project, I have made a text classification model using the scikit-learn library and then made a web app using Flask to deploy the model as a web app. 
</p>
<p>
In the event of an emergency this project can help filter out related or unrelated message, and this is going to help first responders, and emergency organization on helping people who send realted messages. 

For example: When an emargency occurs there will be alot of tweets about it and not all of them are relavent (as in the person who tweeted needs help) therefore this project can help emergency organization at identifying the people who need help to help them. 
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
### Files structure
<pre>
.<br>
├── app<br>
│   ├── run.py<br>
│   └── templates<br>
│       ├── cloud1.png<br>
│       ├── cloud.png<br>
│       ├── general.css<br>
│       ├── go_copy.html<br>
│       ├── go.html<br>
│       ├── master_copy.html<br>
│       └── master.html<br>
├── data<br>
│   ├── disaster_categories.csv<br>
│   ├── disaster_messages.csv<br>
│   ├── DisasterResponse.db<br>
│   ├── process_data.py<br>
│   └── YourDatabaseName.db<br>
├── models<br>
│   ├── classifier.pkl<br>
│   └── train_classifier.py<br>
├── Procfile<br>
├── README.md<br>
└── requirements.txt<br>
</pre>
---
### Files explained
1. run.py: used to the backend server that is running the wep app. 
2. general.css: css code for the htnl pages. 
3. master.html: the main page that showing for the web app. 
4. go.html: the html page after submitting a classify query. 
5. process_data.py: NLP piprline, that cleans and merges two databases. 
6. train_classifier.py: building and training the model. 
