# Disaster Response Pipeline Project

## Summary
This project uses Natural Language Processing (NLP) to categorize natural disaster related messages from news, social media, and direct reports into 36 non-exclusive categories. ETL pipeline is used for cleaning and storing data in an SQLite database. ML pipeline is used to train multi-output (milti-label) classification model. Flask-based web application allows users to input custom messages for categorization using model pre-trained on **Figure Eight** / **Udacity** disaster response messages training data set. 

![](resources/main_page_crop.PNG?raw=true)

## Methods
Message classification pipeline consistes of message text pre-processing with `spacy` NLP package to produce normalized and lemmatized using correct part of speech word tokens. The tokens are fed into TF-IDF transformer and then into machine learning classifier.

Initial classifier model (see discussion below) utilizes a very simplistic approach of Complement Naive Bayees Classifier, chosen for its preferred usage with imbalanced classes, wrapped in a MultiOutputClassifier for multi-label classification. Model selection was performed with 5-fold GridSearch cross-validation using Hamming Loss minimazation for scoring.

## Discussion
Initial simplistic model performance is sub-par, *as expected*. There are multiple reasons for poor model performance:
- The training dataset contains highly imbalanced classes. For example, there are no messages in the training dataset labeled *Child Alone*, and multiple categories contain less than 1% of positively labeled samples. As a result, random cross-validation fold and test subset splits do not accurately represent all class labels in the subsets, leading to both poor model training and evaluation.
- Hamming Loss, often used for [multi-label classification](https://en.wikipedia.org/wiki/Multi-label_classification) evaluation, in our case improved model precision, at the cost of recall. However, since classification of messages is likely to be used to provide relief during natural disasters, it is more important to alert an appropriate agency of potential need and not miss any categories even if we create a false alert, rather than not place a message in a category and not alerting of a need when there is one. Thus, recall is more important than precision in our case.

## Model Improvement Next Steps
- Address class imbalance
  - Use Stratified subsetting when splitting data in cv-folds and train/test subsets
  - Integrate re-sampling (SMOTE, ADASYN) into training pipeline
- Use recall or f1-score model evaluation instead of Hamming Loss

---

## Requirements
Application uses Python 3.6+ with packages listed in `requirements.txt`

## Project structure
```
|-- notebooks/  
    |-- `Training Data Exploration`: notebook exploring the training dataset  
    |-- `ETL Pipeline Preparation.ipynb`: notebook detailing pipeline ETL steps  
    |-- `ETL Pipeline Preparation.ipynb`: notebook detailing pipeline ML steps  
|-- data/  
    |-- `disaster_categories.csv`: training data categories for each message  
    |--  `disaster_messages.csv`: training data message with id's corresponding to those in `disaster_categories.csv`  
    |--  `DisasterResponse.db`: clean data pre-processed for classification task and stored in SQLite database  
    |--  `process_data.py`: ETL pipeline module for data loading, cleaning, and storing data  
|-- models/  
    |-- `train_classifier.py`: ML pipeline module for loading data and training, evaluating and storing pre-trained model in .pkl file  
|-- app/  
    |-- templates/  
        |-- `go.html`:   
        |-- `master.html`:  
    |-- `run.py`: Flask application module for running web-app  
```  
Note: due to large size *.pkl models are not provided in this repository.

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the projects root directory to run your web app
    `python app/run.py`

3. Application webpage can be found on your local machine by going to http://127.0.0.1:5000/
