# Disaster Response Pipeline Project

## Summary
This project uses Natural Language Processing (NLP) to categorize natural disaster related messages from news, social media, and direct reports into 36 non-exclusive categories. ETL pipeline is used for cleaning and storing data in an SQLite database. ML pipeline is used to train multi-output (milti-label) classification model. Flask-based web application allows users to input custom messages for categorization using model pre-trained on **Figure Eight** / **Udacity** disaster response messages training data set. 

![](resources/main_page_crop.PNG?raw=true)

## Methods
Message classification pipeline consistes of message text pre-processing with `spacy` NLP package to produce normalized and lemmatized using correct part of speech word tokens. The tokens are fed into TF-IDF transformer and then into machine learning classifier.

Initial classifier model (see discussion below) utilizes a very simplistic approach of Complement Naive Bayees Classifier, chosen for its preferred usage with imbalanced classes, wrapped in a MultiOutputClassifier for multi-label classification. Model selection was performed with 5-fold GridSearch cross-validation using Hamming Loss minimazation for scoring.
More sophisticated approach uses custom ImbalancedMultiOutputClassifier class that allows separate over- and under-sampling for each target output in multi-output classfication problem (tested SMOTE and ADASYN).

## Discussion
Initial simplistic model performance is sub-par, especially for highly imbalanced output targets, *as expected*. There are multiple reasons for poor model performance:
- The training dataset contains highly imbalanced classes. For example, there are no messages in the training dataset labeled *Child Alone*, and multiple categories contain less than 1% of positively labeled samples. As a result, random cross-validation fold and test subset splits do not accurately represent all class labels in the subsets, leading to both poor model training and evaluation.
- Hamming Loss, often used for [multi-label classification](https://en.wikipedia.org/wiki/Multi-label_classification) evaluation, in our case improved model precision, at the cost of recall. However, since classification of messages is likely to be used to provide relief during natural disasters, it is more important to alert an appropriate agency of potential need and not miss any categories even if we create a false alert, rather than not place a message in a category and not alerting of a need when there is one. Thus, recall is more important than precision in our case.

Incorporation of per-output class balancing over-sampling into the training process resulted in drastic double-digit increase of recall for highly imbalanced classes. Exploratory results show that a simple classifier combined with over-sampling can outperformed more complex classifiers in improvement of recall for highly class-imbalanced outputs. Examples of the more pronounced improvements can be seen below:

RECALL SCORES:

```
message class higly | ComplementNB | SMOTE + ComplementNB |   RF     | RF + SMOTE |
imbalanced classes  |              |                      |          |            |
-----------------------------------------------------------------------------------
medical_help        |    0.12      |    **0.65**          |    0.06  |    0.27    |
medical_products    |    0.10      |    **0.69**          |    0.07  |    0.29    |
search_and_rescue   |    0.02      |    **0.53**          |    0.02  |    0.07    |
security            |    0.00      |    **0.41**          |    0.00  |    0.02    |
military            |    0.16      |    **0.82**          |    0.07  |    0.24    |
water               |    0.11      |    **0.81**          |    0.37  |    0.60    |
...
-----------------------------------------------------------------------------------
OVERALL:
micro avg           |    0.51      |    **0.77**          |    0.54  |    0.60    |
macro avg           |    0.17      |    **0.62**          |    0.21  |    0.29    |
weighted avg        |    0.51      |    **0.77**          |    0.54  |    0.60    |
samples avg         |    0.48      |    **0.59**          |    0.48  |    0.50    |
```

It is important to note that while SMOTE significantly improved classification recall, aggressive synthetic oversampling resulted in reduced classification precision. Further sampling and model selection optimization is necessarry, perhaps using F1 scoring to maintain balance between precision and recall.

## Model Improvement Next Steps
- Sampling and model optimization


## Acknowledgements
- Template code and notebooks are provided by **Udacity**
- Training data set is provided by **Figure Eight**

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
