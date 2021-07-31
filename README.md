# Happy Customer Bank
## Machine Learning prediction if a client should get disburse
---
---

## Project Structure

### Scripts
- ML_project.py - MachineLearning script responsible for data processing and learning ML models
- vis.py - streamlit script (streamlit run vis.py)

### Packages
- data - train/test data
- machine_learning - package with modules required to run machine learning script (ML_project.py)
- models - trained models saved as *.joblib
- streamlit_app - package with modules required to run streamlit app (vis.py)
- utils - package with universal modules

---

## Instructions

### General 
1. Install libraries from requirements.txt

### Model training 
1. To train models run ML_project.py
2. Follow the steps, if you are not satisfied with the default models scores, to generate models based on different params:
  - change/add new params, in classifiers->params(dict) from machine_learning.estimators, for each estimator and re-run ML_project.py
  - when you are satisfied with the scores acquired on you params, delete scores.csv and all saved models in "models" directory **(leave the "best_trained_models" directory)**
  - run ML_project.py for the last time to save your best models and their scores in "models" directory -> copy-paste them to "best_trained_models" __(This step is required for the Streamlit app to work properly)__
  - also note that there should be only one occurence of the score for each estimator
  



### Streamlit application for 
