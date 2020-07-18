# Breast Cancer Prediction Using Machine Learning 🤖

Live at: https://breastcancer-predictor.herokuapp.com/

This web app uses machine learning to predict whether a person has breast cancer using some of their clinical data.

It takes the following input:
* BMI (kg/m2)
* Glucose (mg/dL)
* Insulin (µU/mL)
* HOMA
* Resistin

And returns either "**no breast cancer**" or "**breast cancer present**" along with their probabilities.

Dataset used: [Breast Cancer Coimbra](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Coimbra)

## Additional Info
* Web app built with [Streamlit](https://github.com/streamlit/streamlit) (which is an amazing tool, if you haven't heard of it already).
* If you're using conda, ```environment.yml``` is for you.
* ```pipeline.py``` is the code used to generate ```model.pkl```. [TPOT](https://github.com/EpistasisLab/tpot) was used to create this pipeline.
