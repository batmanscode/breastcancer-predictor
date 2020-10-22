# Breast Cancer Prediction Using Artificial Intelligence 🤖

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/batmanscode/breastcancer-predictor/app.py)

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

Details described in: [Patrício, M., Pereira, J., Crisóstomo, J. et al. Using Resistin, glucose, age and BMI to predict the presence of breast cancer. BMC Cancer 18, 29 (2018)](https://doi.org/10.1186/s12885-017-3877-1)

### Demo
![demo](https://github.com/batmanscode/breastcancer-predictor/blob/master/demo.gif)

## How to Run Locally
Clone this repository, create a new environment, and enter the following in your terminal:
```shell
streamlit run app.py
```
This will create a local web server which should open in your default browser. If not just use one of the links returned in your terminal.

## Additional Info
* Web app built with [Streamlit](https://github.com/streamlit/streamlit) (which is an amazing tool, if you haven't heard of it already).
* If you're using conda, you can use `environment.yml` to create a new environment.
* `pipeline.py` is the code used to generate `model.pkl`. [TPOT](https://github.com/EpistasisLab/tpot) was used to create this pipeline.

## Possible Issues
If your cloned version gives you an error, please generate a new `model.pkl` by running `pipeline.py` and try again. This will overwrite the existing `model.pkl`.

I ran into ```ValueError: Buffer dtype mismatch, expected 'SIZE_t' but got 'int'``` when trying to deploy on Heroku despite it working in my local env. Turns out you can't pickle on one architecture and unpickle on a different one (32/64-bit). Thanks to [this answer](https://stackoverflow.com/questions/27595982/how-to-save-a-randomforest-in-scikit-learn/27596667) for helping me understand what the issue was.
