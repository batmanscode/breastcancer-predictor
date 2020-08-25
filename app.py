import base64
import pickle
import warnings

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import FastICA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.tree import DecisionTreeClassifier
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive

# running model pickled in google colab

# running the locally generated pickle didn't work because:
# https://stackoverflow.com/questions/21033038/scikits-learn-randomforrest-trained-on-64bit-python-wont-open-on-32bit-python
# and
# https://stackoverflow.com/questions/27595982/how-to-save-a-randomforest-in-scikit-learn/27596667

# https://docs.streamlit.io/en/latest/api.html#streamlit.beta_set_page_config
# https://discuss.streamlit.io/t/version-0-65-0/4880
st.beta_set_page_config(page_title="Breast Cancer Predictor", page_icon="🔬", layout='centered', initial_sidebar_state='auto')

st.title('Breast Cancer Prediction Using Artificial Intelligence 🤖')

'''
This web app uses machine learning to predict whether a person has breast cancer using some of their clinical data.

❗ **Not a diagnostic tool**  
This is just a demo application of machine learning.

🚧 **Limitations**  
The dataset used to train the model is small and there might be better variables that could have been used.

*Original dataset available here: [Breast Cancer Coimbra](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Coimbra)*

*Details are described in: [Patrício, M., Pereira, J., Crisóstomo, J. et al. Using Resistin, glucose, age and BMI to predict the presence of breast cancer. BMC Cancer 18, 29 (2018)](https://doi.org/10.1186/s12885-017-3877-1)*

'''


#load data
@st.cache
def load_data():
    return pd.read_csv("cleaned_data.csv")

data = load_data()

st.write('###### *click to show/hide')
if st.checkbox('Data'):
    '''
    ## Data Used for Training
    '''

    data
    
    st.write('###### ' + str(data.shape[0]) + ' rows and ' + str(data.shape[1]) + ' columns.')

#dowload cleaned data
    #using the function from https://github.com/dataprofessor/basketball-heroku/blob/fe1e45eb1a238737e50adb8bd685bf54d1171642/basketball_app.py#L48
    #original discussion https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
    def filedownload(df):
        csv = data.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
        href = f'<a href="data:file/csv;base64,{b64}" download="cleaned_data.csv">Download as CSV</a>'
        return href

    st.markdown(filedownload(data), unsafe_allow_html=True)

    st.write('This is the cleaned data, please see the link to the dataset source to see raw data')
    st.write('Under the classification column: 0=healthy, 1=patient with cancer')


#tiny bit of data viz
    st.header('Feature Correlations')

    corr_features = data.drop('Classification', axis=1).apply(lambda x: x.corr(data['Classification']))
    corr_df = pd.DataFrame(corr_features.sort_values(ascending=False), columns=['correlation to breast cancer classification'])
    st.table(corr_df)
    '''
    This table shows the correlation between the blood test results and whether or not a person has cancer
    '''
    st.markdown('---')


@st.cache(allow_output_mutation=True) #added 'allow_output_mutation=True' because kept getting 'CachedObjectMutationWarning: Return value of load_model() was mutated between runs.'
def load_model():
    return pickle.load(open('model.pkl', 'rb'))

model = load_model()

#show evaluation metrics
if st.checkbox('Model Metrics'):

    #split the same way as used to train (from cleaned_data_pipeline.py)
    features = data.drop('Classification', axis=1)
    training_features, testing_features, training_target, testing_target = \
                train_test_split(features, data['Classification'], random_state=42, test_size=0.2)

    y_pred = model.predict(testing_features)
    y_true = testing_target

    accuracy = accuracy_score(y_true, y_pred, normalize=True)*100
    confusion_matrix(y_true, y_pred)

    # extracting true_positives, false_positives, true_negatives, false_negatives
    tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred).ravel()

    tnr = tn/(tn+fn)*100
    fpr = fp/(tp+fp)*100
    fnr = fn/(tn+fn)*100
    tpr = tp/(fp+tp)*100

    '''# 📊'''
    st.write(f"Accuracy: {accuracy:.2f}%")
    st.write(f"True negative rate: {tnr:.2f}%")
    st.write(f"False Positive rate: {fpr:.2f}%")
    st.write(f"False Negative rate: {fnr:.2f}%")
    st.write(f"True Positive rate: {tpr:.2f}%")

    
st.markdown('---')

st.title("Get a Prediction🔮")

'''
###### Note: Use this on Desktop for the best experience. If you're on mobile, use the arrow button on the upper left corner to open the sidebar and access input methods.
'''

#selectbox section starts
###################################################################################################

input_type = st.sidebar.selectbox('Input Method', ['Move Sliders', 'Enter Values'], index=1)

if input_type == 'Enter Values': #display text input fields, show user input, submit button

    #number input fields for features

    #format="%.3f rfom https://discuss.streamlit.io/t/st-number-input-formatter-displaying-blank-input-box/1217
    BMI = st.sidebar.number_input('BMI (kg/m2)', format="%.4f", step=0.0001)
    Glucose = st.sidebar.number_input('Glucose (mg/dL)', format="%.0f")
    Insulin = st.sidebar.number_input('Insulin (µU/mL)', format="%.4f", step=0.0001)
    HOMA = st.sidebar.number_input('HOMA', format="%.4f",step=0.0001)
    Resistin = st.sidebar.number_input('Resistin (ng/mL)', format="%.4f", step=0.0001)

    st.sidebar.info(
    '''
    💡 **Tip:**
    Change input to "move sliders" to get a feel for how the model thinks. Play around with the sliders and watch the predictions change.
    '''
    )

    # show user input
    '''
    ## These are the values you entered
    '''
    f"**BMI**: {BMI:.4f} kg/m2"
    f"**Glucose**: {Glucose:.0f} mg/dL"
    f"**Insulin**: {Insulin:.4f} µU/mL"
    f"**HOMA**: {HOMA:.4f}"
    f"**Resistin**: {Resistin:.4f} ng/mL"

    #button to create new dataframe with input values
    if st.button("submit ✅"):
        dataframe = pd.DataFrame(
            {'BMI':BMI,
            'Glucose':Glucose,
            'Insulin':Insulin,
            'HOMA':HOMA,
            'Resistin ':Resistin }, index=[0]
        )

if input_type == 'Move Sliders': #display slider input fields

    BMI = st.sidebar.slider('BMI (kg/m2)',
                min_value=10.0,
                max_value=50.0,
                value=float(data['BMI'][0]),
                step=0.01)

    Glucose = st.sidebar.slider('Glucose (mg/dL)',
                min_value=25,
                max_value=250,
                value=int(data['Glucose'][0]),
                step=1)
    # i kept getting an error so i just used int() after about an hour or so of frustration
    # fyi before applying int(), value was <class 'numpy.int64'>
    # the error:
    # StreamlitAPIException: Slider value should either be an int/float or a list/tuple of 0 to 2 ints/floats

    Insulin = st.sidebar.slider('Insulin (µU/mL)',
                        min_value=1.0,
                        max_value=75.0,
                        value=float(data['Insulin'][0]),
                        step=0.01)
    
    HOMA = st.sidebar.slider('HOMA',
                        min_value=0.25,
                        max_value=30.0,
                        value=float(data['HOMA'][0]),
                        step=0.01)

    Resistin = st.sidebar.slider('Resistin (ng/mL)',
                        min_value=1.0,
                        max_value=100.0,
                        value=float(data['Resistin'][0]),
                        step=0.01)
    # got KeyError: <class ‘numpy.float64’> and had to add float()
    # https://discuss.streamlit.io/t/keyerror-class-numpy-float64/5147
    
    #slider values to dataframe
    dataframe = pd.DataFrame(
        {'BMI':BMI,
        'Glucose':Glucose,
        'Insulin':Insulin,
        'HOMA':HOMA,
        'Resistin ':Resistin }, index=[0]
    )

    '''
    ## Move the Sliders to Update Results ↔
    '''

#selectbox section ends
###################################################################################################


try:
    '''
    ## Results 📋
    '''

    if model.predict(dataframe)==0:
        st.write('Prediction: **NO BREAST CANCER 🙌**')
    else:
        st.write('Prediction: **BREAST CANCER PRESENT**')

    for healthy, cancer in model.predict_proba(dataframe):
        
        healthy = f"{healthy*100:.2f}%"
        cancer = f"{cancer*100:.2f}%"

        st.table(pd.DataFrame({'healthy':healthy,
                                'has breast cancer':cancer}, index=['probability']))

    if input_type == 'enter values':
        st.success('done 👍')
    else:
        pass

    import warnings
    warnings.filterwarnings('ignore')

except:
    st.write('*press submit to compute results*')
    # st.write("if you see this, it finally worked :O")

st.markdown('---')

st.info(
'''
Source code available [here](https://github.com/batmanscode/breastcancer-predictor), please feel free to leave feedback and contribute 😊
'''
)

# hide hamburger menu
# https://discuss.streamlit.io/t/remove-made-with-streamlit-from-bottom-of-app/1370/2
# https://github.com/streamlit/streamlit/issues/395#issuecomment-579526417

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)
