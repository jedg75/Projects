import joblib
import pandas as pd
import streamlit as st
import shap
import requests
import plotly.graph_objs as go
import json
import pickle
import streamlit.components.v1 as components
import plotly.figure_factory as ff
import plotly.express as px
import numpy as np

st.set_option('deprecation.showPyplotGlobalUse', False)

@st.cache_resource
def load_model(path, way):
    pickle_in = open(path, way)
    classifier_pipeline = pickle.load(pickle_in)
    return classifier_pipeline

classifier = load_model("classifier_lr_few.pkl","rb")
#classifier = joblib.load("classifier_lr_few.pkl")

# Load the preprocessed dataset
df = pd.read_csv("data_loan_few_features.csv")
model_params = classifier.get_params()

# Afficher les paramètres
print("Paramètres du modèle :")
for param, value in model_params.items():
    print(f"{param}: {value}")
#load JS vis in the notebook
shap.initjs()

@st.cache_data
def plot_shap_values(input_type = 'df', input_df = None, input_id = None):
    # plot the feature importance
    
    model_step = classifier.named_steps['model']
    preprocessed_data = classifier.named_steps['preprocessor'].transform(df)

    if input_type == 'df':
        selected_data = input_df
        numeric_features = selected_data.select_dtypes(
            include=['int64', 'float64'])

    else :
        selected_data = df.loc[df['SK_ID_CURR'] == input_id]
        numeric_features = selected_data.select_dtypes(include=['int64', 'float64']).drop(
            columns=['SK_ID_CURR','AMT_CREDIT','TARGET'])
    transformed_row = classifier.named_steps['preprocessor'].transform(selected_data)
    categorical_features = selected_data.select_dtypes(include=['object'])

    if hasattr(classifier.named_steps['preprocessor'].named_transformers_['cat'], 'get_feature_names_out'):
        column_names = classifier.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(
            input_features=categorical_features.columns)
    else:
        # If get_feature_names_out is not available, use the original column names
        column_names = categorical_features.columns

    explainer_water = shap.Explainer(model_step.predict, preprocessed_data)
    explainer_summary = shap.Explainer(model_step, preprocessed_data, feature_names=column_names)

    # Create a new DataFrame with the transformed data and column names
    preprocessed_row = pd.DataFrame(transformed_row, columns=numeric_features.columns.tolist() + column_names.tolist())
    shap_values_water = explainer_water(preprocessed_row)

    shap_values_summary = explainer_summary.shap_values(preprocessed_data)

    st.pyplot(shap.summary_plot(shap_values_summary, preprocessed_data, feature_names = numeric_features.columns.tolist() + column_names.tolist()))
    st.pyplot(shap.plots.waterfall(shap_values_water[0], max_display=10))

@st.cache_data
def plot_score(prediction):
    # Define the data for the stacked bar chart
    x_labels = ['Score']
    y_values_bin1 = [50]
    y_values_bin2 = [50]
    #y_values_bin3 = [10]

    # Define the value for the vertical line
    vertical_line_value = prediction*100

    bar_colors = ['#636EFA', '#FF9900']
    line_color = 'black'

    # Create stacked bar traces
    trace_bin1 = go.Bar(
        x=y_values_bin1,
        y=x_labels,
        orientation='h',
        name='Remboursement incertain',
        marker_color=bar_colors[1],

    )

    trace_bin2 = go.Bar(
        x=y_values_bin2,
        y=x_labels,
        orientation='h',
        name='Bon potentiel',
        marker_color=bar_colors[0],

    )

    # Create a layout for the graph (optional)
    layout = go.Layout(
        title='Analyse des seuils de remboursement',
        barmode='stack'
    )

    # Create a Figure and add the traces and layout
    fig = go.Figure(data=[trace_bin1, trace_bin2], layout=layout)

    fig.add_vline(x=vertical_line_value, line_dash="dot")

    # Show the graph
    st.plotly_chart(fig)

st.title("Modélisation")
st.write('#### Prediction du remboursement du prêt')

pages = ["Rechercher un client existant", "Effectuer une nouvelle prediction"]
page = st.sidebar.radio("Aller vers la page :", pages)

if page == pages[0]:

    st.write("Quel est l'ID du client pour la prédiction.")
    input_id = st.number_input('Entrer un ID client.', min_value=df.index.min(), max_value=df.index.max())
    st.write(f"Les ID vont de {df['SK_ID_CURR'].min()} à {df['SK_ID_CURR'].max()}.")
    pred_button = st.button("Effectuer la prediction.")

    if pred_button:

        request = requests.post(f'https://loan-score-dashboard-55640624b325.herokuapp.com/predict_proba/{input_id}')

        if request.status_code == 200:
            data = request.json()
            prediction = round(data['probabilité'], 2)

        row = df.loc[df['SK_ID_CURR'] == input_id]
        ##prediction = classifier.predict_proba(row)
        plot_shap_values(input_type = 'data',input_id = input_id)

        plot_score(prediction)
        st.write(f"Le score de prediction du modèle pour le client ID : {input_id} est {prediction*100}.")


if page == pages[1]:

    st.write("Remplisser le formulaire pour effectuer une prediction.")
    EXT_SOURCE_1 = st.number_input('Score source externe : EXT_SOURCE_1 ')
    EXT_SOURCE_2 = st.number_input('Score source externe : EXT_SOURCE_2')
    EXT_SOURCE_3 = st.number_input('Score source externe : EXT_SOURCE_3')
    AMT_CREDIT = st.number_input('Montant du crédit.')
    AMT_GOODS_PRICE = st.number_input('Prix des biens necessitant le prêt.')
    FLAG_OWN_CAR = st.selectbox('Voiture:', ['Y','N'])
    DAYS_EMPLOYED = st.number_input('Nombre de jour employé dans la position actuelle.')
    AMT_ANNUITY = st.number_input('Annuités.')
    NAME_EDUCATION_TYPE = st.selectbox('Niveau Education:', ['Academic degree','Higher education','Incomplete higher','Lower secondary','Secondary / secondary special'])
    DAYS_YEAR = st.number_input('Age du client.')
    DAYS_BIRTH = DAYS_YEAR*-365

    make_pred = st.button("Predict")

    if make_pred:

        dict = {
            'EXT_SOURCE_3': EXT_SOURCE_3,
            'EXT_SOURCE_2': EXT_SOURCE_2,
            'AMT_CREDIT': AMT_CREDIT,
            'EXT_SOURCE_1': EXT_SOURCE_1,
            'AMT_GOODS_PRICE': AMT_GOODS_PRICE,
            'AMT_ANNUITY': AMT_ANNUITY,
            'FLAG_OWN_CAR': FLAG_OWN_CAR,
            'NAME_EDUCATION_TYPE': NAME_EDUCATION_TYPE,
            'DAYS_EMPLOYED': DAYS_EMPLOYED,
            'DAYS_BIRTH': DAYS_BIRTH
        }

        request = requests.post(url = 'https://loan-score-dashboard-55640624b325.herokuapp.com/predict_new', data = json.dumps(dict))

        if request.status_code == 200:

            data = request.json()
            prediction = round(data['probabilité'], 2)
            input_data_df = pd.DataFrame(data['dataframe_dict'], index=[0]).drop(columns=["AMT_CREDIT"])

            print(input_data_df.columns)
            plot_shap_values(input_type ='df',input_df = input_data_df)
            plot_score(prediction)

            st.write(f"Le  score de prédiction du modèle pour le client est {prediction*100}.")
