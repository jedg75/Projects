import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests

data0 = pd.read_csv("data_loan_few_features.csv")

df_no_na = data0.dropna()

@st.cache_data
def plot_bivarie(x, y, title="Scatter Plot", x_label="X-axis", y_label="Y-axis"):
    """
    Plot a scatter plot using Plotly Express.

    Parameters:
    - x: Pandas Series or NumPy array, x-axis values.
    - y: Pandas Series or NumPy array, y-axis values.
    - title: String, title of the plot (default is "Scatter Plot").
    - x_label: String, label for the x-axis (default is "X-axis").
    - y_label: String, label for the y-axis (default is "Y-axis").
    """
    # Create a DataFrame with the given variables
    df = pd.DataFrame({x.name: x, y.name: y})

    # Create a scatter plot
    fig = px.scatter(df, x=x.name, y=y.name, title=title, labels={x.name: x_label, y.name: y_label})

    # Display the plot
    st.plotly_chart(fig)

st.title("Analyse des variables")

st.markdown("Explorez les variables pour comprendre comment elles évoluent entre elles et traduisent le comportement des consommateurs. "
            "En étudiant les resultats, nous pouvons comprendre comment le classificateur prend ses décisions de classification des données.")

st.sidebar.header("Etude des différentes variables")

options = st.sidebar.radio("Choix de la variable",
                           options=["Remboursement du pret",
                                        "Montant du crédit",
                                    "Prix des biens",
                                    "Ancienneté",
                                    "Annuités",
                                    "Niveau d'education",
                                    "Age",
                                    "Analyse bivariée"
                                    ])

request = requests.get(url = 'https://loan-score-dashboard-55640624b325.herokuapp.com/predict_new')
data = request.json()
if request.status_code == 200:
        data = request.json()
else:
        st.write("Il faut renseigner les informations d'un nouveau client")

if options == "Remboursement du pret":
        st.write("#### Etude de la cible : remboursement de prêt.")
        st.write("0 = Prêt remboursé.")
        st.write("1 = Prêt non remboursé.")

        fig_target = go.Figure()
        fig_target = px.histogram(data0, x='TARGET', color='TARGET', color_discrete_sequence=["#EF553B", "#636EFA"])
        fig_target.update_layout(title='Distribution des remboursement')
        fig_target.update_layout(bargap=0.2)

        st.plotly_chart(fig_target)

if options == "Age":

        age_customer = data['dataframe_dict']['DAYS_BIRTH'] / -365

        st.write('#### Distribution par Age')
        st.write("Etude du remboursement de prêt en fonction de la variable age.")

        fig1 = go.Figure()
        fig1 = px.histogram(data0, x=-data0['DAYS_BIRTH'] / 365, nbins=25, labels={'x': 'Age (Years)', 'y': 'Count'})
        fig1.update_layout(title='Repartition par age des clients', xaxis_title='Age (Année)', yaxis_title='Quantité (en milliers)')
        fig1.add_vline(x=age_customer, line_dash="dot", name = 'client')
        fig1.update_layout(bargap=0.2)

        st.write("0 = Prêt remboursé.")
        st.write("1 = Prêt non remboursé.")
        fig2 = go.Figure()
        # Create a histogram for 'target == 0'
        hist0, edges0 = np.histogram(data0.loc[data0['TARGET'] == 0, 'DAYS_BIRTH'] / -365, bins=25, density = True)
        fig2.add_trace(go.Scatter(x=edges0, y=hist0, name='target == 0'))

        # Create a histogram for 'target == 1'
        hist1, edges1 = np.histogram(data0.loc[data0['TARGET'] == 1, 'DAYS_BIRTH'] / -365, bins=25, density = True)
        fig2.add_trace(go.Scatter(x=edges1, y=hist1, name='target == 1'))
        fig2.add_vline(x=age_customer, line_dash="dot", name = 'client')

        fig2.update_layout(
                title="Remboursement du pret en fonction de la variable age",
                xaxis_title='Age (Année)',
                yaxis_title='Quantité (en % de la totalité)')

        st.plotly_chart(fig1)
        st.plotly_chart(fig2)

if options == "Prix des biens":

        st.write("Etude du remboursement de prêt en fonction du prix des biens.")

        target_1_data = df_no_na[df_no_na['TARGET'] == 1]['AMT_GOODS_PRICE']
        target_0_data = df_no_na[df_no_na['TARGET'] == 0]['AMT_GOODS_PRICE']
        customer_amt_goods_price = data['dataframe_dict']['AMT_GOODS_PRICE']

        fig1 = go.Figure()
        fig1 = px.histogram(df_no_na, x=-df_no_na['AMT_GOODS_PRICE']*(-1), nbins=25, labels={'x': 'price', 'y': 'Count'}, color_discrete_sequence= ["#FF97FF"])
        fig1.update_layout(title='Repartition par prix', xaxis_title='price', yaxis_title='Quantité (en milliers)')
        fig1.add_vline(x=customer_amt_goods_price, line_dash="dot", name = 'client')
        fig1.update_layout(bargap=0.2)

        st.write("0 = Prêt remboursé.")
        st.write("1 = Prêt non remboursé.")
        fig2 = go.Figure()
        # Create a histogram for 'target == 0'
        hist0, edges0 = np.histogram(df_no_na.loc[df_no_na['TARGET'] == 0, 'AMT_GOODS_PRICE'], bins=25, density = True)
        fig2.add_trace(go.Scatter(x=edges0, y=hist0, name='target == 0',line=dict(color="#FF97FF")))

        # Create a histogram for 'target == 1'
        hist1, edges1 = np.histogram(df_no_na.loc[df_no_na['TARGET'] == 1, 'AMT_GOODS_PRICE'], bins=25, density = True)
        fig2.add_trace(go.Scatter(x=edges1, y=hist1, name='target == 1',line=dict(color="rgb(253,218,236)")))

        fig2.update_layout(
                title="Remboursement du pret en fonction de la variable AMT_GOODS_PRICE",
                xaxis_title='Prix',
                yaxis_title='Quantité (en % de la totalité)')

        st.write(f"{data['dataframe_dict']['AMT_GOODS_PRICE']}")

        fig2.add_vline(x=customer_amt_goods_price, line_dash="dot", name = 'client')

        st.plotly_chart(fig1)
        st.plotly_chart(fig2)

if options == "Ancienneté":
        st.write("Etude du remboursement de prêt en fonction de lancienneté.")

        target_1_data = df_no_na[df_no_na['TARGET'] == 1]['DAYS_EMPLOYED']*(-1)
        target_0_data = df_no_na[df_no_na['TARGET'] == 0]['DAYS_EMPLOYED']*(-1)
        customer_days_employed = data['dataframe_dict']['DAYS_EMPLOYED']

        fig1 = go.Figure()
        fig1 = px.histogram(df_no_na, x=-df_no_na['DAYS_EMPLOYED'], nbins=25,
                            labels={'x': 'jour', 'y': 'Count'})

        fig1.add_vline(x=customer_days_employed, line_dash="dot", name = 'client')

        fig1.update_layout(title='Repartition par jour', xaxis_title='jour', yaxis_title='Quantité (en milliers)')
        fig1.update_layout(bargap=0.2)

        st.write("0 = Prêt remboursé.")
        st.write("1 = Prêt non remboursé.")
        fig2 = go.Figure()
        # Create a histogram for 'target == 0'
        hist0, edges0 = np.histogram(df_no_na.loc[df_no_na['TARGET'] == 0, 'DAYS_EMPLOYED'], bins=25, density=True)
        fig2.add_trace(go.Scatter(x=edges0, y=hist0, name='target == 0'))

        # Create a histogram for 'target == 1'
        hist1, edges1 = np.histogram(df_no_na.loc[df_no_na['TARGET'] == 1, 'DAYS_EMPLOYED'], bins=25, density=True)
        fig2.add_trace(go.Scatter(x=edges1, y=hist1, name='target == 1'))

        fig2.update_layout(
                title="Remboursement du pret en fonction de la variable DAYS_EMPLOYED",
                xaxis_title='Jour',
                yaxis_title='Quantité (en % de la totalité)')

        fig2.add_vline(x=customer_days_employed, line_dash="dot", name = 'client')

        st.plotly_chart(fig1)
        st.plotly_chart(fig2)

if options == "Annuités":
        st.write("Etude du remboursement de prêt en fonction des annuités.")

        target_1_data = df_no_na[df_no_na['TARGET'] == 1]['AMT_ANNUITY']
        target_0_data = df_no_na[df_no_na['TARGET'] == 0]['AMT_ANNUITY']
        customer_annuity = data['dataframe_dict']['AMT_ANNUITY']

        fig1 = go.Figure()
        fig1 = px.histogram(df_no_na, x=-df_no_na['AMT_ANNUITY'] * (-1), nbins=25,
                            labels={'x': 'montant', 'y': 'Count'})
        fig1.update_layout(title='Repartition par montant', xaxis_title='montant', yaxis_title='Quantité (en milliers)')
        fig1.add_vline(x=customer_annuity, line_dash="dot", name = 'client')

        fig1.update_layout(bargap=0.2)

        st.write("0 = Prêt remboursé.")
        st.write("1 = Prêt non remboursé.")
        fig2 = go.Figure()
        # Create a histogram for 'target == 0'
        hist0, edges0 = np.histogram(df_no_na.loc[df_no_na['TARGET'] == 0, 'AMT_ANNUITY'], bins=25, density=True)
        fig2.add_trace(go.Scatter(x=edges0, y=hist0, name='target == 0'))

        # Create a histogram for 'target == 1'
        hist1, edges1 = np.histogram(df_no_na.loc[df_no_na['TARGET'] == 1, 'AMT_ANNUITY'], bins=25, density=True)
        fig2.add_trace(go.Scatter(x=edges1, y=hist1, name='target == 1'))
        fig2.add_vline(x=customer_annuity, line_dash="dot", name = 'client')

        fig2.update_layout(
                title="Remboursement du pret en fonction de la variable AMT_ANNUITY",
                xaxis_title='Montant',
                yaxis_title='Quantité (en % de la totalité)')

        st.plotly_chart(fig1)
        st.plotly_chart(fig2)

if options == "Montant du crédit":
        st.write("Etude du remboursement de prêt en fonction des annuités.")

        target_1_data = df_no_na[df_no_na['TARGET'] == 1]['AMT_CREDIT']
        target_0_data = df_no_na[df_no_na['TARGET'] == 0]['AMT_CREDIT']
        customer_amt_credit = data['dataframe_dict']['AMT_CREDIT']

        fig1 = go.Figure()
        fig1 = px.histogram(df_no_na, x=-df_no_na['AMT_CREDIT'] * (-1), nbins=25,
                            labels={'x': 'montant', 'y': 'Count'})
        fig1.add_vline(x=customer_amt_credit, line_dash="dot", name = 'client')
        fig1.update_layout(title='Repartition par montant', xaxis_title='montant', yaxis_title='Quantité (en milliers)')
        fig1.update_layout(bargap=0.2)

        st.write("0 = Prêt remboursé.")
        st.write("1 = Prêt non remboursé.")
        fig2 = go.Figure()
        # Create a histogram for 'target == 0'
        hist0, edges0 = np.histogram(df_no_na.loc[df_no_na['TARGET'] == 0, 'AMT_CREDIT'], bins=25, density=True)
        fig2.add_trace(go.Scatter(x=edges0, y=hist0, name='target == 0'))

        # Create a histogram for 'target == 1'
        hist1, edges1 = np.histogram(df_no_na.loc[df_no_na['TARGET'] == 1, 'AMT_CREDIT'], bins=25, density=True)
        fig2.add_trace(go.Scatter(x=edges1, y=hist1, name='target == 1'))

        fig2.add_vline(x=customer_amt_credit, line_dash="dot", name = 'client')

        fig2.update_layout(
                title="Remboursement du pret en fonction de la variable AMT_CREDIT",
                xaxis_title='Montant',
                yaxis_title='Quantité (en % de la totalité)')

        st.plotly_chart(fig1)
        st.plotly_chart(fig2)

if options == "Niveau d'education":

        st.write("Etude du remboursement de prêt en fonction du niveau d'études.")

        st.write("0 = Prêt remboursé.")
        st.write("1 = Prêt non remboursé.")

        df_no_na = data0.dropna()

        grouped_df = df_no_na.groupby(['NAME_EDUCATION_TYPE', 'TARGET']).size().reset_index(name='count')

        # Pivot the DataFrame to have "TARGET" values as columns
        pivoted_df = grouped_df.pivot(index='NAME_EDUCATION_TYPE', columns='TARGET', values='count').fillna(0)

        # Calculate the total count for each "NAME_EDUCATION_TYPE" and sort in ascending order
        pivoted_df['total'] = pivoted_df.sum(axis=1)

        sorted_df = pivoted_df.drop(columns='total')

        # Normalize values based on "TARGET"
        normalized_df = sorted_df.div(sorted_df.sum(axis=1), axis=0).sort_values(by=1, ascending=True)

        # Reset the index for a cleaner DataFrame
        normalized_df.reset_index(inplace=True)
        colors = {'0': 'lightblue', '1': 'orange'}

        # Create a stacked bar chart with Plotly Express
        fig = px.bar(normalized_df,
                     x="NAME_EDUCATION_TYPE",
                     y=sorted_df.columns,  # Exclude the 'total' column
                     title="Type d'education Stacked Bar Chart en pourcentage par type d'education",
                     labels={'TARGET': 'Education Type'},
                     color_discrete_map=colors,
                     height=400)
        st.plotly_chart(fig)

if options == "Analyse bivariée":

        st.markdown("Selectionnez une ou plusieurs variables à analyser:")
        variable_1 = st.selectbox('Variable 1:', ["Montant du crédit", "Annuités", "Ancienneté", "Prix des biens", "Age"])
        variable_2 = st.selectbox('Variable 2:', ["Montant du crédit", "Annuités", "Ancienneté", "Prix des biens", "Age"])
        variable_dictionary = {"Montant du crédit" : "AMT_CREDIT", "Annuités": "AMT_ANNUITY",
                                "Ancienneté": "DAYS_EMPLOYED","Prix des biens": "AMT_GOODS_PRICE", "Age": 'DAYS_BIRTH'}

        anciennete = df_no_na[variable_dictionary["Ancienneté"]] * (-1)
        age = df_no_na[variable_dictionary["Age"]] / -365

        if variable_1 == variable_2:
                st.write("Veuillez choisir 2 variables différentes.")

        if variable_1 == "Montant du crédit":
                if variable_2 == "Age":
                        plot_bivarie(df_no_na[variable_dictionary[variable_1]],
                                     age, title="Graphe d'analyse bivariée",
                                     x_label=f"{variable_1}", y_label=f"{variable_2}")
                if variable_2 == "Ancienneté":
                        plot_bivarie(df_no_na[variable_dictionary[variable_1]],
                                     anciennete, title="Graphe d'analyse bivariée",
                                     x_label=f"{variable_1}", y_label=f"{variable_2}")
                else:
                        plot_bivarie(df_no_na[variable_dictionary[variable_1]], df_no_na[variable_dictionary[variable_2]], title="Graphe d'analyse bivariée",
                                     x_label=f"{variable_1}", y_label=f"{variable_2}")

        if variable_1 == "Annuités":
                if variable_2 == "Age":
                        plot_bivarie(df_no_na[variable_dictionary[variable_1]],
                                     age, title="Graphe d'analyse bivariée",
                                     x_label=f"{variable_1}", y_label=f"{variable_2}")
                if variable_2 == "Ancienneté":
                        plot_bivarie(df_no_na[variable_dictionary[variable_1]],
                                     anciennete, title="Graphe d'analyse bivariée",
                                     x_label=f"{variable_1}", y_label=f"{variable_2}")
                else:
                        plot_bivarie(df_no_na[variable_dictionary[variable_1]], df_no_na[variable_dictionary[variable_2]], title="Graphe d'analyse bivariée",
                                     x_label=f"{variable_1}", y_label=f"{variable_2}")

        if variable_1 == "Prix des biens":
                if variable_2 == "Age":
                        plot_bivarie(df_no_na[variable_dictionary[variable_1]],
                                     age, title="Graphe d'analyse bivariée",
                                     x_label=f"{variable_1}", y_label=f"{variable_2}")
                if variable_2 == "Ancienneté":
                        plot_bivarie(df_no_na[variable_dictionary[variable_1]],
                                     anciennete, title="Graphe d'analyse bivariée",
                                     x_label=f"{variable_1}", y_label=f"{variable_2}")
                else:
                    plot_bivarie(df_no_na[variable_dictionary[variable_1]], df_no_na[variable_dictionary[variable_2]],
                                 title="Graphe d'analyse bivariée", x_label=f"{variable_1}", y_label=f"{variable_2}")

        if variable_1 == "Age":
                if variable_2 == "Ancienneté":
                        plot_bivarie(age,
                                     anciennete, title="Graphe d'analyse bivariée",
                                     x_label=f"{variable_1}", y_label=f"{variable_2}")
                else :
                        plot_bivarie(age, df_no_na[variable_dictionary[variable_2]], title="Graphe d'analyse bivariée",
                                     x_label=f"{variable_1}", y_label=f"{variable_2}")

        if variable_1 == "Ancienneté":
                if variable_2 == "Age":
                        plot_bivarie(anciennete,
                                     age, title="Graphe d'analyse bivariée",
                                     x_label=f"{variable_1}", y_label=f"{variable_2}")
                else :
                        plot_bivarie(anciennete, df_no_na[variable_dictionary[variable_2]], title="Graphe d'analyse bivariée",
                                     x_label=f"{variable_1}", y_label=f"{variable_2}")
