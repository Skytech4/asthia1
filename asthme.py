import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image
import base64
import pickle
import io
import requests
import os
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import json



@st.cache_data
def load_data(dataset):
    df = pd.read_csv(dataset)
    return df

def file_download(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="BMI_IOS_SCD_Asthma.csv">Download CSV File</a>'
    return href

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

lottie_accueil = load_lottiefile("anima_pr/pulmo.json")

lottie_chatbot = load_lottiefile("anima_pr/asthi_bot.json")

# Uniquement pour la probabilité
with open("nettoyage/model_asthma.pkl", "rb") as file:
    loaded_model_with_proba = pickle.load(file)

def predict_asthma(Hydroxyurea, ICS, LABA, Gender, Age, Height , Weight, BMI, R5Hz_PP, R20Hz_PP, X5Hz_PP, Fres_PP):
    try:
        with open('nettoyage/model_asthma.pkl', 'rb') as file:
                        loaded_model = pickle.load(file)

        nouvelles_donnees = pd.DataFrame({
            "Hydroxyurea": [Hydroxyurea],
            "ICS": [ICS],
            "LABA": [LABA],
            "Gender": [Gender],
            "Age": [Age],
            "Height": [Height],
            "Weight": [Weight],
            "BMI": [BMI],
            "R5Hz_PP": [R5Hz_PP],
            "R20Hz_PP": [R20Hz_PP],
            "X5Hz_PP": [X5Hz_PP],
            "Fres_PP": [Fres_PP]
        })

        prediction = loaded_model.predict(nouvelles_donnees)
        if prediction[0] == 0:
            return "Normal"
        else:
            return "Asthmatique"
    except FileNotFoundError:
        st.error("❌ File of Machin Learning Model Unfounded")
    except Exception as e:
        st.error(f'⚠️ Error into the prediction {str(e)}')
    


def load_local_css(file_path):
    with open(file_path, "r") as f:
        css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

load_local_css("Folder_style/asthma.css")


TRANSLATIONS = {
    "Bienvenue sur ASTHIA": {
        "fr": "Bienvenue sur ASTHIA",
        "en": "Welcome to ASTHIA"
    },
    "Qu'est-ce que l'asthme ?": {
        "fr": "Qu'est-ce que l'asthme ?",
        "en": "What is asthma?"
    },
    "L’asthme est une maladie respiratoire chronique qui se caractérise par une inflammation, mais aussi un rétrécissement des voies respiratoires, ce qui rend la respiration plus difficile.": {
        "fr": "L’asthme est une maladie respiratoire chronique qui se caractérise par une inflammation, mais aussi un rétrécissement des voies respiratoires, ce qui rend la respiration plus difficile.",
        "en": "Asthma is a chronic respiratory condition which is characterized by inflammation and narrowing of the airways, making breathing difficult."
    },
    "Visualisation": {"fr": "Visualisation", "en": "Visualization"},
    "Analyse": {"fr": "Analyse", "en": "Analysis"},
    "ChatBot": {"fr": "ChatBot", "en": "ChatBot"},
    "Home": {"fr": "Accueil", "en": "Home"},
    "Langue": {"fr": "🇫🇷", "en": "🇬🇧"},
    "Formulaire des paramètres asthmatiques": {
        "fr": "Formulaire des paramètres asthmatiques",
        "en": "Asthma Parameters Form"
    },
    "Entrez les paramètres démographiques :": {
        "fr": "Entrez les paramètres démographiques :",
        "en": "Enter demographic parameters:"
    },
    "Entrez les paramètres oscillométriques :": {
        "fr": "Entrez les paramètres oscillométriques :",
        "en": "Enter oscillometric parameters:"
    },
    "Entrez les autres paramètres :": {
        "fr": "Entrez les autres paramètres :",
        "en": "Enter other parameters:"
    },
    "Prédiction": {"fr": "Prédiction", "en": "Prediction"},
    "Le patient est": {"fr": "Le patient est", "en": "The patient is"},
    "Normale": {"fr": "Normale", "en": "Normal"},
    "Asthmatique": {"fr": "Asthmatique", "en": "Asthmatic"},
    "Visualisation des données": {
        "fr": "Visualisation des données",
        "en": "Data Visualization"
    },
    "Count Plot du top 10 des valeurs du BMI" : {
        "fr": "Count Plot du top 10 des valeurs du BMI",
        "en": "Count Plot of the top 10 BMI values"
    },
    "Pie Chart du top 4 des valeurs fréquentes" : {
        "fr": "Pie Chart du top 4 des valeurs fréquentes",
        "en": "Pie Chart of the top 4 frequent values"
    },
    "Scatter Plot des tailles en fonction des poids": {
        "fr": "Scatter Plot des tailles en fonction des poids",
        "en": "Scatter Plot of heights vs weights"
    },
    "Boxplot du genre en fonction de l’âge": {
        "fr": "Boxplot du genre en fonction de l’âge",
        "en": "Boxplot of gender vs age"
    },
    "Sélectionnez une constante": {
        "fr": "Sélectionnez une constante",
        "en": "Select a constant"
    },
    "Moyenne de R5Hz_PP 🩸": {
        "fr": "Moyenne de R5Hz_PP 🩸",
        "en": "Average of R5Hz_PP 🩸"
    },
    "Compte par genre 🚹/🚺": {
        "fr": "Compte par genre 🚹/🚺",
        "en": "Count by gender 🚹/🚺"
    },
    "Moyenne de R20Hz_PP 💉":  {
        "fr": "Moyenne de R20Hz_PP 💉",
        "en": "Average of R20Hz_PP 💉"
    },
    "Moyenne de Fres_PP 💊": {
        "fr": "Moyenne de Fres_PP 💊",
        "en": "Average of Frez_PP 💊"
    },
    "Choisissez le statut de la maladie": {
        "fr": "Choisissez le statut de la maladie",
        "en": "Choose the disease's status"
    },
    "Sélectionnez une image médicale pour l'analyse par imagerie": {
        "fr": "Sélectionnez une image médicale pour l'analyse par imagerie",
        "en": "Select an medical image for the imagery analysis"
    },
    "⚠️ Je ne peux répondre à cette question. Veuillez réessayer": {
        "fr": "⚠️ Je ne peux répondre à cette question. Veuillez réessayer",
        "en": "⚠️ I can't answer to this question. Please try again"
    },
    "Veuillez poser votre question (ex: Que signifie R5Hz_PP ?)":{
        "fr": "Veuillez poser votre question (ex: Que signifie R5Hz_PP ?)",
        "en": "Please ask your question (ex: What does R5Hz_PP mean ?)"
    },
    "ASTHIBOT réfléchit...": {
        "fr": "ASTHIBOT réfléchit...",
        "en": "ASTHIBOT thinks..."
    },
    "👋 Bonjour, je suis AsthiBot ! Comment puis-je vous aider aujourd'hui ?": {
        "fr": "👋 Bonjour, je suis AsthiBot ! Comment puis-je vous aider aujourd'hui ?",
        "en": "👋 Good morning, I am AsthiBot ! How can I help you today ?"
    },
    "Discutez avec ASTHIBOT": {
        "fr": "Discutez avec ASTHIBOT",
        "en": "Discuss with ASTHIBOT"
    },
    "Image sélectionnée": {
        "fr": "Image sélectionnée",
        "en": "image selected"
    },
    "Etant une condition assez complexe, l'on note l'intervention de plusieurs sous-mécanismes dits physiopathologiques qui sont : une inflammation des voies respiratoires, une hyperréactivité des bronches et un remodelage des voies respiratoires.": {
        "fr": "Etant une condition assez complexe, l'on note l'intervention de plusieurs sous-mécanismes dits physiopathologiques qui sont : une inflammation des voies respiratoires, une hyperréactivité des bronches et un remodelage des voies respiratoires.",
        "en": "Being a rather complex condition, several sub-mechanisms known as pathophysiological mechanisms are involved: inflammation of the airways, bronchial hyperreactivity, and airway remodeling."
    },
    "Quels sont les symptomes et les facteurs déclencheurs de l'asthme ?": {
        "fr": "Quels sont les symptomes et les facteurs déclencheurs de l'asthme ?",
        "en": "What are the symptoms and triggering factors of asthma?"
    },
    "L'asthme est perçu à l'observation des symptômes suivants :": {
        "fr": "L'asthme est perçu à l'observation des symptômes suivants :",
        "en": "Asthma is perceived by observing the following symptoms :"
    },
    "- La toux persistante, surtout la nuit ou tôt le matin ;": {
        "fr": "- La toux persistante, surtout la nuit ou tôt le matin ;",
        "en": "- Persistent cough, especially at night or early in the morning ;"
    },
    "- La respiration sifflante ou bruyante ;": {
        "fr": "- La respiration sifflante ou bruyante ;",
        "en": "- Wheezing or noisy breathing ;"
    },
    "- L'essoufflement ou difficulté à respirer ;": {
        "fr": "- L'essoufflement ou difficulté à respirer ;",
        "en": "- Shortness of breath or difficulty breathing ;"
    },
    "- L'oppression thoracique.": {
        "fr": "- L'oppression thoracique.",
        "en": "- Chest tightness."
    },
    "Les symptômes peuvent varier d’une personne à l’autre. Et parfois, ils peuvent s’aggraver de façon considérable.": {
        "fr": "Les symptômes peuvent varier d’une personne à l’autre. Et parfois, ils peuvent s’aggraver de façon considérable.",
        "en": "Symptoms can vary from person to person. And sometimes, they can worsen considerably."
    },
    "Concernant les facteurs déclencheurs, ce sont les éléments nuisibles pouvant provoquer une manifestation des symptomes. En somme, ils peuvent inclure : les allergènes, les irritants, les efforts physiques ou encore les changements climatiques ...": {
        "fr": "Concernant les facteurs déclencheurs, ce sont les éléments nuisibles pouvant provoquer une manifestation des symptomes. En somme, ils peuvent inclure : les allergènes, les irritants, les efforts physiques ou encore les changements climatiques ...",
        "en": "Triggering factors are harmful elements that can cause the onset of symptoms. In summary, they may include allergens, irritants, physical exertion, or even climatic changes ..."
    },
    "Quels sont les traitements contre l'asthme ?": {
        "fr": "Quels sont les traitements contre l'asthme ?",
        "en": "What are the treatments for asthma?"
    },
    "Malgré le fait que l’asthme ne puisse pas encore être guéri, l’on peut du moins la contrôler, ceci grâce à plusieurs traitements méticuleux pouvant permettre aux personnes atteintes de s’épanouir. D’après l’OMS et aux vues des avancées de la médecine moderne, le traitement le plus courant de nos jours reste encore l’utilisation d’un inhalateur qui diffuse le médicament directement dans les poumons.":{
        "fr": "Malgré le fait que l’asthme ne puisse pas encore être guéri, l’on peut du moins la contrôler, ceci grâce à plusieurs traitements méticuleux pouvant permettre aux personnes atteintes de s’épanouir. D’après l’OMS et aux vues des avancées de la médecine moderne, le traitement le plus courant de nos jours reste encore l’utilisation d’un inhalateur qui diffuse le médicament directement dans les poumons.",
        "en": "Although asthma cannot yet be cured, it can at least be controlled thanks to several meticulous treatments that allow affected individuals to thrive. According to the WHO and in light of modern medical advances, the most common treatment today remains the use of an inhaler that delivers medication directly into the lungs."
    },
    "- Les bronchodilatateurs à savoir le salbutamol communément appelée ventoline qui ouvrent les voies aériennes et soulagent les symptômes. Et ;": {
        "fr": "- Les bronchodilatateurs à savoir le salbutamol communément appelée ventoline qui ouvrent les voies aériennes et soulagent les symptômes. Et ;",
        "en": "- Bronchodilators, such as salbutamol—commonly known as Ventolin—which open the airways and relieve symptoms. And ;"
    },
    "- Les stéroïdes comme la béclométasone ou la ciclésonide qui réduisent l’inflammation des voies respiratoires, réduisant ainsi les risques de crises d’asthme graves et de décès.": {
        "fr": "- Les stéroïdes comme la béclométasone ou la ciclésonide qui réduisent l’inflammation des voies respiratoires, réduisant ainsi les risques de crises d’asthme graves et de décès.",
        "en": "- Steroids such as beclometasone or ciclesonide, which reduce inflammation in the airways, thereby lowering the risk of severe asthma attacks and death."
    },
    "Parallèlement, nous avons les thérapies biologiques qui requièrent l’utilisation des anticorps monoclonaux pour l’élimination d’éléments inflammatoires en cas d’asthme sévère chez un individu.": {
        "fr": "Parallèlement, nous avons les thérapies biologiques qui requièrent l’utilisation des anticorps monoclonaux pour l’élimination d’éléments inflammatoires en cas d’asthme sévère chez un individu.",
        "en": "In parallel, there are biological therapies that involve the use of monoclonal antibodies to eliminate inflammatory elements in cases of severe asthma in an individual."
    },
    "Le patient est :": {
        "fr": "Le patient est :",
        "en": "The patient is :"
    },
    "La probabilité de ce résultat est élevée à :": {
        "fr": "La probabilité de ce résultat est élevée à :",
        "en": "The probability of this result is high at :"
    },
    "Sélectionnez un jeu de données pour l'analyse fichier": {
        "fr": "Sélectionnez un jeu de données pour l'analyse fichier",
        "en": "Select a dataset for file analysis"
    },
    "Télécharger le fichier CSV": {
        "fr": "Télécharger le fichier CSV",
        "en": "Download CSV file"
    },
    "Analyse par formulaire": {
        "fr": "Analyse par formulaire",
        "en": "Form-based analysis"
    },
    "Analyse par CSV": {
        "fr": "Analyse par CSV",
        "en": "CSV-based analysis"
    },
    "Prédictions sur le jeu de données":{
        "fr": "Prédictions sur le jeu de données",
        "en": "Predictions on the dataset"
    }
}

def t(texte):
    """Fonction pour traduire un texte selon la langue choisie"""
    langue = st.session_state.get("langue", "fr")
    return TRANSLATIONS.get(texte, {}).get(langue, texte)


def main():
    # Initialiser la langue
    if 'langue' not in st.session_state:
        st.session_state.langue = 'fr'

    # Initialiser la page active si elle n'existe pas encore
    if 'page' not in st.session_state:
        st.session_state.page = "Home"

    # Sidebar image path correction
    st.sidebar.image("im_pr/asthia.png", width=300)

    # Sélecteur de langue
    langue_actuelle = st.session_state.get("langue", "fr")

    langue_choisie = st.sidebar.selectbox(
        "🌐 Langue/Language",
        options=["fr", "en"],
        format_func=lambda x: "🇫🇷 Français" if x == "fr" else "🇬🇧 English"
    )

    # Met à jour la langue si elle a changé
    if langue_choisie != langue_actuelle:
        st.session_state.langue = langue_choisie




    # Boutons du menu sidebar
    if st.sidebar.button(t("Home")):
        st.session_state.page = "Home"
    if st.sidebar.button(t("Visualisation")):
        st.session_state.page = "Visualisation"
    if st.sidebar.button(t("Analyse")):
        st.session_state.page = "Analyse"
    if st.sidebar.button(t("ChatBot")):
        st.session_state.page = "ChatBot"

    # Utiliser st.session_state.page comme choix
    choice = st.session_state.page

    if choice not in ["Visualisation", "Analyse", "ChatBot"]:
        collo1, collo2 = st.columns([0.8, 0.2])
        with collo1:
            st.title(t("Bienvenue sur ASTHIA"))
        with collo2:
            st_lottie(lottie_accueil, height=85)


    # Load dataset
    data = load_data("Dataset/BMI_IOS_SCD_Asthma.csv")
    
    if choice == "Home":
        st.markdown("-------------------")
        st.subheader(t("Qu'est-ce que l'asthme ?"))
        st.markdown("")

        ltab1, ltab2 = st.columns(2)
        with ltab1:
            st.markdown(f"""
            <div style="text-align: justify; font-size: 16px;">
            {t("L’asthme est une maladie respiratoire chronique qui se caractérise par une inflammation, mais aussi un rétrécissement des voies respiratoires, ce qui rend la respiration plus difficile.")}
            </div>
            """, unsafe_allow_html=True) 
            st.markdown(f"""
            <div style="text-align: justify; font-size: 16px;">
            {t("Etant une condition assez complexe, l'on note l'intervention de plusieurs sous-mécanismes dits physiopathologiques qui sont : une inflammation des voies respiratoires, une hyperréactivité des bronches et un remodelage des voies respiratoires.")}
            </div>
            """, unsafe_allow_html=True)

            
        with ltab2:
            st.image("im_pr/asthme11.jpg", width=400)

        st.markdown("")
        st.markdown("-------------------")

        st.subheader(t("Quels sont les symptomes et les facteurs déclencheurs de l'asthme ?"))
        st.markdown("")
        
        xxtab1, xxtab2 = st.columns(2)
        with xxtab1:
            st.markdown(f"""
            <div style="text-align: justify; font-size: 16px;">
            {t("L'asthme est perçu à l'observation des symptômes suivants :")}
            </div>
            """, unsafe_allow_html=True)
            st.markdown("")
            st.markdown(f"""
            <div style="text-align: justify; font-size: 16px;">
            {t("- La toux persistante, surtout la nuit ou tôt le matin ;")}
            </div>
            """, unsafe_allow_html=True)
            st.markdown("")
            st.markdown(f"""
            <div style="text-align: justify; font-size: 16px;">
            {t("- La respiration sifflante ou bruyante ;")}
            </div>
            """, unsafe_allow_html=True)
            st.markdown("")
            st.markdown(f"""
            <div style="text-align: justify; font-size: 16px;">
            {t("- L'essoufflement ou difficulté à respirer ;")}
            </div>
            """, unsafe_allow_html=True)
            st.markdown("")
            st.markdown(f"""
            <div style="text-align: justify; font-size: 16px;">
            {t("- L'oppression thoracique.")}
            </div>
            """, unsafe_allow_html=True)
            st.markdown("")
        
        with xxtab2:
            st.image("im_pr/asthme33.png", width=400)


        st.markdown(f"""
        <div style="text-align: justify; font-size: 16px;">
        {t("Les symptômes peuvent varier d’une personne à l’autre. Et parfois, ils peuvent s’aggraver de façon considérable.")}
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")
        st.markdown(f"""
        <div style="text-align: justify; font-size: 16px;">
        {t("Concernant les facteurs déclencheurs, ce sont les éléments nuisibles pouvant provoquer une manifestation des symptomes. En somme, ils peuvent inclure : les allergènes, les irritants, les efforts physiques ou encore les changements climatiques ...")}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")
        st.markdown("")
        st.markdown("-------------------")

        st.subheader(t("Quels sont les traitements contre l'asthme ?"))
        st.markdown("")
        st.markdown(f"""
        <div style="text-align: justify; font-size: 16px;">
        {t("Malgré le fait que l’asthme ne puisse pas encore être guéri, l’on peut du moins la contrôler, ceci grâce à plusieurs traitements méticuleux pouvant permettre aux personnes atteintes de s’épanouir. D’après l’OMS et aux vues des avancées de la médecine moderne, le traitement le plus courant de nos jours reste encore l’utilisation d’un inhalateur qui diffuse le médicament directement dans les poumons.")}
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")
        st.markdown(f"""
        <div style="text-align: justify; font-size: 16px;">
        {t("- Les bronchodilatateurs à savoir le salbutamol communément appelée ventoline qui ouvrent les voies aériennes et soulagent les symptômes. Et ;")}
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")
        st.markdown(f"""
        <div style="text-align: justify; font-size: 16px;">
        {t("- Les stéroïdes comme la béclométasone ou la ciclésonide qui réduisent l’inflammation des voies respiratoires, réduisant ainsi les risques de crises d’asthme graves et de décès.")}
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")
        st.markdown(f"""
        <div style="text-align: justify; font-size: 16px;">
        {t("Parallèlement, nous avons les thérapies biologiques qui requièrent l’utilisation des anticorps monoclonaux pour l’élimination d’éléments inflammatoires en cas d’asthme sévère chez un individu.")}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        btab1, btab2 = st.columns(2)
        with btab1:
            st.image("im_pr/asthme.jpeg", width=400)

        with btab2:
            st.image("im_pr/asthme3.jpg", width=400)
        
        
        
    elif choice == "Visualisation":
        st.title(t("Visualisation des données"))

        asthma_filter = st.selectbox(t("Choisissez le statut de la maladie"), pd.unique(data['Asthma']))
        data = data[data['Asthma'] == asthma_filter]

        avg_r5 = np.mean(data['R5Hz_PP'])
        count_gender= int(data[(data['Gender'] == 'Male')]['Gender'].count())
        avg_r20 = np.mean(data['R20Hz_PP'])
        avg_fres = np.mean(data['Fres_PP'])

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric(label=t("Moyenne de R5Hz_PP 🩸"), value=round(avg_r5), delta=round(avg_r5))
        kpi2.metric(label=t("Compte par genre 🚹/🚺"), value=count_gender, delta=round(count_gender))
        kpi3.metric(label=t("Moyenne de R20Hz_PP 💉"), value=f'{round(avg_r20,2)}', delta=f'{round(avg_r20,2)}')
        kpi4.metric(label=t("Moyenne de Fres_PP 💊"), value=round(avg_fres), delta=round(avg_fres))

        col1, col2 = st.columns(2)
        with col1:
            st.subheader(t("Count Plot du top 10 des valeurs du BMI"))
            fig1 = plt.figure(figsize=(9, 5.97))
            top_values = data['BMI'].value_counts().nlargest(10).index
            filtered_data = data[data['BMI'].isin(top_values)]
            sns.countplot(data=filtered_data, x="BMI", palette='muted')
            plt.title(t('Count Plot des 10 premiers BMI'))
            st.pyplot(fig1)


            st.subheader(t("Scatter Plot des tailles en fonction des poids"))
            fig2 = plt.figure(figsize=(9, 7.5))
            sns.scatterplot(x="Height (cm)", y="Weight (Kg)", data=data)
            plt.title(t("Scatter Plot des tailles en fonction des poids"))
            st.pyplot(fig2)
        
        with col2:
            st.subheader(t("Pie Chart du top 4 des valeurs fréquentes"))
            numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col != 'Unnamed: 0']

            if len(numeric_cols) > 0:
                selected_col = st.selectbox(t("Sélectionnez une constante"), ['R5Hz_PP', 'R20Hz_PP', 'Fres_PP'])
                top_values = data[selected_col].value_counts().nlargest(4)
                fig3 = plt.figure(figsize=(2,2.2))
                plt.pie(top_values.values, labels=top_values.index, autopct='%1.1f%%', startangle=90)
                plt.title(f'Distribution du top 4 de valeurs fréquentes {selected_col}')
                plt.axis("equal")
                st.pyplot(fig3)
            
            st.subheader(t('Boxplot du genre en fonction de l’âge'))
            data_filtered = data[data['Gender'] != 'male']
            fig4 = plt.figure(figsize=(5,4.35))
            sns.boxplot(x="Gender", y="Age (months)", data=data_filtered, palette='viridis')
            st.pyplot(fig4)
            
    elif choice == "Analyse":
        st.title(t("Analyse"))
        tab1,tab2 = st.tabs([":microscope: Machine Learning", ":brain: Deep Learning"])
        
        with tab1:

            tab11, tab12 = st.tabs([t("Analyse par formulaire"), t("Analyse par CSV")])
            with tab11:

                # Sliders pour les valeurs numériques
                st.subheader(t("Formulaire des paramètres asthmatiques"))

                st.markdown("")
                st.markdown("")

                st.write(t("Entrez les paramètres démographiques :"))
                stab1, stab2 = st.columns(2)

                with stab1:
                        Age = st.number_input("Age", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
                        Height = st.number_input("Height", min_value=0, max_value=230, value=0, step=1)
                        Gender = st.selectbox("Gender", options=["0", "1"])

                with stab2:
                        BMI = st.number_input("BMI", min_value=0.0, max_value=40.0, value=0.0, step=1.0)
                        Weight = st.number_input("Weight", min_value=0, max_value=150, value=0, step=1)


                st.markdown("")
                st.markdown("")
                st.markdown("")

                st.write(t("Entrez les paramètres oscillométriques :"))
                sstab1, sstab2 = st.columns(2)
                with sstab1:
                        R5Hz_PP = st.number_input("R5Hz_PP", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
                        R20Hz_PP = st.number_input("R20Hz_PP", min_value=0.0, max_value=1.0, value=0.0, step=0.01)

                with sstab2:
                        X5Hz_PP = st.number_input("X5Hz_PP", min_value=-0.70, max_value=0.70, value=0.0, step=0.01)
                        Fres_PP = st.number_input("Fres_PP", min_value=0.0, max_value=50.0, value=0.0, step=0.01)


                st.markdown("")
                st.markdown("")
                st.markdown("")

                st.write(t("Entrez les autres paramètres :"))
                ssstab1, ssstab2 = st.columns(2)
                with ssstab1:
                    LABA = st.selectbox("LABA", options=["0", "1"])
                    ICS = st.selectbox("ICS", options=["0", "1"])
                
                with ssstab2:
                    Hydroxyurea = st.selectbox("Hydroxyurea", options=["0", "1"])


                st.markdown("")
                st.markdown("")

                button1 = st.button(t("Prédiction"),  key="predict_ml1")
                if button1:
                    New_Asthma = predict_asthma(Hydroxyurea, ICS, LABA, Gender, Age, Height , Weight, BMI, R5Hz_PP, R20Hz_PP, X5Hz_PP, Fres_PP)
                    if New_Asthma:
                        st.success(f"{t('Le patient est :')} **{New_Asthma}**")

                            # -- Création du vecteur d'entrée au bon format --
                        input_data = pd.DataFrame([[
                            float(Hydroxyurea), float(ICS), float(LABA), float(Gender), Age, Height,
                            Weight, BMI, R5Hz_PP, R20Hz_PP, X5Hz_PP, Fres_PP
                        ]], columns=[
                            "Hydroxyurea", "ICS", "LABA", "Gender", "Age", "Height",
                            "Weight", "BMI", "R5Hz_PP", "R20Hz_PP", "X5Hz_PP", "Fres_PP"
                        ])

                        # -- Récupération de la probabilité --
                        proba = loaded_model_with_proba.predict_proba(input_data)[0]
                        class_index = 1 if New_Asthma == "Asthmatique" else 0
                        proba_val = round(100 * proba[class_index], 2)

                        # -- Affichage de la probabilité --
                        st.markdown(
                            f"<div style='padding: 10px; border: 2px solid #aaa; border-radius: 8px; font-size: 16px; text-align: center;'>"
                            f"{t('La probabilité de ce résultat est élevée à :')} <b>{proba_val} %</b>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

                        st.markdown("")
                        st.markdown("")

                        # -- Jauge avec Plotly --
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=proba_val,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            number={'font': {'color': 'black'}},
                            gauge={
                                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white", 'tickfont': {'color': 'white'}},
                                'bar': {'color': "#d9534f" if class_index == 1 else "#5cb85c"},
                                'bgcolor': "#000000",
                                'borderwidth': 2,
                                'bordercolor': "gray",
                                'steps': [
                                    {'range': [0, 50], 'color': '#111111'},
                                    {'range': [50, 75], 'color': '#222222'},
                                    {'range': [75, 100], 'color': '#333333'}
                                ],
                                'threshold': {
                                    'line': {'color': "#d9534f" if class_index == 1 else "#5cb85c", 'width': 4},
                                    'thickness': 0.75,
                                    'value': proba_val
                                }
                            }
                        ))
                        fig.update_layout(paper_bgcolor="#000000", plot_bgcolor="#000000", height=250, margin=dict(t=20, b=0, l=20, r=20))
                        st.plotly_chart(fig, use_container_width=True)



                        # Sauvegarde des données dans session_state pour le ChatBot
                        st.session_state['derniere_prediction'] = {
                            "Hydroxyurea": Hydroxyurea,
                            "ICS": ICS,
                            "LABA": LABA,
                            "Gender": Gender,
                            "Age": Age,
                            "Height": Height,
                            "Weight": Weight,
                            "BMI": BMI,
                            "R5Hz_PP": R5Hz_PP,
                            "R20Hz_PP": R20Hz_PP,
                            "X5Hz_PP": X5Hz_PP,
                            "Fres_PP": Fres_PP,
                            "Résultat": New_Asthma,
                            "Probabilité": proba_val
                        }

            with tab12:
                
                # Téléversement du fichier CSV
                file_uploaded = st.file_uploader(t("Sélectionnez un jeu de données pour l'analyse fichier"), type=["csv"])

                if file_uploaded:
                    datas_csv = load_data(file_uploaded)

                    # Définition de l'ordre attendu par le modèle
                    colonnes_modele = [
                        "Hydroxyurea", "ICS", "LABA", "Gender",
                        "Age", "Height", "Weight", "BMI",
                        "R5Hz_PP", "R20Hz_PP", "X5Hz_PP", "Fres_PP"
                    ]

                    # Réorganisation des colonnes dans le bon ordre
                    datas_csv = datas_csv[colonnes_modele]

                    # Conversion des valeurs catégorielles en numériques
                    datas_csv["Hydroxyurea"] = datas_csv["Hydroxyurea"].map({"Oui": 1, "Non": 0})
                    datas_csv["ICS"] = datas_csv["ICS"].map({"Oui": 1, "Non": 0})
                    datas_csv["LABA"] = datas_csv["LABA"].map({"Oui": 1, "Non": 0})
                    datas_csv["Gender"] = datas_csv["Gender"].map({"Homme": 1, "Femme": 0})

                    ndf = None

                    if st.button(t("Prédiction"), key="predict_ml2"):
                        loaded_model1 = pickle.load(open("nettoyage/model_asthma.pkl", "rb"))
                        prediction_csv = loaded_model1.predict(datas_csv)

                        # Création d'un DataFrame avec la colonne Avis
                        pp = pd.DataFrame(prediction_csv, columns=["Result"])
                        ndf = pd.concat([datas_csv, pp], axis=1)

                        # Remplacement des valeurs numériques par labels
                        ndf["Result"].replace(0, "Normal", inplace=True)
                        ndf["Result"].replace(1, "Asthmatique", inplace=True)

                        st.subheader(t("Prédictions sur le jeu de données"))
                        st.write(ndf)

                        # Bouton téléchargement CSV
                        button2 = st.button(t("Télécharger le fichier CSV"), key="download_csv_ml2")
                        if button2:
                            st.markdown(file_download(ndf), unsafe_allow_html=True)

        with tab2:
            
            deep_model = load_model('nettoyage/deep_asthma.h5')

            image_height = 200
            image_width = 200

            uploaded_file = st.file_uploader(t("Sélectionnez une image médicale pour l'analyse par imagerie"), type=["jpg", "jpeg", "png"])

            if uploaded_file is not None:

                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption=t("Image sélectionnée"), use_container_width=True)

                # Prétraitement
                img = image.resize((image_height, image_width))
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                prediction = deep_model.predict(img_array)

                # Obtenir la probabilité brute
                prob = float(prediction[0][0])
                result_dl = "Asthmatique" if prob > 0.5 else "Normal"
                proba_text = round(prob * 100, 2) if result_dl == "Asthmatique" else round((1 - prob) * 100, 2)

                st.success(f"{t('Le patient est :')} **{result_dl}**")

                st.markdown(
                    f"<div style='padding: 10px; border: 2px solid #aaa; border-radius: 8px; font-size: 16px; text-align: center;'>"
                    f"{t('La probabilité de ce résultat est élevée à :')} <b>{proba_text} %</b>"
                    f"</div>",
                    unsafe_allow_html=True
                )

                st.markdown("")
                st.markdown("")
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = proba_text,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    number = {'font': {'color': 'black'}},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white", 'tickfont': {'color': 'white'}},
                        'bar': {'color': "#d9534f" if result_dl == "Asthmatique" else "#5cb85c"},
                        'bgcolor': "#000000",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 50], 'color': '#111111'},
                            {'range': [50, 75], 'color': '#222222'},
                            {'range': [75, 100], 'color': '#333333'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': proba_text
                        }
                    }
                ))

                fig.update_layout(paper_bgcolor="#000000", plot_bgcolor="#000000", height=250, margin=dict(t=20, b=0, l=20, r=20))
                st.plotly_chart(fig, use_container_width=True)



                # Sauvegarde dans session_state pour le ChatBot
                st.session_state['derniere_prediction_dl'] = {
                    "Résultat_DL": result_dl,
                    "Probabilité_DL": proba_text
                }


    elif choice == "ChatBot":
        col1, col2 = st.columns([0.8, 0.2])  # Ajuste les proportions selon la taille de l’image
        
        with col1:
            st.title(t("Discutez avec ASTHIBOT"))

            st.markdown("""
            <style>
            .typewriter-container {
                text-align: center;
                margin-top: 10px;
                margin-bottom: 1px;
                font-family: Germania One, sans-serif;
                font-size: 20px;
                color: #ffffff;
            }

            .typewriter-box {
                display: inline-block;
                background-color: #000000; /* couleur de fond du cadre */
                color: white; /* couleur du texte à l'intérieur */
                padding: 10px 14px; /* espace interne */
                border: 2px solid #2a2a2a; /* bordure du cadre */
            }

            .typewriter-text {
                display: inline-block;
                overflow: hidden;
                white-space: nowrap;
                animation:
                    typing 3s steps(50, end),
                    fadeBlur 1.5s ease-out forwards;
                filter: blur(6px);
            }

            @keyframes typing {
                from { width: 0 }
                to { width: 100% }
            }

            @keyframes fadeBlur {
                0% { filter: blur(6px); }
                100% { filter: blur(0px); }
            }
            </style>

            <div class="typewriter-container">
                <div class="typewriter-box">
                    <div class="typewriter-text">👋 Bonjour, je suis AsthiBot ! Comment puis-je vous aider aujourd'hui ?</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st_lottie(lottie_chatbot, height=85)

        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Liste des mots-clés autorisés pour filtrer les questions
        allowed_keywords = [
            "asthme", "asthma", "symptôme", "symptoms", "traitement", "treatment",
            "R5Hz", "R20Hz", "X5Hz", "Fres", "BMI", "ICS", "LABA", "Hydroxyurea",
            "prédiction", "prediction", "résultat", "result", "diagnostic", "diagnosis",
            "proposition", "proposition de traitement", "question", "réponse",
            "répondre", "answer", "aide", "help", "conseil", "advice", "informations", "information",
            "image", "radiographie", "scanner", "état de santé", "health status", "probabilité", "probability"
        ]

        def is_question_relevant(question):
            return any(keyword.lower() in question.lower() for keyword in allowed_keywords)

        API_URL = "https://router.huggingface.co/nebius/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {os.environ['Token1']}",
        }

        def query_hf_model(message, prediction_data=None, prediction_data_dl=None):

            lang = st.session_state.get("langue", "fr")

            if lang == "fr":
                # Construction du contexte avec les dernières données de prédiction
                context = "Tu es un expert en santé respiratoire. Réponds uniquement aux questions concernant " \
                        "l'asthme ou les résultats de prédiction liés à l'asthme. Donne des réponses détaillées, " \
                        "claires et faciles à comprendre pour un patient ou un professionnel de santé."
            else:
                context = "You are a respiratory health expert. Answer only questions related to asthma or asthma prediction results. " \
                        "Provide detailed, clear, and easy-to-understand responses for patients or healthcare professionals."    


            if prediction_data:
                context += f"""
            Voici les dernières données du patient à considérer pour le contexte :
            Hydroxyurea: {prediction_data.get("Hydroxyurea", "N/A")}
            ICS: {prediction_data.get("ICS", "N/A")}
            LABA: {prediction_data.get("LABA", "N/A")}
            Gender: {prediction_data.get("Gender", "N/A")}
            Age: {prediction_data.get("Age", "N/A")}
            Height: {prediction_data.get("Height", "N/A")}
            Weight: {prediction_data.get("Weight", "N/A")}
            BMI: {prediction_data.get("BMI", "N/A")}
            R5Hz_PP: {prediction_data.get("R5Hz_PP", "N/A")}
            R20Hz_PP: {prediction_data.get("R20Hz_PP", "N/A")}
            X5Hz_PP: {prediction_data.get("X5Hz_PP", "N/A")}
            Fres_PP: {prediction_data.get("Fres_PP", "N/A")}
            Résultat de la prédiction : {prediction_data.get("Résultat", "N/A")}
            """
            
            if prediction_data_dl:
                    context += f"""
            Voici le résultat de la dernière prédiction Deep Learning :
            Résultat prédiction DL : {prediction_data_dl.get("Résultat_DL", "N/A")}
            """

            payload = {
                "messages": [
                    {"role": "system", "content": context},
                    {"role": "user", "content": message}
                ],
                "model": "microsoft/phi-4",
                "temperature": 0.4,
                "max_tokens": 768
            }
            try:
                response = requests.post(API_URL, headers=headers, json=payload)
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"]
                else:
                    return f"Erreur d'appel API : {response.status_code}"
            except Exception as e:
                return f"Erreur : {str(e)}"
        

        # Récupérer les données de la dernière prédiction
        prediction_data = st.session_state.get('derniere_prediction', {})

        prediction_data_dl = st.session_state.get('derniere_prediction_dl', {})

        user_message = st.chat_input(t("Veuillez poser votre question (ex: Que signifie R5Hz_PP ?)"))

        if user_message:
            if is_question_relevant(user_message):
                with st.spinner(t("ASTHIBOT réfléchit...")):
                    bot_reply = query_hf_model(user_message, prediction_data, prediction_data_dl)
                

                # Stocker la conversation
                st.session_state.messages.append({"role": "user", "content": user_message})
                st.session_state.messages.append({"role": "assistant", "content": bot_reply})
            else:
                st.warning(t("⚠️ Je ne peux répondre à cette question. Veuillez réessayer"))
        
        # Affichage des messages avec style bulle
        for msg in st.session_state.messages:
            clean_content = msg["content"].replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
            
            if msg["role"] == "user":
                st.markdown(f"""
                    <div style='display: flex; justify-content: flex-end; margin: 10px 0;'>
                        <div style='position: relative; background-color: #f0f0f0; color: black; padding: 10px 14px; border-radius: 16px; max-width: 70%;'>
                            <div style='white-space: pre-wrap;'>{clean_content}</div>
                            <div style="
                                content: '';
                                position: absolute;
                                right: -10px;
                                top: 10px;
                                width: 0;
                                height: 0;
                                border-top: 10px solid transparent;
                                border-bottom: 10px solid transparent;
                                border-left: 10px solid #f0f0f0;
                            "></div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div style='display: flex; justify-content: flex-start; margin: 10px 0;'>
                        <div style='position: relative; background-color: #2b2a2a; color: white; padding: 10px 14px; border-radius: 16px; max-width: 70%;'>
                            <div style='white-space: pre-wrap;'>{clean_content}</div>
                            <div style="
                                content: '';
                                position: absolute;
                                left: -10px;
                                top: 10px;
                                width: 0;
                                height: 0;
                                border-top: 10px solid transparent;
                                border-bottom: 10px solid transparent;
                                border-right: 10px solid #2b2a2a;
                            "></div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()