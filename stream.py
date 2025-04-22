
import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from nltk.corpus import stopwords
import nltk
import stanza

st.markdown(
    """
    <style>
        :root {
            --background-color: #0e1117;
            --text-color: #ffffff;
        }
        body {
            background-color: var(--background-color) !important;
            color: var(--text-color) !important;
        }
        div[data-testid="stSidebar"] {
            background-color: #161a23 !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

nltk.download("stopwords")

stop_words_fr = stopwords.words('french')

def normalize(text):
    nlp = stanza.Pipeline("fr", processors="tokenize,mwt,pos,lemma")
    doc = nlp(text)
    lemmatized_filtered_text = [word.lemma.lower() for sentence in doc.sentences for word in sentence.words if word.lemma.lower() not in stop_words_fr]
    return " ".join(lemmatized_filtered_text)

with open('svm.pkl', 'rb') as file:
    svm = pickle.load(file)

with open('naivebayes.pkl', 'rb') as file:
    nb = pickle.load(file)

with open("vectorizer.pkl", "rb") as file:
    vectorizer= pickle.load(file)

bdd = pd.read_excel("BDD_normalisé.xlsx")

def vectorize(text):
    text = normalize(text)
    vect_text = vectorizer.transform([text])
    return vect_text


st.markdown("<h3 style='text-align: center; font-weight: bold;'> 📧Détection d'E-mail SPAM et NON SPAM📧</h3>", unsafe_allow_html=True)
model_choice = st.radio("",["SVM", "Naïve Bayes"], horizontal=True)
st.markdown(" ##### Entrez votre E-mail ici ")
email = st.text_area("", height = 240, placeholder="Entrez votre E-mail...")

st.markdown("""
    <style>
        div[data-testid="stButton"] {
            display: flex;
            justify-content: center;                      
        }
        div[data-testid="stButton"] > button {
            width: 50% ;
            height: 50px; 
        }
    </style>
""", unsafe_allow_html=True)

if st.button("**Prédiction**", type="primary") :
    if not email.strip():
        st.markdown("### **Veuillez entrer un E-mail valide.**")
    else :
        vect = vectorize(email)
        if model_choice == "SVM" :
            pred = svm.predict(vect)[0]
        else :
            pred = nb.predict(vect)[0]
        if pred == 1:
            st.markdown("<br><br>", unsafe_allow_html=True)

            st.markdown("""
                <style>
                    .fade-in {
                        opacity: 0;
                        animation: fadeIn 0.2s linear forwards;
                    }

                    @keyframes fadeIn {
                        from { opacity: 0; }
                        to { opacity: 1; }
                    }
                    
                </style>
                <h3 class="fade-in" style='text-align: center; font-weight: bold; font-size: 3rem'>❌ SPAM</h3>
            """, unsafe_allow_html=True)
            st.markdown(
                """
                <style>
                div[data-testid="stApp"] {
                    background-color: #aa3232 !important;
                    transition: background-color 0.5s linear;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            

  
        elif pred == 0:
            st.markdown("<br><br>", unsafe_allow_html=True)

            st.markdown("""
                <style>
                    .fade-in {
                        opacity: 0;
                        animation: fadeIn 0.2s linear forwards;
                    }

                    @keyframes fadeIn {
                        from { opacity: 0; }
                        to { opacity: 1; }
                    }
                   
                </style>
                <h3 class="fade-in" style='text-align: center; font-weight: bold; font-size: 3rem'>✅ NON SPAM</h3>
            """, unsafe_allow_html=True)

            st.markdown(
                """
                <style>
                div[data-testid="stApp"] {
                    background-color: #149c1f !important;
                    transition: background-color 0.5s linear;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
    

