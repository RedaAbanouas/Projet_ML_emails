
import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from nltk.corpus import stopwords
import nltk
import stanza

stop_words_fr = stopwords.words('french')

def normalize(text):
    nlp = stanza.Pipeline("fr", processors="tokenize,mwt,pos,lemma")
    doc = nlp(text)
    lemmatized_filtered_text = [word.lemma.lower() for sentence in doc.sentences for word in sentence.words if word.lemma.lower() not in stop_words_fr]
    return " ".join(lemmatized_filtered_text)

with open('naivebayes.pkl', 'rb') as file:
    model = pickle.load(file)

with open("vectorizer_selector.pkl", "rb") as f:
    vectorizer, selector = pickle.load(f)

bdd = pd.read_excel("BDD_normalisé.xlsx")

def vectorize(text):
    text = normalize(text)
    vect_text = vectorizer.transform([text])
    vect_selector = selector.transform(vect_text)
    return vect_selector

st.markdown("<h2 style='text-align: center; font-weight: bold;'> 📧Détection d'E-mail SPAM et NON SPAM📧</h2>", unsafe_allow_html=True)
st.markdown(" ##### Entrez votre E-mail ici :")
email = st.text_area("", height = 275, placeholder="Entrez votre E-mail...")

st.markdown("""
    <style>
        div[data-testid="stButton"] {
            display: flex;
            justify-content: center;
        }
        div[data-testid="stButton"] > button {
            width: 30% ;
            height: 50px; 
        }
    </style>
""", unsafe_allow_html=True)

if st.button("**Prédiction**", type="primary") :
    if email == "":
        st.write("dghfghgft")
    else :
        vect = vectorize(email)
        pred = model.predict(vect)[0]
        if pred == 'Spam' :
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
                <h3 class="fade-in" style='text-align: center; font-weight: bold; font-size: 3rem'> ❌ SPAM  </h3>
            """, unsafe_allow_html=True)
            st.markdown(
                f"""
                <style>
                div[data-testid="stApp"] {{
                    background-color: #aa3232 !important;
                    transition: background-color 0.5s linear;
                }}
                </style>
                """,
                unsafe_allow_html=True
            )
            

  
        elif pred == "Non spam":
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
                    body{
                        animation: bgnonspam 0.5s forwards;
                    }
                        
                    @keyframes bgnonspam {
                        from { background-color: #ffffff; }
                        to { background-color: #ff9a9e; }
                    }
                </style>
                <h3 class="fade-in" style='text-align: center; font-weight: bold; font-size: 3rem'> ✅ NON SPAM  </h3>
            """, unsafe_allow_html=True)

            st.markdown(
                f"""
                <style>
                div[data-testid="stApp"] {{
                    background-color: #149c1f !important;
                    transition: background-color 0.5s linear;
                }}
                </style>
                """,
                unsafe_allow_html=True
            )
    

