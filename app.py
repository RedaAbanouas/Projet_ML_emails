from flask import Flask, render_template, request
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

app = Flask(__name__)

bdd = pd.read_excel("BDD_normalisé.xlsx")

vectorizer = TfidfVectorizer(stop_words=stop_words_fr)
vect = vectorizer.fit_transform(bdd['email'])
selector = SelectKBest(chi2, k=200)
X_new = selector.fit_transform(vect, bdd['type'])

def vectorize(text):
    text = normalize(text)
    vect_text = vectorizer.transform([text])
    vect_selector = selector.transform(vect_text)
    return vect_selector

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict', methods = ['POST'])
def predict():
    pred = None
    email_text = request.form.get("text_data", "")
    vect = vectorize(email_text)
    pred = model.predict(vect)[0]
    result = "SPAM" if pred == 1 else "NON SPAM"
    print(f"Prediction result: {result}")
    return render_template("index.html", prediction = result)


if __name__ == "__main__" :
    app.run(host='192.168.1.4', port='8000', debug=True)
