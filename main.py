from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
import stanza

nltk.download("stopwords")

stop_words_fr = stopwords.words('french')
nlp = stanza.Pipeline("fr", processors="tokenize,mwt,pos,lemma")

def normalize(text):
    doc = nlp(text)
    lemmatized_filtered_text = [word.lemma.lower() for sentence in doc.sentences for word in sentence.words if word.lemma.lower() not in stop_words_fr]
    return " ".join(lemmatized_filtered_text)

with open('svm.pkl', 'rb') as file:
    svm = pickle.load(file)

with open('naivebayes.pkl', 'rb') as file:
    nb = pickle.load(file)

with open("vectorizer.pkl", "rb") as file:
    vectorizer= pickle.load(file)

app = Flask(__name__)

bdd = pd.read_excel("BDD_normalisé.xlsx")


def vectorize(text):
    text = normalize(text)
    vect_text = vectorizer.transform([text])
    return vect_text

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predict', methods = ['POST'])
def predict():

    email_content = request.form.get('email')
    model = request.form.get('model')

    if not email_content or not model:
        return jsonify({'error': 'Email content or model selection is missing'}), 400
    
    vect_email = vectorize(email_content)

    if model == "svm":
        prediction = svm.predict(vect_email)[0]
        pred_result = 'SPAM' if prediction == 1 else 'NOT SPAM'
    elif model == "nb":
        prediction = nb.predict(vect_email)[0]
        pred_result = 'SPAM' if prediction == 1 else 'NOT SPAM'

    return jsonify({'prediction': pred_result})

    if __name__ == '__main__':
        app.run(host='0.0.0.0', port=8000)
