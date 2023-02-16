# Core Pkgs
import streamlit as st
import os
import re
import openai 
			
import pandas as pd
import matplotlib.pyplot as plt
from pycaret.nlp import *
# NLP Pkgs
from textblob import TextBlob
import spacy
from gensim.summarization.summarizer import summarize
import nltk
nltk.download('punkt')

# Sumy Summary Pkg
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
#CHATGPT
def generate_response(prompt):
    completions = openai.Completion.create(
        engine = "text-davinci-003",
        prompt = prompt,
        max_tokens = 1024,
        n = 1,
        stop = None,
        temperature=0.5, )
    message = completions.choices[0].text
    return message
# function gensim summarization
def summarize(doc):
   doc = re.sub(r'\n|\r', ' ', doc)
   doc = re.sub(r' +', ' ', doc)
   doc = doc.strip()
   result_gen= " ".join(doc.split()[:int(len(doc.split())/2)])
   return result_gen
# Function for Sumy Summarization
def sumy_summarizer(docx):
	parser = PlaintextParser.from_string(docx,Tokenizer("english"))
	lex_summarizer = LexRankSummarizer()
	summary = lex_summarizer(parser.document,3)
	summary_list = [str(sentence) for sentence in summary]
	result = ' '.join(summary_list)
	return result

# Function to Analyse Tokens and Lemma
@st.cache_data
def text_analyzer(my_text):
	nlp = spacy.load('en_core_web_sm')
	docx = nlp(my_text)
	# tokens = [ token.text for token in docx]
	allData = [('"Token":{},\n"Lemma":{}'.format(token.text,token.lemma_))for token in docx ]
	return allData

# Function For Extracting Entities
@st.cache_data
def entity_analyzer(my_text):
	nlp = spacy.load('en_core_web_sm')
	docx = nlp(my_text)
	tokens = [ token.text for token in docx]
	entities = [(entity.text,entity.label_)for entity in docx.ents]
	allData = ['"Token":{},\n"Entities":{}'.format(tokens,entities)]
	return allData

# Enlever les caractères spéciaux , ponctuations
def remove_special_characters(Description):
    # define the pattern to keep
    pat = r'[^a-zA-z0-9.,/:;\"\'\s]' 
    return re.sub(pat, '', Description)

#importation de fichier
def load_data(file):
    data = pd.read_csv(file)
    return data
#features
def select_features(df):
    features = st.multiselect("Sélectionnez les features", df.columns.tolist())
    return features

def select_target(df):
    target = st.selectbox("Sélectionnez la target", df.columns.tolist(),key="unique_key")
    return target

def preprocess_data(df):
    df = df.sample(1000, random_state=786).reset_index(drop=True)
    return df

def setup_experiment(df):
    exp_nlp101 = setup(data=df, target=target, session_id=123)
    return exp_nlp101

@st.cache(allow_output_mutation=True)
def create_lda_model():
    ldafr = create_model('lda')
    return ldafr

@st.cache(allow_output_mutation=True)
def tune_lda_model():
    tuned_classification = tune_model(model='lda', multi_core=True, supervised_target=target)
    return tuned_classification
def main():
	""" NLP Application sur Streamlit """

	# Title
	st.title(" NLP Application")
	st.subheader("Natural Language Processing Application")
	st.markdown("""
    	#### Description
    	+ C'est une application de Natural Language Processing(NLP) Basée sur du traitement automatique du language Naturel, entre autres nous avons: 
    	Tokenization , Lemmatization, Reconnaissance d'entités de nom (NER), Analyse de Sentiments ,  Résumé du texte...! 
    	""")
 
	# Summarization
	if st.checkbox("Trouvez le Résumé de votre texte"):
		st.subheader("Résumez votre Texte")

		message = st.text_area("Entrez le Texte à résumer","Tapez Ici...")
		summary_options = st.selectbox("choisir méthode",['sumy','gensim'])
		if st.button("Summarize"):
			if summary_options == 'sumy':
				st.text("Utilisant la méthode Sumy ..")
				summary_result = sumy_summarizer(message)
			elif summary_options == 'gensim':
				st.text("Utilisant la méthode Gensim  ..")
				summary_result = summarize(message)
			else:
				st.warning("Using Default Summarizer")
				st.text("Utilisant la méthode Gensim  ..")
				summary_result = summarize(message)
			st.success(summary_result)
  
	# Sentiment Analysis
	if st.checkbox("Trouvez le Sentiment de votre  texte"):
		st.subheader("Identification du Sentiment dans votre Texte")

		message = st.text_area("Entrez le Texte à identifier","Tapez Ici...")
		if st.button("Analyse"):
			blob = TextBlob(message)
			result_sentiment = blob.sentiment
			st.success(result_sentiment)

	# Entity Extraction
	if st.checkbox("Trouvez les  Entités de noms dans votre texte"):
		st.subheader("Identification des Entités dans votre texte")

		message = st.text_area("Entrez le Texte pour extraire NER","Tapez Ici...")
		if st.button("Extraction"):
			entity_result = entity_analyzer(message)
			st.json(entity_result)

	# Tokenization 
	if st.checkbox("Trouvez les Tokens et les Lemmas du texte"):
		st.subheader("Tokenisez votre Texte")

		message = st.text_area("Entrez le Texte à Tokeniser","Tapez Ici...")
		if st.button("Tokenise"):
			nlp_result = text_analyzer(message)
			st.json(nlp_result)
			
 	# Translate 
	if st.checkbox("Trouvez la Traduction du  texte en Anglais"):
		st.subheader("Traduisez votre Texte")

		message = st.text_area("Entrez le Texte à traduire","Tapez Ici...")
		if st.button("Traduction"):
		  
			traduction = TextBlob(message).translate(from_lang="fr",to="en")
			st.success(traduction)


  # Remove 
	if st.checkbox("Effacez les caractères spéciaux du texte"):
		st.subheader("Effacez les caractères spéciaux")

		message = st.text_area("Entrez le Texte à nettoyer","Tapez Ici...")
		if st.button("Efface"):
		  
			ResultRemove = remove_special_characters(message)
			st.success(ResultRemove)   

	# ChatBot 
	if st.checkbox("Interagissez avec le Chatbot"):
		st.subheader("Intéraction avec le bot")

		message = st.text_area("Entrez votre recherche ","Tapez Ici...")
		if st.button("Recherche"):
		  
			response = generate_response(message)
			st.success(response)   



       # Charger les données


file = st.file_uploader("Upload file", type=["csv"])
if file is not None:
     df = load_data(file)

        # Afficher les premières lignes du fichier
     st.write("## Les premières lignes du fichier:")
     st.write(df.head())

        # Afficher des informations sur les colonnes
     st.write("## Informations sur les colonnes:")
     st.write(df.info())

        # Afficher les statistiques descriptives
     st.write("## Statistiques descriptives:")
     st.write(df.describe())

        # Afficher la taille des données
     st.write("## La taille des données:")
     st.write(df.shape)

        # Sélectionner les features et la target


st.write("## Choix de la target et des features:")
target = select_target(df)
features = select_features(df)

        # Prétraiter les données

df = preprocess_data(df)
st.write("## Prétraitement:")
st.write(df)

        # Configurer l'expérience PyCaret


exp_nlp101 = setup_experiment(df)
st.write("## Setup:")
st.write(exp_nlp101)
        # Créer le modèle LDA

ldafr = create_lda_model()
st.write(ldafr)
        # Assigner les topics aux documents
lda_results = assign_model(ldafr)
st.write(lda_results)

        # Évaluer le modèle
eval=evaluate_model(ldafr)
st.write(eval)

        # Ajuster le modèle
	
tuned_classification = tune_lda_model()
st.write(tuned_classification)
        # Visualiser les résultats
st.title("Topic Modeling avec PyCaret et Streamlit")
st.subheader("Word Cloud")
stop_words = stopwords.words('french')

tx1=""
for info in df[target]:
    tx1 = tx1 + str(info) + " "

wc = WordCloud(
    background_color='white',
    max_words=2000,
    stopwords=stop_words
 )

wc.generate(str(tx1))

plt.imshow(wc)
plt.axis('off')
st.pyplot()

st.sidebar.subheader("Information sur  de l'Application de Bases de connaissances")
st.sidebar.text("BDC (Bases De Connaissances) Application.")
st.sidebar.info("Cette Application permet de trouver le sentiment score, les tokens et les lemmas dans une phrase ou texte, les entités de noms, suppressions des caractères sspéciaux et Resumé  du texte.!")
	

if __name__ == '__main__':
	main()
