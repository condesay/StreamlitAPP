# Core Pkgs
import streamlit as st
import os
import re
import openai 

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
#target
def select_target(df):
    target = st.selectbox("Sélectionnez la target", df.columns.tolist())
    return target

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
         
	# ChatBot 
	if st.checkbox("Classification avec Pycaret"):
		st.subheader("Classification avec Pycaret")

		if st.button("Pycaret"):
		        file = st.file_uploader("Upload file", type=["csv"])
				 
         

              

            
			

	st.sidebar.subheader("Information sur l'application de Bases de connaissances")
	st.sidebar.text("BDC (Bases De Connaissances) Application.")
	st.sidebar.info("Cette Application permet de trouver le sentiment score, les tokens et les lemmas dans une phrase ou texte, les entités de noms, suppression des caractères sspéciaux et Resumé  du texte.!")
	

if __name__ == '__main__':
	main()
