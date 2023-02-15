# Core Pkgs
import streamlit as st
import os
import re
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
@st.cache
def text_analyzer(my_text):
    nlp = spacy.load('en_core_web_sm')
    docx = nlp(my_text)
    # tokens = [ token.text for token in docx]
    allData = [('"Token":{},\n"Lemma":{}'.format(token.text,token.lemma_))for token in docx ]
    return allData

# Function For Extracting Entities
@st.cache
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

def main():
    """ NLP Application sur Streamlit """

    # Title
    st.title(" NLP Application")
    st.subheader("Natural Language Processing Application")
    st.markdown("""
        #### Description
        + C'est une application de Natural Language Processing(NLP) Basée sur du traitement du language Naturel, entre autres nous avons: 
        Tokenization , Lemmatization, Reconnaissance d'entités de nom (NER), Analyse de Sentiments ,  Résumé de texte...! 
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
            if summary_options == 'gensim':
                st.text("Utilisant la méthode Gensim ..")
                summary_result = summarize(message)
                st.write(summary_result)

# Entity Extraction
if st.checkbox("Trouvez les entités de votre texte"):
    st.subheader("Entités de Texte")
    if st.button("Extract"):
        entity_result = entity_analyzer(message)
        st.json(entity_result)

# Sentiment Analysis
if st.checkbox("Analyser le sentiment de votre texte"):
    st.subheader("Analyse de Sentiments")
    message = st.text_area("Entrez le texte","Tapez Ici...")
    if st.button("Analyze"):
        blob = TextBlob(message)
        result_sentiment = blob.sentiment
        st.success(result_sentiment)

# Tokenization and Lemma
if st.checkbox("Tokenization et Lemmatization"):
    st.subheader("Tokenize et Lemmatize")
    message = st.text_area("Entrez le texte","Tapez Ici...")
    if st.button("Analyze"):
        nlp_result = text_analyzer(message)
        st.json(nlp_result)

# Removing special characters and punctuations
if st.checkbox("Nettoyage de Texte"):
    st.subheader("Nettoyez le texte des caractères spéciaux et des ponctuations")
    message = st.text_area("Entrez le texte","Tapez Ici...")
    if st.button("Nettoyez"):
        clean_text = remove_special_characters(message)
        st.text("Texte Nettoyé")
        st.write(clean_text)

if __name__ == '__main__':
    main()

