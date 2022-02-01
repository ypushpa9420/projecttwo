# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 22:00:42 2021

@author: Pushpa Yadav
"""
# import streamlit
import streamlit as st
from io import StringIO
import string
import nltk
import pandas as pd
# =============================================================================
# import bs4 as bs
# import urllib.request
# import re
# =============================================================================


def generate_ngrams(words_list, n):
    ngrams_list = []
 
    for num in range(0, len(words_list)):
        ngram = ' '.join(words_list[num:num + n])
        ngrams_list.append(ngram)
 
    return ngrams_list

def app():
    uploaded_file = st.sidebar.file_uploader("Choose a file", type="txt")
    if uploaded_file is not None:
     
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        #st.write(bytes_data)

        # To convert to a string based IO:
        stringio = StringIO(uploaded_file.getvalue().decode("ISO-8859-1"))
        #st.write(stringio)

        # To read file as string:
        string_data = stringio.read()
        #st.write(string_data)
        genre = st.sidebar.radio("Please choose one option",('EDA / VDA', 'Summary'))
        strAllTexts  = string_data
        

# =============================================================================
#         scraped_data = urllib.request.urlopen('https://en.wikipedia.org/wiki/Reinforcement_learning')
#         article = scraped_data.read()
#         parsed_article = bs.BeautifulSoup(article,'lxml')
#         paragraphs = parsed_article.find_all('p')
#         article_text = ""
#         for p in paragraphs:
#             strAllTexts += p.text
# =============================================================================
        
        #storing number of char
        vTotChars = len(strAllTexts)
        
        #Storing number of lines
        lstAllLines = strAllTexts.split('\r')
        vTotLines = len(lstAllLines)
        #print(lstAllLines)
        
        
        lstTmpWords = []
        for i in range(0,len(lstAllLines)):
            strLine = lstAllLines[i]
            lstWords = strLine.split(" ")
            lstTmpWords.append(lstWords)
        
        # split each line into a list of words
        lstTmpWords = []
        for i in range(0,len(lstAllLines)):
            strLine = lstAllLines[i]
            lstWords = strLine.split(" ")
            lstTmpWords.append(lstWords)

            # merge in single list
        lstAllWords = []    
        for lstWords in lstTmpWords:
            for strWord in lstWords:
                lstAllWords.append(strWord)
    
        vTotWords = len(lstAllWords)

        nLineSmry = int(vTotLines * 0.1)
        #st.text(nLineSmry)
            
        #############################################################
        # compute word freq & word weight
        #############################################################
        from nltk.tokenize import word_tokenize
        lstAllWords = word_tokenize(strAllTexts)

        # Convert the tokens into lowercase: lower_tokens
        lstAllWords = [t.lower() for t in lstAllWords]
        
        # retain alphabetic words: alpha_only
        
        lstAllWords = [t.translate(str.maketrans('','','01234567890')) for t in lstAllWords]
        lstAllWords = [t.translate(str.maketrans('','',string.punctuation)) for t in lstAllWords]
        
        # remove all stop words
        # original found at http://en.wikipedia.org/wiki/Stop_words
        lstStopWords = nltk.corpus.stopwords.words('english')
        lstAllWords = [t for t in lstAllWords if t not in lstStopWords]
        
        # remove all bad words ...
        # original found at http://en.wiktionary.org/wiki/Category:English_swear_words
        lstBadWords = ["arse","ass","asshole","bastard","bitch","bloody","bollocks","child-fucker","cunt","damn","fuck","goddamn","godsdamn","hell","motherfucker","shit","shitass","whore"]
        lstAllWords = [t for t in lstAllWords if t not in lstBadWords]
        
        # remove application specific words
        lstSpecWords = ['rt','via','http','https','mailto']
        lstAllWords = [t for t in lstAllWords if t not in lstSpecWords]
        
        # retain words with len > 3
        lstAllWords = [t for t in lstAllWords if len(t)>3]
        
        # import WordNetLemmatizer
        # https://en.wikipedia.org/wiki/Stemming
        # https://en.wikipedia.org/wiki/Lemmatisation
        # https://blog.bitext.com/what-is-the-difference-between-stemming-and-lemmatization/
        from nltk.stem import WordNetLemmatizer
        # instantiate the WordNetLemmatizer
        wordnet_lemmatizer = WordNetLemmatizer()
        # Lemmatize all tokens into a new list: lemmatized
        lstAllWords = [wordnet_lemmatizer.lemmatize(t) for t in lstAllWords]
        
        # create a Counter with the lowercase tokens: bag of words - word freq count
        # import Counter
        from collections import Counter
        dctWordCount = Counter(lstAllWords)
        
    
        
        # print the 10 most common tokens
        #st.text(dctWordCount.most_common(10))
        
        #print('\n*** Convert To Dataframe ***')
        dfunigramscolle  = pd.DataFrame.from_dict(dctWordCount, orient='index').reset_index()
        dfunigramscolle.columns = ['Word','Freq'] 
        dfunigramscolle = dfunigramscolle.sort_values(by='Freq',ascending=False)
        dfunigramscolle=dfunigramscolle.head(10)
        
        #Biagram

        biagram = generate_ngrams(lstAllWords, 2)
        
        biagramcolle = Counter(biagram)        
                     
        dfbiagramcolle  = pd.DataFrame.from_dict(biagramcolle, orient='index').reset_index()
        dfbiagramcolle.columns = ['Word','Freq']
        dfbiagramcolle = dfbiagramcolle.sort_values(by='Freq',ascending=False)
        dfbiagramcolle = dfbiagramcolle.head(10)

        #Trigram

        trigram = generate_ngrams(lstAllWords, 3)
        
        trigramcolle = Counter(trigram)        
                     
        dftrigramcolle  = pd.DataFrame.from_dict(trigramcolle, orient='index').reset_index()
        dftrigramcolle.columns = ['Word','Freq']
        dftrigramcolle = dftrigramcolle.sort_values(by='Freq',ascending=False)
        dftrigramcolle = dftrigramcolle.head(10)

        
        if genre == 'EDA / VDA' :
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("File Size(MB)", round(uploaded_file.size / (1024 * 1024), 4))
            col2.metric("Words", vTotWords)
            col3.metric("Characters", vTotChars)
            col4.metric("Lines", vTotLines)
            #col11, col12 = st.columns(2)

            #with col11:
            st.title("Top keywords")
            st.dataframe(dfunigramscolle) 
            st.title("Top phrases 2-Word")
            st.dataframe(dfbiagramcolle) 
            st.title("Top phrases 3-Word")
            st.dataframe(dftrigramcolle) 
            
            # st.text("Bar Chart")
            # fig = plt.figure(figsize = (5, 5))
            # plt.barh(dfunigramscolle['Word'], dfunigramscolle['Freq'])
            # plt.xlabel("Number of Freq")
            # plt.ylabel("Words")    
            # #plt.title("Word Freq Count") 
            # st.pyplot(fig)
                
            #with col12:
            st.title("Word Cloud top 30 words")
            #plot word cloud
            # word cloud options
            # https://www.datacamp.com/community/tutorials/wordcloud-python
            #print('\n*** Plot Word Cloud - Top 100 ***')
            import matplotlib.pyplot as plt
            from wordcloud import WordCloud
            
            dftWordCount  = pd.DataFrame.from_dict(dctWordCount, orient='index').reset_index()
            dftWordCount.columns = ['Word','Freq'] 
            dftWordCount = dftWordCount.sort_values(by='Freq',ascending=False)
            dftWordCount=dftWordCount.head(30)
            
            d = {}
            for a, x in dftWordCount[['Word','Freq']].values:
                d[a] = x 
            wordcloud = WordCloud(background_color="white")
            wordcloud.generate_from_frequencies(frequencies=d)
            fig = plt.figure(figsize = (5, 5))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            st.pyplot(fig)

            
        if genre == 'Summary' :
            
            
            
            # word weight = word-count / max(word-count)
            # replace word count with word weight
            max_freq = sum(dctWordCount.values())
            for word in dctWordCount.keys():
                dctWordCount[word] = (dctWordCount[word]/max_freq)
            # weights of words
            
            #############################################################
            # create sentences / lines
            #############################################################
            
            # split scene_one into sentences: sentences
            from nltk.tokenize import sent_tokenize
            lstAllSents = sent_tokenize(strAllTexts)

            
            # convert into lowercase
            lstAllSents = [t.lower() for t in lstAllSents]
            
            # remove punctuations

            lstAllSents = [t.translate(str.maketrans('','','[]{}<>')) for t in lstAllSents]
            lstAllSents = [t.translate(str.maketrans('','','0123456789')) for t in lstAllSents]
            
            # sent score
            dctSentScore = {}
            for Sent in lstAllSents:
                for Word in nltk.word_tokenize(Sent):
                    if Word in dctWordCount.keys():
                        if len(Sent.split(' ')) < 30:
                            if Sent not in dctSentScore.keys():
                                dctSentScore[Sent] = dctWordCount[Word]
                            else:
                                dctSentScore[Sent] += dctWordCount[Word]
            
            
            #############################################################
            # summary of the article
            #############################################################
            # The "dctSentScore" dictionary consists of the sentences along with their scores. 
            # Now, top N sentences can be used to form the summary of the article.
            # Here the heapq library has been used to pick the top 5 sentences to summarize the article
            import heapq
            lstBestSents = heapq.nlargest(nLineSmry, dctSentScore, key=dctSentScore.get)
            # for vBestSent in lstBestSents:
            #     st.write(vBestSent)
            
            # # final summary
            # strTextSmmry = '. '.join(lstBestSents) 
            # strTextSmmry = strTextSmmry.translate(str.maketrans(' ',' ','\n'))
            # st.write(strTextSmmry)
            
            
            # #############################################################
            # # gensim
            # # https://radimrehurek.com/gensim_3.8.3/summarization/summariser.html
            # #############################################################
            
            # #import gensim
            # from gensim.summarization.summarizer import summarize
            
            # # pass the document along with desired word count to get the summary
            # my_summary = summarize(strAllTexts, word_count=200)
            # st.write("1")
            # st.write(my_summary)
            
            # #############################################################
            # # lexrank
            # # https://iq.opengenus.org/lexrank-text-summarization/
            # #############################################################
            
            # #import sumy
            # from sumy.summarizers.lex_rank import LexRankSummarizer
            
            # # plain text parsers since we are parsing through text
            # from sumy.parsers.plaintext import PlaintextParser
            # from sumy.nlp.tokenizers import Tokenizer
            
            # # parser object with AllTexts
            # parserObject = PlaintextParser.from_string(strAllTexts,Tokenizer("english"))
            # # summarizer object
            # summarizer = LexRankSummarizer()
            # # create summary
            # my_summary = summarizer(parserObject.document,7)
            # st.write("2")
            # st.write(my_summary)
            
            
            # #############################################################
            # # luhn summary
            # # https://iq.opengenus.org/luhns-heuristic-method-for-text-summarization/
            # #############################################################
            
            # #import sumy
            # from sumy.summarizers.luhn import LuhnSummarizer
            
            # #Plain text parsers since we are parsing through text
            # from sumy.parsers.plaintext import PlaintextParser
            # from sumy.nlp.tokenizers import Tokenizer
            
            # # parser object with AllTexts
            # parserObject = PlaintextParser.from_string(strAllTexts,Tokenizer("english"))
            # # summarizer object
            # summarizer = LuhnSummarizer()
            # # create summary
            # my_summary = summarizer(parserObject.document,7)
            # st.write("3")
            # st.write(my_summary)
            
            
            #############################################################
            # lsa summary
            # https://iq.opengenus.org/latent-semantic-analysis-for-text-summarization/
            #############################################################
            
            #import sumy
            ##We're choosing a plaintext parser here, other parsers available for HTML etc.
            from sumy.parsers.plaintext import PlaintextParser
            from sumy.nlp.tokenizers import Tokenizer
            ##We're choosing Luhn, other algorithms are also built in
            from sumy.summarizers.lsa import LsaSummarizer as Summarizer
            from sumy.nlp.stemmers import Stemmer
            from sumy.utils import get_stop_words
            
            # parser object with AllTexts
            parserObject = PlaintextParser.from_string(strAllTexts,Tokenizer("english"))
            stemmer = Stemmer("english")
            summarizer = Summarizer(stemmer)


            # summarizer object
            
            st.write("**********Summary of your document**********")
            summarizer.stop_words = get_stop_words("english")

            summaryList = []
            ##Summarize the document with 5 sentences
            my_summary = summarizer(parserObject.document, nLineSmry) # sentence count set to 10
            summaryList = list(my_summary)
            for i in summaryList:
                st.write(str(i).strip())
                
