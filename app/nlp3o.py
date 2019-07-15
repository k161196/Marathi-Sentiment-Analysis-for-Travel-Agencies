#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .inputhandler import readStopwords
import nltk
# import cltk
import gensim
import pandas as pd
from gensim.test.utils import datapath
from gensim.models import KeyedVectors
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from cltk.tokenize.sentence import TokenizeSentence
tokenizer = TokenizeSentence('marathi')
stemmer = LancasterStemmer()

class TextAnalyser:
        # stemming values
    NO_STEMMING = 0
    STEM = 1
    LEMMA = 2
    
    'Text object, with analysis methods' 
    def __init__(self, inputText, language = "MR"):
        self.text = inputText
        self.tokens = []
        self.sentences = []

        self.adj = pd.read_csv("app/static/adjPol.csv")
        self.adver = pd.read_csv("app/static/advPol.csv")
        self.neg = pd.read_csv("app/static/negPol.csv")
        # self.inp = 'खूप नाही छान '
        self.data = inputText.split()
        self.adv_counter=0
        self.adj_counter=0
        self.adv_sum=0
        self.adv_sub=0
        self.adv_inten=0
        self.adj_sum=0
        self.adj_sub=0
        self.adj_inten=0
        self.adj_con=False
        self.adv_con=False
        self.neg_con=False
        self.training_data = []
        self.corpus_words = {}
        self.class_words = {}
        self.training_data.append({"class":"स्थळ", "sentence":"कर्जतचा धबधबा हे एक प्रेक्षणीय स्थळ आहे"})
        self.training_data.append({"class":"स्थळ", "sentence":"हे अभयारण्य वन्यजीवांसाठी उत्कृष्ट अधिवास असल्याचे निदर्शनास आल्यामुळेच या जंगलाला वन्यप्राण्यांसाठी ‘संरक्षित’ केले"})
        self.training_data.append({"class":"स्थळ", "sentence":"लाखभर पक्ष्यांचे आश्रयस्थान असलेला ठाणे खाडी परिसर राज्य सरकारने ‘फ्लेमिंगो अभयारण्य’ म्हणून जाहीर केला आहे"})
        self.training_data.append({"class":"स्थळ", "sentence":"सह्याद्रीच्या पर्वत रांगांमध्ये वसलेल्या पश्चिम घाटामध्ये असलेले निसर्गरम्य ठिकाण म्हणजे ताम्हिणी घाट"})

        self.training_data.append({"class":"प्रवास", "sentence":"हा प्रवास लोणावळ्याहून सुरू होतो"})
        self.training_data.append({"class":"प्रवास", "sentence":"पुणे जिल्ह्यातील तालुक्याचे ठिकाण असलेल्या भोर गावी जाण्यासाठी स्वारगेटहून बस आहेत"})
        self.training_data.append({"class":"प्रवास", "sentence":"सफारी हाच या जंगलसृष्टीच्या स्वर्गासुखाचा मार्ग"})
        self.training_data.append({"class":"प्रवास", "sentence":"एकावेळी  गाड्या  गाडी सोडण्यात येतात"})

        self.training_data.append({"class":"प्रवास", "sentence":"नागपूर-भंडारा हे 65 कि.मी. अंतर अंतर अंतर"})
        print ("%s sentences of training data" % len(self.training_data))
        print('\n'.join(map(str, self.training_data)))
        
        self.word_vectors = KeyedVectors.load_word2vec_format(datapath('mr.vec'), binary=False)
        classes = list(set([a['class'] for a in self.training_data]))
        for c in classes:
            self.class_words[c] = [] 
        for data in self.training_data:
            for word in tokenizer.tokenize(data['sentence']):
                stemmed_word = word
                print(stemmed_word)
                if stemmed_word not in self.corpus_words:
                    self.corpus_words[stemmed_word] = 1
                else:
                    self.corpus_words[stemmed_word] += 1
                self.class_words[data['class']].extend([stemmed_word])
        print ("Corpus words and counts: %s \n" % self.corpus_words)
        print ("Class words: %s" % self.class_words)
        
        self.suffixes = {
    1: ["े", "ू", "ु", "ी", "ि", "ा" , " ौ"  , " ै" ,  "स" , "ल" , "त" , "म" , "अ" ,  "त"],
    2: ["नो" , "तो" , "ने" , "नी" , "ही" , "ते" ,"या" , "ला" , "ना" , "ऊण" , "शे" , "शी" , "चा" , "ची" , "चे", "ढा" , "रु" , "डे" ,  "ती" , "ान" , " ीण" , "डा" , "डी" , "गा" , "ला" , "ळा" , "या" , "वा" , "ये" , "वे" , "ती" ],
    3: ["शया" , "हून"],
    4: [" ुरडा"],
}
        
        self.language = language
        self.stopWords = set(readStopwords(language))

    # def topicClass(self):
    def sentiScore(self):
        for temp in self.data:
            print(temp)
            adj_check = self.adj.loc[self.adj['word'].str.contains(temp)]
            adver_check = self.adver.loc[self.adver['word'].str.contains(temp)]
            neg_check = self.neg.loc[self.neg['word'].str.contains(temp)]
            # nm checking
            nm=self.adver.index[ self.adver['word'].str.contains(temp)].tolist()
            nmadj=self.adj.index[ self.adj['word'].str.contains(temp)].tolist()
            nmneg=self.neg.index[ self.neg['word'].str.contains(temp)].tolist()
            if not adj_check.empty:
                print( "adj")
            # if not ver_check.empty:
            #     print "verb"
            if not adver_check.empty:
                print ("adverb")
            if not neg_check.empty:
                self.neg_con=True
                print ("neg")
            # checking of valuses and calculatng 

            if(len(nm)>0):
                self.adv_con=True
                loc=self.adver.loc[nm[0],'pol']
                sub=self.adver.loc[nm[0],'sub']
                inten=self.adver.loc[nm[0],'int']
                # print(inten)
                self.adv_sum=self.adv_sum+loc
                self.adv_sub=self.adv_sub+sub
                self.adv_inten=self.adv_inten+inten
                
                # print(adv_sum)
                # print(nm)
                self.adv_counter+=1
            if(len(nmadj)>0):
                self.adj_con=True
                loc=self.adj.loc[nmadj[0],'pol']
                sub=self.adj.loc[nmadj[0],'sub']
                inten=self.adj.loc[nmadj[0],'int']
                # print(inten)
                self.adj_sum=self.adj_sum+loc
                self.adj_sub=self.adj_sub+sub
                self.adj_inten=self.adj_inten+inten
                # print(adj_sum)
                # print(nmadj[0])
                self.adj_counter+=1
            # end for 
        # aveage cal
        if self.adv_con==True: 
            self.adv_avg=self.adv_sum/self.adv_counter
            self.adv_sub_avg=self.adv_sub/self.adv_counter
            self.adv_inten_avg=self.adv_inten/self.adv_counter
        if self.adj_con==True:
            self.adj_sub_avg=self.adj_sub/self.adj_counter
            self.adj_inten_avg=self.adj_inten/self.adj_counter
            self.adj_avg=self.adj_sum/self.adj_counter
        # polarity cal
        polariy=0
        if self.adv_con and self.adj_con:
            if self.neg_con:
                polariy = (-0.5)*1/(self.adv_inten_avg)*self.adj_avg
                print('neg+adv+adj')
            else:
                polariy = self.adv_sum+self.adj_sum
                print('adv+adj')
            if polariy>1:
                polariy=1
            elif polariy < -1:
                polariy=-1
                # print('hi')
        # elif adv_con and adj_con:
        #     polariy = adv_sum+adj_sum
        #     if polariy>1:
        #         polariy=1
        #     elif polariy < -1:
        #         polariy=-1
        #     # print(polariy)
        #     print('adv+adj')
        elif self.adj_con and self.neg_con:
            print('adj+neg')
            polariy=(-0.5)*self.adj_avg
            if polariy>1:
                polariy=1
            elif polariy < -1:
                polariy=-1
        elif self.adj_con==True:
            print('adj')
            polariy=self.adj_avg
            if polariy>1:
                polariy=1
            elif polariy < -1:
                polariy=-1
        elif self.adv_con==True:
            print('adv')
            polariy=self.adv_avg
            if polariy>1:
                polariy=1
            elif polariy < -1:
                polariy=-1
        # if adj_con==True:
        #     print('hiii')
        print(polariy)
        return polariy
    def w2vDis(self):
        return self.word_vectors.most_similar(self.tokens[1])
    def hi_stem(self,word):
        for L in 4, 3, 2, 1:
            if len(word) > L + 1:
                for suf in self.suffixes[L]:
                    if word.endswith(suf):
                        return word[:-L]+'|'+word[-L:]
                        #return word[:-L]
        return word+'|'+ 'null'
    def calculate_class_score(self,sentence, class_name, show_details=True):
        score = 0
        for word in nltk.word_tokenize(sentence):
            if stemmer.stem(word.lower()) in self.class_words[class_name]:
                score += (1 / self.corpus_words[word])

                if show_details:
                    print ("   match: %s (%s)" % (stemmer.stem(word), 1 / self.corpus_words[stemmer.stem(word)]))
        return score
    def classify(self,sentence):
        high_class = None
        high_score = 0
    
        for c in self.class_words.keys():
        
            score = self.calculate_class_score( sentence, c, show_details=False)
        
            if score > high_score:
                high_class = c
                high_score = score

        return high_class, high_score
    
    def classWords(self):
        return  self.class_words    
    def corpusWords(self):
        return  self.corpus_words
    def sentacePrint(self):
        return  "\n".join(map(str, self.training_data))
    def sentaceLen(self):
        return  len(self.training_data)
    def length(self):
        """ return length of text in chars """
        return len(self.text)
        
    def tokenise(self):
        """ split the text into tokens, store and returns them """
        self.tokens = self.text.split() # split by space; return a list
    
    def tokeniseNLTK(self):
        self.tokens = nltk.word_tokenize(self.text)
    
    def getTokens(self):
        """ returns the tokens (need to be previously tokenised) """
        return len(self.tokens)
    
    def splitSentences(self):
        self.sentences = nltk.sent_tokenize(self.text)
    
    def getSentences(self):
        """ returns the sentences (need to be previously split) """
        return len(self.sentences)
    # demo piechart 
    def demo_piechart(self):
        """
        pieChart page
        """
        xdata = ["Apple", "Apricot", "Avocado", "Banana", "Boysenberries", "Blueberries", "Dates", "Grapefruit", "Kiwi", "Lemon"]
        ydata = [52, 48, 160, 94, 75, 71, 490, 82, 46, 17]

        extra_serie = {"tooltip": {"y_start": "", "y_end": " cal"}}
        chartdata = {'x': xdata, 'y1': ydata, 'extra1': extra_serie}
        charttype = "pieChart"

        data = {
            'charttype': charttype,
            'chartdata': chartdata,
        }
        return data
    # def removePunctuation(self):
    #     """ remove punctuation from text"""
    #     import re

    #     self.text = re.sub(r'([^\s\w_]|_)+', '', self.text.strip()) # remove punctuation

    def removeStopWords(self):
        """ remove stop words from text.
        Stopwords are defined at initialisation based on language.
        Only one set of stopwords is possible (no language mix)"""
        self.tokens = [token for token in self.tokens if token not in self.stopWords]
    

    def lemmatiseVerbs(self):

        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(w,'v') for w in self.tokens]

        #return len(set(lemmatized))
    
    def stemTokens(self):
        porter = nltk.PorterStemmer()
        return [porter.stem(t) for t in self.tokens]

    def preprocessText(self, removeStopWords=True):
        """ pre-process the text:
            1. lower case 
            2. remove punctuation
            3. tokenise the text
            4. remove stop words"""
            
        self.splitSentences()
        
        # if lowercase:
        #     self.text = self.text.lower()
    
        # self.removePunctuation()

        self.tokenise()  
        
        if removeStopWords:
            self.removeStopWords()
        
            # if stemming then do it
        # if stemming == TextAnalyser.STEM:
        #     self.tokens = self.stemTokens()
        # elif stemming == TextAnalyser.LEMMA:
        #     self.tokens = self.lemmatiseVerbs()


        
    def uniqueTokens(self):
        """ returns the unique tokens"""
        return (len(set(self.tokens)))
    
    def getMostCommonWords(self, n=10):
        """ get the n most common words in the text;
        n is the optional paramenter"""
        from collections import Counter

        wordsCount = Counter(self.tokens) # count the occurrences
    
        return wordsCount.most_common()[:n]
    
    def getMostCommonWordsNLTK(self, n=10):
        """ get the n most common words in the text;
        n is the optional paramenter"""
        # Calculate frequency distribution
        fdist = nltk.FreqDist(self.tokens)

        # Output top 20 words

        return fdist.most_common(n)
    
    def findLongest(self):
      #Find the longest word in text1 and that word's length.
        longest = max(self.tokens, key=len)
        return (longest, len(longest))


    def findSentences(self):
    
        sentences = nltk.sent_tokenize(self.text)
        return len(self.tokens) / len(sentences)
    
    
    