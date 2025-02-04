#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from app import app
from flask import render_template, flash, request
from .forms import InputTextForm
from .nlp3o import TextAnalyser
from .inputhandler import getSampleText

    # Submit button in web for pressed
@app.route('/', methods=['POST'])
@app.route('/index', methods=['POST'])
def manageRequest():

      # some useful initialisation
    theInputForm = InputTextForm()
    userText = "किती गाड्या"
    typeText = "किती गाड्या"
    language = "MR"
    
      # POST - retrieve all user submitted data

    inputFromBook = request.form['book'] # which text?

    # DEBUG flash('the book selected is: %s' % inputFromBook)

    if inputFromBook == "mobydick":
        userText, typeText = getSampleText(1)
        language = "EN"

    elif inputFromBook == "marinetti":
        userText, typeText = getSampleText(2)
        language = "IT"

    elif inputFromBook == "urteil":
        userText, typeText = getSampleText(3)
        language = "DE"

    else:
        if theInputForm.validate_on_submit():
            userText = theInputForm.inputText.data
            typeText = "Your own text"
            language = "MR" # which language?

    # DEBUG flash('read:  %s' % typeText)
    
    stemmingEnabled = request.form.get("stemming")
    stemmingType = TextAnalyser.NO_STEMMING
    
    if stemmingEnabled:
        if request.form.get("engine"):
            stemmingType = TextAnalyser.STEM
        else:
            stemmingType = TextAnalyser.LEMMA
    else:
        stemmingType = TextAnalyser.NO_STEMMING


    #flash('read:  %s' % stemmingEnabled)


          # Which kind of user action ?
    if 'TA'  in request.form.values():
            # GO Text Analysis

               # start analysing the text
        myText = TextAnalyser(userText, language) # new object
        print("d")
        myText.preprocessText(removeStopWords = True)

               # display all user text if short otherwise the first fragment of it
        if len(userText) > 99:
            fragment = userText[:99] + " ..."
        else:
            fragment = userText

              # check that there is at least one unique token to avoid division by 0
        if myText.uniqueTokens() == 0:
            uniqueTokensText = 1
        else:
            uniqueTokensText = myText.uniqueTokens()

        print(fragment)
              # render the html page
        return render_template('results.html',
                           title='Sentiment Analysis',
                           inputTypeText = typeText,
                           originalText = fragment,
                           numChars = myText.length(),
                           numSentences = myText.getSentences(),
                           numTokens = myText.getTokens(),
                           uniqueTokens = uniqueTokensText,
                           sentaceLen =myText.sentaceLen(),
                           sentacePrint =myText.sentacePrint(),
                           classWords =myText.classWords(),
                           corpusWords =myText.corpusWords(),
                           topicScore  =myText.classify(fragment),
                           stemMarathi  =myText.hi_stem(fragment),
                            sentiScore=myText.sentiScore(),
                           w2vDis  =myText.w2vDis(),
                           commonWords = myText.getMostCommonWords(10))

    else:
        return render_template('future.html',
                           title='Not yet implemented')



  # render web form page
@app.route('/', methods=['GET'])
@app.route('/index', methods=['GET'])
def initial():
      # render the initial main page
    return render_template('index.html',
                           title = 'Your input',
                           form = InputTextForm())

@app.route('/results')
def results():
    return render_template('index.html',
                           title='sentiment Analysis')

  # render about page
@app.route('/about')
def about():
    return render_template('about.html',
                           title='About ')
