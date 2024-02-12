# Corpus_Analysis_HW1
**'FinanceInc's Auditor Sentiment` dataset from HuggingFace analyzed. Top Log-Likelihood Ratio tokens returned, top ten topics returned(based on conditions), and top topics for each classifier returned.**

## Files / Folders
* **requirements.txt** - Contains necessary packages to run script
* **analysis.py** - Script that returns top LLR tokens, top topics, and top label topics
* **report.md** - Post-project report on process and learning
* **sorted_llr_dictionary.json** - Saved dictionary with each token, classifier, and llr
* **images/** - Contains images used in report

## Instructions
* Install necessary requirements found in `requirements.txt` into your environment <br>
* `$ python analysis.py --arguments` <br>
* Output returned as text
<br></br>

**LLR Dictionary already saved -- Naive Bayes model already stored** <br>
Feel free to add your own code to save your preferred LDA model.


## Arguments
* `'-h' '--help'` Argument help <br>
* `'-s' '--stem'` Stems corpus to root word using nltk Port Stemmer<br>
* `'-i' '--idf'` Multiplies BoW matrix by Inverse-Document-Frequency<br>
* `'-u' '--update'` Updates Naive Bayes LLR dictionary. Only necessary if DB is updated. Default is False. 
* `vector choice: "count", "binary", "frequency"` Chooses which term-frequency format for BoW. Default is count

## Dataset Specifics
* Training set used ~3500 rows
* Composed of `'sentence'` column with text and `'label'` column with classifier
    * Labels [0, 1, 2] correspond with ["Positive Sentiment", "Neutral Sentiment", "Negative Sentiment"]
