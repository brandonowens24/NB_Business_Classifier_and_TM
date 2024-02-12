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


## Arguments
* `'-h' '--help'` Argument help <br>
* `'-s' '--stem'` Stems corpus to root word using nltk Port Stemmer<br>
* `'-i' '--idf'` Multiplies BoW matrix by Inverse-Document-Frequency 
