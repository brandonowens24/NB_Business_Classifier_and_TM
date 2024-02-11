# Final Report

## 1. DATASET
The provided dataset was taken from [zeroshot's 'twitter-financial-news-sentiment' on Hugging Face](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment). 
<br></br>
As alluded to in the title, it contains finance-related news tweets and their sentiment. It contains roughly 10,000 rows of financial news tweets and an associated label in its training set. The labels [0, 1, 2] correspond with [Bearish News, Bullish News, and Neutral News]. As an example, 
>TEXT: "WORK, XPO, PYX and AMKR among after hour movers", LABEL: 2
In other words, the tweet has neutral sentiment.
<br></br>
This could be useful if an analyst, user, or owner of some commodity of a company or multiple companies has a very limited amount of time to analyze the public sentiment and news occurring with the companies that they are involved with. 
<br></br>

![Dataset](https://github.com/brandonowens24/NLP_HW1/blob/main/images/Dataset.png)


## 2. METHODOLOGY
After importing necessary libraries, I grabbed the training data from the datasets, and set specific command line arguments to be executed. These options allow the operator to choose which type of bag of words format they would like to have when creating their bag of words matrix and preprocessing. 
<br></br>
When computing document vectors, I was assisted with some of Dr. Wilson's in class code and followed his format of tokenization -- excluding specific characters such as emails, urls, stopwords, words under two characters, and excluded digits. The digits argument was important for my corpus because there are random numbers that shouldn't be associated with any sort of sentiment or likelihood. For instance, if "10%" only occurred in the training set with a 'bearish' label, then it would register as being more likely bearish in the test set. But the 10% itself is completely arbitrary to the label -- a company could've risen or fallen 10% -- the number itself doesn't have a negative connotation. 
<br></br>
Training the naive-bayes model drew upon some of our Colab code; however, implementation of grabbing the log-likelihood ratio required me to compute the likelihood for each word in each context, comparing the likelihoods, picking the most likely context, combining the numerator and denominator counts for the unlikely options, and then computing the log-likelihood ratio after. The reason I applied smoothing at the end is because if I took the lesser likelihoods that were smoothed, I would have been combining two smoothed values -- adding 2x the vocabulary to the denominator count. This would have signifiantly decreased the probability of discovering specific tokens in lesser likelihoods. 
<br></br>
Using the gensim library, I was able to apply an lda model to the corpus of documents and use built in functions to grab the top ten frequently occurring topics. I then iterated through all of the documents in my corpus and grabbed the probabilities (summing to 1) that each topic would be contained in the document. Afterwards, I grabbed all training documents with each specified label and averaged their topic probabilities. I then reported the top three for each classifier. This shows the probabilities of the top three most probable topics occurring based on the label the training tweet is classified as. 
<br></br>
All data returned from this script was delivered via command-line text; thus, all analysis was completed in RStudio and Microsoft Excel.

## 3. RESULTS AND ANALYSIS
### Naive-Bayes Log-Likelihood Ratios for Top 10 Tokens
![Top 10](https://github.com/brandonowens24/NLP_HW1/blob/main/images/Top_llrs.png)<br>
Provides the 10 tokens with the largest log-likelihood ratios from all 10,000 documents found within the training corpus and identifies their associated sentiment classifier that is most likely. This graph was created in R.


## 4. DISCUSSION

### Dataset Findings

### Lessons Learned
