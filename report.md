# Final Report

## 1. DATASET
The provided dataset was taken from [FinanceInc 'auditor_sentiment' on Hugging Face](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment). 
<br></br>
As alluded to in the title, it contains small text clippings of financial news. The dataset is roughly 3500 rows with these bits of business news and an associated label in its training set. The labels [0, 1, 2] correspond with [Negative News, Neutral News, and Positive News]. As an example, 
>TEXT: "Altia 's operating profit jumped to EUR 47 million from EUR 6.6 million.", LABEL: "2"

In other words, the following statement has positive sentiment.
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
> Provides the 10 tokens with the largest log-likelihood ratios out of all tokens in the 3,500 documents found within the training corpus. This means these tokens have the highest difference between its likelihood from one label compared to the other two. From the graph, this means that "slipped" has the highest likelihood of happening with its label ("negative") compared to the other two labels. This graph was created in R.

### Topic Modeling: Top 10 Topics 
![Topics](https://github.com/brandonowens24/NLP_HW1/blob/main/images/Topics.png)<br>
> **Green Table:** The ten most probable topics from the dataset (human-labeled). Each topic has the ten most probable words for the topic and their probabilites of occurring in these topics. The dataset was tokenized, but not stemmed, and put into BoW based on count. <br>
> **Orange Table:** The ten most probable topics from the dataset (human-labeled). Each topic has the ten most probable words for the topic and their probabilites of occurring in these topics. The dataset was tokenized, including stemming, and put into BoW based on count with IDF. 

## Topic Modeling: Top 3 Topics for Each Classifier
![Top Document Topics](https://github.com/brandonowens24/NLP_HW1/blob/main/images/TopDocTopics.png)<br>
>**Left Table:** Tokenization without stemming, BoW without IDF. Provides the top three topics for each label. Gives the probability of that topic occurring for a document of that label type. These topics correspond to the top 10 topics found in the documents from the section above `Topic Modeling: Top 10 Topics`. For example, a document with negative classification has a 10.3% chance to fall into the "production" topic. <br>
>**Right Table:** Tokenization with stemming, BoW with IDF. Provides the top three topics for each label. Gives the probability of that topic occurring for a document of that label type. These topics correspond to the top 10 topics found in the documents from the section above `Topic Modeling: Top 10 Topics`. For example, a document with negative classification has a 24.4% chance to fall into the "comparative outcomes" topic.

## 4. DISCUSSION

## Dataset Findings
### Log-Likelihood Ratio
After doing the Naive-Bayes analysis and determining the log-likelihood ratios and their most likely classifier for each token, my graph has the exact results that you would expect. Words like "slipped", "warning", "decreased", and "drop" have the highest likelihood differences and thus have the highest probability of occurring for the "negative" label. This makes sense because all of these words have a poor connotation -- especially in terms of the financial market. The same is true, but opposite for the positive words like "improved", "grew", "narrowed", "positive", etc. An interesting finding was that among the top ten log-likelihood ratios, no tokens were deemed to be "neutral". This is most likely because a lot of the "neutral" labelled statement had both negative and positive remarks within them.

### Topic Modelling
After applying Latent-Dirichlet Allocation on my chosen dataset, I was a bit disappointed at the similarity of topics. Their word coherence to form a topic based on words appearing together makes sense that most topics would be related and have some repeat words, but I found it very difficult to manually assign topic names when they all fell into the same realm. This issue may have been resolved if I could have found a financial dataset with full news articles as the documents instead of 1-2 sentence bodies of text. Additionally, topics were more distinguishable when I applied the inverse-document-frequency argument. This makes sense because words that appear over the course of more documents are weighed less, and thus it is easier to distinguish more unique cohesive words.  <br>

Additionally, when applying Topic Modelling to find the top 3 topics for each label, I was more satisfied with the inverse-document-frequency argument once again. Because the topics were more distinguishable, I am able to see better trends between documents labeled "negative" and "positive" based on their topics. For instance, both focused on comparative outcomes and balance sheet statements, but negative documents tended to focus more on mentioning words of currency like "million", "euro", "mln", "eur", etc. The neutral label was also better defined as including more news words that didn't show benefit or decrement like "product", "system", "company", "news", etc. 

## Lessons Learned
### Naive Bayes
The naive bayes model that we created in lecture was extremely useful at determining logical log-likelihood-ratios. With that being said, my NB model takes forever to train (~3 minutes per 1,000 documents). It was easier to just save the json dictionary of the llr_ratios instead of having to run my funciton and train the model every single time. Perhaps using a library would be much faster for future projects.

### Topic Modelling
I realized pretty quickly into topic modelling that I should have chosen a more diverse dataset that wasn't all taken from the same news source. They had a tendency of writing their updates in a Twitter-like format with short sentences that made distinguishing the topics themselves very difficult. It would have been much cooler and more useful to see what sectors and companies had a positive/negative year based on their association with a label in the news. In the future, I need to do a better job of finding a dataset with larger bodies of text for the corpora. 

### Libraries/Code
I was able to familiarize myself with gensim and some of the cool features they had in there. Specifically that the id2token we created in class for our Naive-Bayes model actually could be plugged back in as a dictionary for the unsupervised learning! I also got more practice with Python -- specifically some problems I was having that was solved with the enumerate function, and other issues like grabbing inner_keys from a dictionary of dictionaries. 

### For the Future
Because of time conflicts, I settled for modelling my visualizations in R and Microsoft Excel. I would have much rather preferred to automate the entire project, but the time it would have taken was not worth it for me. For future projects I would like to try to automate them, but I am learning quickly that sometimes the time is not worth the extra effort -- especially for a singular homework assignment. This is something I struggle with and need to better weigh my time management vs. value of the project I am working on. 
