from datasets import load_dataset
dataset = load_dataset("FinanceInc/auditor_sentiment")
from tqdm import tqdm
import collections
import math
import nltk
from nltk import PorterStemmer
from nltk.corpus import stopwords; nltk.download('stopwords')
stopwords = set(stopwords.words('english'))
import string
import re
import numpy as np
import argparse
from scipy.sparse import csr_matrix
import gensim
import json

email_pat = re.compile(r"\S+@\S+\.\S+")
url_pat = re.compile("^https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)$")

# Grab command-line arguments
def from_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('vector', default='count', nargs='?', choices=['count', 'binary', 'frequency'], help='Choose format for box of words. Default is count.')
    parser.add_argument('-i', '--idf', action='store_true', help="Uses Inverse Document Frequency")
    parser.add_argument('-s', '--stem', action='store_true', help="Applied Porter Stemmer to tokens")
    args = parser.parse_args()

    return args

# Separate individual documents being passed into a list of tokens
def tokenize(document, stem='False'):
    doc_tokens = []
    # use nltk sentence tokenization
    sentences = nltk.sent_tokenize(document)
    for sentence in sentences:
        # use nltk word tokenization
        # remove email addresses
        sentence = re.sub(email_pat,'',sentence)
        sentence = re.sub(url_pat,'',sentence)
        sent_tokens = nltk.word_tokenize(sentence)
        # remove punctuation
        sent_tokens = [word.translate(str.maketrans('', '', string.punctuation)) for word in sent_tokens]
        # lowercase and remove empty strings, stopwords, and numbers (all punctuation will become empty after previous line)
        sent_tokens = [word.lower() for word in sent_tokens if word]
        sent_tokens = ([word for word in sent_tokens if
                            word not in stopwords
                            #and word in vocab
                            and not re.search(r'\d+', word)
                            and not re.search(r'https\S?', word)
                            and len(word) > 2])
        if stem:
            porter = PorterStemmer()
            sent_tokens = ([porter.stem(word) for word in sent_tokens])
        # either use char ngrams or full words
        doc_tokens += sent_tokens
    return doc_tokens

# Takes collection of documents and converts them into a B.O.W matrix depending on arguments
def compute_doc_vectors_(documents, tf_type='count', use_idf='False', stem='False'):
    token2id = {}
    current_next_id = 0
    document_dicts = []

    print("Tokenizing Corpora:")
    for document in tqdm(documents):
        # Tokenize each document individually
        tokens = tokenize(document, stem)
        document_dict = collections.defaultdict(int)

        for token in tokens:
            if token not in token2id:
                # This can serve as a list of the vocab with a specific id string
                token2id[token] = current_next_id
                current_next_id += 1
            token_id = token2id[token]
            document_dict[token_id] += 1
        document_dicts.append(document_dict)
    
    vectors = []
    print("Converting to Matrix: ")
    for document_dict in tqdm(document_dicts):
        vector = [document_dict[token_id] for token_id in range(current_next_id)]
        # Vectors are composed of the counts of each token from each document
        vectors.append(vector)
    
    matrix = np.array(vectors, dtype='float64')

    if tf_type == 'count':
        pass
    elif tf_type == 'binary':
        matrix = np.where(matrix > 0, 1, 0)
    elif tf_type == 'freq':
        row_sums = np.sum(matrix, axis=1)
        matrix = matrix/row_sums
    else:
        raise ValueError("vector type must be count, binary, freq")
    if use_idf:
        num_docs = matrix.shape[0]
        df_scores = np.sum(np.where(matrix > 0, 1, 0), axis = 0)
        idf_scores = np.log(num_docs / df_scores)
        matrix = matrix * idf_scores

    # Convert with Scipy to a sparse matrix
    matrix = csr_matrix(matrix)
    id2token = {id: tok for tok, id in token2id.items() }

    return matrix, id2token

# Use Naive Bayes to take matrix and comput log-likelihood ratios for each word with each classifier tag
def train_nb(training_labels, matrix, id2token):
    num_mat_rows = matrix.shape[0]
    num_mat_col = matrix.shape[1]
    labels = [0, 1, 2]
    word_with_label_counts = {}
    total_words_with_label = collections.defaultdict(int)

    # Go through each training corpus provided and the total word counts for labels and the counts of specific words for labels
    print("Training Naive Bayes:")
    for doc in tqdm(range(num_mat_rows)):
        label = training_labels[doc]
        # Collect unique sentiments
        if label not in word_with_label_counts:
            word_with_label_counts[label] = collections.defaultdict(int)
        
        total_words_with_label[label] += matrix[doc, :].sum(axis=1)

        for col in range(num_mat_col):
            word = id2token[col]
            # With the Scipy Matrix, some rows and columns do not exist
            word_with_label_counts[label][word] += matrix[doc, col]

    # Calculate likelihoods
    likelihood_comp = {}
    llr = {}
    for col in range(num_mat_col):
        word = id2token[col]
        likelihood_comp[word] = {} 
        llr[word] = {}
        numerator = 0
        denominator = 0 
        max_val = float('-inf')
        most_likely_label = None
        # Calculate the likelihood for each label based on a given word. Not smoothed    
        for label in labels:
            try:
                numerator = word_with_label_counts[label][word]
            except KeyError:
                numerator = 0
            denominator = total_words_with_label[label]
            likelihood_comp[word][label] = ((numerator + 1) / (denominator + num_mat_col))
        
            # Find which label had the highest likelihood
            value = likelihood_comp[word][label]
            if value > max_val:
              
              max_val = value
              most_likely_label = label
        
        # Grab labels that are not most likely
        other_likely = [label for label in labels if label != most_likely_label]
        numerator = 0
        denominator = 0
        log_unlikelihood = 0
        log_likelihood = 0
        # Have to re-grab the num/denom and then apply smoothing because if I just combined them, they woulud have double the vocab in the denominator
        for label in other_likely:
            try:
                numerator += word_with_label_counts[label][word]
            except KeyError:
                pass
            denominator += total_words_with_label[label]

        numerator = numerator.item()
        denominator = denominator.item()
        max_val = max_val.item()

        log_unlikelihood = math.log((numerator + 1) / (denominator + num_mat_col))
        log_likelihood = math.log(max_val) 
        
        # Creates dictionary with each word, their most likely label, and the llr
        llr[word][most_likely_label] = log_likelihood - log_unlikelihood

    sorted_llr = dict(sorted(llr.items(), key=lambda x: list(x[1].values()), reverse=True))

    with open('sorted_llr_dictionary.json', 'w') as file:
        json.dump(sorted_llr, file)

    return sorted_llr

# Takes the top 10 llr from the corpus and prints them out.
def print_top_10_llr(sorted_llr):
    appended_sorted_llr = {k: sorted_llr[k] for k in list(sorted_llr)[:10]}
    print("Top Ten Tokens with Highest LLRs: ")
    for key, value in appended_sorted_llr.items():
        print(f'{key:<15}{value}')

# Creates an LDA Model, which finds the top 10 corpus topics, and then determines the probability of these topics in each document
def model_topics(matrix, id2token):
    corpus = gensim.matutils.Sparse2Corpus(matrix, documents_columns=False)
    lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2token, num_topics=10)
    # lda.save('lda.model')
    
    # lda = gensim.models.ldamodel.LdaModel.load('lda.model')

    lda_topics = lda.print_topics(num_topics=10, num_words=10)

    print("Top Ten Topics in Corpus: ")
    for topic in lda_topics:
        print(topic)


    document_topic_probs = {}
    print("Determining Document Topic Probabilities:")
    for i, doc in tqdm(enumerate(corpus)):

        doc_topics = lda.get_document_topics(doc, minimum_probability=0)

        topic_probs = {}

        for topic, prob in doc_topics:
            topic_probs[topic] = prob
        document_topic_probs[i] = topic_probs


    return document_topic_probs

# Returns the the topic distribution for each label of the training dataset
def average_label_topic_distribution(document_topic_probs, l0rows, l1rows, l2rows):
    classifier_topic_avg_prob = {}
    for classifier in range(0, 3):
        classifier_topic_avg_prob[classifier] = {}
        label_rows = locals()[f"l{classifier}rows"]
        for topic in range(0, 10):
            prob = 0
            for row in label_rows:
                prob += document_topic_probs[row][topic]
            classifier_topic_avg_prob[classifier][topic] = prob/len(locals()[f"l{classifier}rows"])
    return classifier_topic_avg_prob

def determine_top_topics_for_classifiers(classifier_topic_avg_prob):
    print("Top 3 Topics for Each Classifier: ")
    for outer_key, outer_values in classifier_topic_avg_prob.items():
        sorted_inner = sorted(outer_values.items(), key=lambda x: x[1], reverse=True)
        print(outer_key)
        print(sorted_inner[:3])
        

if __name__ == "__main__":
    print("Initializing: ")
    documents = dataset['train']['sentence']
    training_labels = dataset['train']['label']

    args = from_args()
    vector_type = args.vector
    idf = args.idf
    stem = args.stem
    matrix, id2token = compute_doc_vectors_(documents, args.vector, args.idf, args.stem)
    # sorted_llr = train_nb(training_labels, matrix, id2token)

    # Grab Saved JSON of sorted_llr dictionary so my model doesn't have to train every time.
    with open('sorted_llr_dictionary.json', 'r') as file:
        sorted_llr = json.load(file)
    
    print_top_10_llr(sorted_llr)
    document_topic_probs = model_topics(matrix, id2token)

    rows_with_label_0 = []
    rows_with_label_1 = []
    rows_with_label_2 = []

    print("Filtering Rows by Classifier: ")
    for i, label in tqdm(enumerate(dataset['train']['label'])):
        # Check if the label is equal to 0
        if label == 0:
            # If so, append the row number to the list
            rows_with_label_0.append(i)
        elif label == 1:
            # If so, append the row number to the list
            rows_with_label_1.append(i)
        else:
            rows_with_label_2.append(i)

    classifier_topic_avg_prob = average_label_topic_distribution(document_topic_probs, rows_with_label_0, rows_with_label_1, rows_with_label_2)
    determine_top_topics_for_classifiers(classifier_topic_avg_prob)
