from datasets import load_dataset
dataset = load_dataset("FinanceInc/auditor_sentiment")
from tqdm import tqdm
import collections
import math
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stopwords = set(stopwords.words('english'))
import string
import re
import numpy as np
import argparse
from scipy.sparse import csr_matrix

email_pat = re.compile(r"\S+@\S+\.\S+")
url_pat = re.compile("^https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)$")

def from_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('vector', default='count', nargs='?', choices=['count', 'binary', 'frequency'], help='Choose format for box of words. Default is count.')
    parser.add_argument('-i', '--idf', action='store_true', help="Uses Inverse Document Frequency")
    args = parser.parse_args()

    return args


def tokenize(document):
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
                            and not re.search('\d+',word)
                            and len(word) > 2])
        # either use char ngrams or full words
        doc_tokens += sent_tokens
    return doc_tokens


def compute_doc_vectors_(documents, tf_type='count', use_idf='False'):
    token2id = {}
    current_next_id = 0
    document_dicts = []

    for document in tqdm(documents):
        # Tokenize each document individually
        tokens = tokenize(document)
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
    for document_dict in document_dicts:
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



def train_nb(training_labels, matrix, id2token):
    num_mat_rows = matrix.shape[0]
    num_mat_col = matrix.shape[1]
    labels = [0, 1, 2]
    word_with_label_counts = {}
    total_words_with_label = collections.defaultdict(int)

    # Go through each training corpus provided and the total word counts for labels and the counts of specific words for labels
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
            numerator = word_with_label_counts[label][word]
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
            numerator += word_with_label_counts[label][word]
            denominator += total_words_with_label[label]
        log_unlikelihood = math.log((numerator + 1) / (denominator + num_mat_col))
        log_likelihood = math.log(max_val) 
        
        # Creates dictionary with each word, their most likely label, and the llr
        llr[word][most_likely_label] = log_likelihood - log_unlikelihood

    return llr

if __name__ == "__main__":
    documents = dataset['train']['sentence'][::200]
    training_labels = dataset['train']['label'][::200]
    args = from_args()
    vector_type = args.vector
    idf = args.idf
    matrix, id2token = compute_doc_vectors_(documents, args.vector, args.idf)
    llr = train_nb(training_labels, matrix, id2token)
    print(llr)