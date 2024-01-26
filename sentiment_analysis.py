from datasets import load_dataset
dataset = load_dataset("FinanceInc/auditor_sentiment")
from tqdm import tqdm
import collections
import math
import nltk
nltk.download('punkt')
import string

def get_char_ngrams(words, N=3, start_token="<s>", end_token="</s>"):
    counter = collections.defaultdict(int)
    for word in words:
        chars = [start_token] + list(word) + [end_token]
        for i in range(len(chars)-(N-1)):
            ngram = chars[i:i+N]
            counter[''.join(ngram)] += 1
    return counter

def tokenize(document, char_ngrams=3):
    doc_tokens = []
    # use nltk sentence tokenization
    sentences = nltk.sent_tokenize(document)
    for sentence in sentences:
        # use nltk word tokenization
        sent_tokens = nltk.word_tokenize(sentence)
        # remove punctuation
        sent_tokens = [word.translate(str.maketrans('', '', string.punctuation)) for word in sent_tokens]
        # lowercase and remove empty strings (all punctuation will become empty after previous line)
        sent_tokens = [word.lower() for word in sent_tokens if word]
        # either use char ngrams or full words
        if char_ngrams:
            doc_tokens += get_char_ngrams(sent_tokens, char_ngrams)
        else:
            doc_tokens += sent_tokens
    return doc_tokens

def train_nb(training_data):
    total_words_with_label = collections.defaultdict(int)
    word_with_label_counts = {}
    vocab = set()
    labels = set()
    total_docs = len(training_data['sentence'])

    # Go through each training corpus provided and the total word counts for labels and the counts of specific words for labels
    for d in tqdm(range(total_docs)):
        document = training_data['sentence'][d]
        label = training_data['label'][d]
        # Collect unique sentiments
        labels.add(label)
        if label not in word_with_label_counts:
            word_with_label_counts[label] = collections.defaultdict(int)
        words = tokenize(document)
        for word in words:
            vocab.add(word)
            word_with_label_counts[label][word] += 1
            total_words_with_label[label] += 1

    print(total_words_with_label, word_with_label_counts)


    # Calculate likelihoods
    likelihood_comp = {}
    llr = {}
    for word in vocab:
        likelihood_comp[word] = {} 
        llr[word] = {}
        numerator = 0
        denominator = 0 
        max_val = float('-inf')
        most_likely_label = None
        # Calculate the likelihood for each label based on a given word    
        for label in labels:
            numerator = word_with_label_counts[label][word]
            denominator = total_words_with_label[label]
            likelihood_comp[word][label] = (numerator + 1)/ (denominator + len(vocab))
        
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
        log_unlikelihood = math.log((numerator + 1) / (denominator + len(vocab)))
        log_likelihood = math.log(max_val) 
        
        # Creates dictionary with each word, their most likely label, and the llr
        llr[word][most_likely_label] = log_likelihood - log_unlikelihood

    return llr

            

        
    
