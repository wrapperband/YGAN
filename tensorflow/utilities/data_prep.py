"""
@author: Sean A. Cantrell
"""
import numpy as np
from nltk.tokenize import TweetTokenizer
from collections import Counter
from tqdm import tqdm

TWTK = TweetTokenizer(strip_handles=True, reduce_len=True)

def convert_data(data, freq_threshold):
    """
    Prepares the data for training.
    Convert the data to indices
    """
    # Build the vocab
    print('Building the vocab')
    vocab, word_id, id_word = build_vocab(data, freq_threshold)

    # Map the words to indices
    print("Mapping words in input strings to indices")
    x = [[word_id[token] for token in to_tokens(row) if
          token in word_id.keys()] for row in data]
    
    bad = set(np.where([sum(elem) == 0 for elem in x])[0]) ## Set of bad inputs
    x = [elem for i, elem in enumerate(x) if i not in bad]

    print("Finalizing prepped data")
    # Split the data
    data_size = len(x)
    train_size = int(np.floor(0.8 * data_size)) # Train on 90% of data
    training, testing_raw = x[:train_size], x[train_size:]

    # Throw out elements of testing set that contain words not in train set
    train_inds = set([elem for row in training for elem in row])
    testing = []
    for row in testing_raw:
        if set(row) <= train_inds:
            testing.append(row)
        else:
            training.append(row)
    print("Size of training set:", len(training))
    print("Size of testing set:", len(testing))
    return training, testing, vocab, word_id, id_word

def build_vocab(corpora, freq_threshold):
    """
    Determine the set of unique words and their occurrence frequency.
    """
    vocab_count = Counter()
    for doc in tqdm(corpora):
        vocab_count.update(set(to_tokens(doc)))
    # Only keep words above a sufficient frequency
    vocab = {word : (i,freq) for i, (word,freq) in
                  enumerate(vocab_count.items()) if freq>=freq_threshold}
    # Number of words in corpus
    token_count = np.sum([elem[1] for elem in vocab.values()])
    # Number of unique words; +1 to include padding
    vocab_size = len(vocab) + 1
    print("\nTotal vocab:", vocab_size)
    print("Number of words in corpus:", token_count)
    # Create a dictionary that indexes the words and admits padding
    word_id = {word:i+1 for i, word in enumerate(sorted(vocab.keys()))}
    word_id['#EoS']=0
    # Create a reverse dictionary that converts indices to words
    id_word = {i:word for word,i in word_id.items()}
    # Create a dictionary that returns the word count given an id
    return vocab, word_id, id_word

def to_tokens(text):
    """
    Tokenize sentences using NLTK Tweet tokenizer.
    I would advise swapping out for spaCy for more general corpora.
    """
    tokens = [str(token).lower() for token in TWTK.tokenize(text) if
              len(token)>0]
    return tokens
