
import random
import nltk
from backoff_model import preprocess_corpus
from nltk.tokenize import word_tokenize
from collections import defaultdict, Counter

def read_corpus(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        corpus = file.read()
    return str(corpus)  
    

# Step 4: Tokenize corpus and limit vocabulary size

def tokenize_corpus(corpus, vocab_size):
    tokens = word_tokenize(corpus)
    token_counts = Counter(tokens)
    vocab = {token for token, count in token_counts.most_common(vocab_size)}
    tokenized_corpus = [token if token in vocab else '<UNK>' for token in tokens]
    return tokenized_corpus

def split_corpus(corpus):
    sentences = nltk.sent_tokenize(corpus)
    random.shuffle(sentences)

    total_sentences = len(sentences)
    train_end = int(0.7 * total_sentences)
    val_end = int(0.1 * total_sentences) + train_end
    train_set = sentences[:train_end]
    val_set = sentences[train_end:val_end]
    test_set = sentences[val_end:]
    
    return train_set, val_set, test_set

def split_corpus_to_token(corpus):
    # Split corpus
    train_set, val_set, _ = split_corpus(corpus)

    # Preprocess corpus
    processed_train_set = preprocess_corpus(' '.join(train_set))
    processed_val_set = preprocess_corpus(' '.join(val_set))
   

    # Tokenize corpus
    vocab_size = 5000
    train_tokens = tokenize_corpus(processed_train_set, vocab_size)
    val_tokens = tokenize_corpus(processed_val_set, vocab_size)

    return train_tokens,val_tokens
