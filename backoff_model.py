import nltk
import re
from collections import  Counter
from nltk.tokenize import word_tokenize
import numpy as np

def read_corpus(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        corpus = file.read()
    return str(corpus)  

def preprocess_corpus(corpus):
    # Remove special characters, keep only full stops and remove stop words
    processed_corpus = re.sub(r'[^\w\s.]', '', corpus)
    processed_corpus = re.sub(r'\b\w{1,3}\b', '', processed_corpus)
    return processed_corpus

# Step 5: Build LM1 language model using backoff method without add-k smoothing
def build_language_model(tokens):
    ngrams = nltk.ngrams(tokens, 3, pad_left=True, pad_right=True)
    bigrams = nltk.ngrams(tokens, 2, pad_left=True, pad_right=True)
    unigrams = tokens

    bigram_counts = Counter(bigrams)
    trigram_counts = Counter(ngrams)
    unigram_counts = Counter(unigrams)

    def calculate_probability(wi_minus_1, wi):
        lambda1 = 1/2  # Update lambdas for LM1
        lambda2 = 1/2

        # Calculate probabilities for 3-gram, 2-gram, and 1-gram without smoothing
        prob_3gram = trigram_counts[(wi_minus_1, wi)] / bigram_counts[wi_minus_1] if bigram_counts[wi_minus_1] > 0 else 0
        prob_2gram = bigram_counts[wi] / unigram_counts[wi_minus_1] if unigram_counts[wi_minus_1] > 0 else 0
        prob_1gram = unigram_counts[wi] / len(tokens)

        # Backoff: Use lower-order n-gram probabilities when higher-order n-grams are not available
        if prob_3gram == 0:
            prob_3gram = prob_2gram if prob_2gram != 0 else prob_1gram
        if prob_2gram == 0:
            prob_2gram = prob_1gram

        return lambda1 * prob_3gram + lambda2 * prob_2gram

    # Return a dictionary-like object representing the language model
    return calculate_probability

# Step 7: Create a text generator using the model
def generate_text_backoff(model, seed_text, length, vocab):
    generated_text = seed_text
    seed_tokens = word_tokenize(seed_text)
    for _ in range(length):  # We replaced 'i' with '_' to indicate that it's not used
        wi_minus_1 = seed_tokens[-1]  # Adjusted to get the last token
        next_token_probabilities = {}
        for token in vocab:
            next_token_probabilities[token] = model(wi_minus_1, token)  # Adjusted to pass only two arguments
        
        # Apply nucleus sampling
        sorted_tokens = sorted(next_token_probabilities.keys(), key=lambda x: next_token_probabilities[x], reverse=True)
        sorted_probs = [next_token_probabilities[token] for token in sorted_tokens]
        sorted_cum_probs = np.cumsum(sorted_probs)

        sorted_cum_probs /= sorted_cum_probs[-1]
        
        # Choose next token using nucleus sampling
        sampled_token_index = np.argmax(sorted_cum_probs > np.random.rand())
        next_token = sorted_tokens[sampled_token_index]

        generated_text += ' ' + next_token
        seed_tokens.append(next_token)
        if next_token == '.':
            break
    return generated_text
