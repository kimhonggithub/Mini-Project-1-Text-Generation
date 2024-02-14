
import nltk
from collections import Counter
from nltk.tokenize import word_tokenize
import numpy as np

# Step 6: Evaluate models on test set using perplexity
def evaluate_model(model, test_tokens):
    total_log_prob = 0
    N = len(test_tokens)
    for i in range(2, N-2):
        wi_minus_2, wi_minus_1, wi, _= test_tokens[i-2:i+2]
        prob = model(wi_minus_2, wi_minus_1, wi)
        total_log_prob += -1 * (prob * (N-4))
    perplexity = 2 ** (total_log_prob / N)
    return perplexity

    
def lamdas_and_k(train_tokens, val_tokens, vocab_size):
    # Perform experiments to find the best values for lambdas and k
    best_perplexity = float('inf')
    best_lambdas = None
    best_k = None
    
    # Define a range of values to experiment with for k and lambdas
    k_values = [0.1, 0.5, 1.0]
    lambda_combinations = [(0.1, 0.3, 0.6), (0.2, 0.4, 0.4), (0.3, 0.3, 0.4)]

    # Iterate over all combinations of lambda values and k values
    for lambdas in lambda_combinations:
        for k in k_values:
            # Build language model with current values of lambdas and k
            model = build_interpolation_language_model(train_tokens, k, vocab_size)
            
            # Evaluate the model on the validation set
            perplexity = evaluate_model(model, val_tokens)
            
            # Update best values if the perplexity is lower
            if perplexity < best_perplexity:
                best_perplexity = perplexity
                best_lambdas = lambdas
                best_k = k
    
    return best_lambdas, best_k



def build_interpolation_language_model(tokens, k, vocab_size):
    ngrams = nltk.ngrams(tokens, 4, pad_left=True, pad_right=True)
    bigrams = nltk.ngrams(tokens, 2, pad_left=True, pad_right=True)
    trigrams = nltk.ngrams(tokens, 3, pad_left=True, pad_right=True)
    unigrams = tokens

    bigram_counts = Counter(bigrams)
    trigram_counts = Counter(trigrams)
    ngram_counts = Counter(ngrams)
    unigram_counts = Counter(unigrams)
 
    def calculate_probability(wi_minus_2, wi_minus_1, wi):
        lambda1 = 1/4  # Including unigram, so update lambda accordingly
        lambda2 = 1/4
        lambda3 = 1/4
        lambda4 = 1/4  # New lambda for unigram
      
        prob_4gram = (ngram_counts[(wi_minus_2, wi_minus_1, wi, wi)] + k) / \
              (trigram_counts[(wi_minus_2, wi_minus_1, wi)] + k*float(vocab_size))

        prob_3gram = (trigram_counts[(wi_minus_1, wi, wi)] + k) / \
                      (bigram_counts[(wi_minus_1, wi)] + k * vocab_size)

        prob_2gram = (bigram_counts[(wi, wi)] + k) / \
                      (unigram_counts[wi] + k * vocab_size)

        prob_1gram = (unigram_counts[wi] + k) / \
                      (len(tokens) + k * vocab_size)

        return lambda1 * prob_4gram + lambda2 * prob_3gram + lambda3 * prob_2gram + lambda4 * prob_1gram

    return calculate_probability






def generate_text_iplt(model, seed_text, length, vocab):
    generated_text = seed_text
    seed_tokens = word_tokenize(seed_text)
    for _ in range(length):
        wi_minus_2, wi_minus_1 = seed_tokens[-2:]
        next_token_probabilities = {}
        for token in vocab:
            next_token_probabilities[token] = model(wi_minus_2, wi_minus_1, token)
        
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
