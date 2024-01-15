# project.py


import pandas as pd
import numpy as np
import os
import re
import requests
import time


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def get_book(url):
    whole_text = requests.get(url).text
    cropped_text = whole_text.split('***')[2]
    fixed_newline_text = re.sub(r"\r\n", "\n", cropped_text)
    return fixed_newline_text


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def tokenize(book_string):
    replaced_newlines = re.sub(r"\n\n+", "\x03 \x02", book_string)
    token_pattern = r"(\w+|\d+|[^\w\s])"
    tokens = re.findall(token_pattern, replaced_newlines)
    return tokens[1:-1]


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


class UniformLM(object):


    def __init__(self, tokens):

        self.mdl = self.train(tokens)
        
    def train(self, tokens):
        unique = set(tokens)
        n = len(unique)
        probability = 1/n
        return pd.Series([probability]*n, index=unique)
    
    def probability(self, words):
        probability = 1
        for word in words:
            if word in self.mdl.index:
                probability *= self.mdl[word]
            else:
                return 0
        return probability
        
    def sample(self, M):
        words = np.random.choice(self.mdl.index, M)
        return ' '.join(words)

        


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


class UnigramLM(object):
    
    def __init__(self, tokens):

        self.mdl = self.train(tokens)
    
    def train(self, tokens):
        return pd.Series(tokens).value_counts()/len(tokens)


    def probability(self, words):
        probability = 1
        for word in words:
            if word in self.mdl.index:
                probability *= self.mdl[word]
            else:
                return 0
        return probability
        
    def sample(self, M):
        words = np.random.choice(self.mdl.index, M, p=self.mdl)
        return ' '.join(words)


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


class NGramLM(object):
    
    def __init__(self, N, tokens):
        
        self.N = N

        ngrams = self.create_ngrams(tokens)
        self.ngrams = ngrams
        self.mdl = self.train(ngrams)

        if N < 2:
            raise Exception('N must be greater than 1')
        elif N == 2:
            self.prev_mdl = UnigramLM(tokens)
        else:
            self.prev_mdl = NGramLM(N-1, tokens)


    def create_ngrams(self, tokens):
        return list(zip(*[tokens[i:] for i in range(self.N)]))

        
    def train(self, ngrams):
        
        # Empty DataFrame for storing grams and probabilities
        train_df = pd.DataFrame()

        # Create N-1 grams
        n1grams = [gram[:-1] for gram in ngrams]

        # Put all grams in a DataFrame
        train_df = train_df.assign(ngram = ngrams, n1gram = n1grams)

        # N-Gram counts C(w_1, ..., w_n)
        # ngram_counts = [ngrams.count(gram) for gram in ngrams]
        ngram_df = train_df[['ngram']].groupby(train_df[['ngram']].columns.tolist(), as_index=False).size()
        train_df = train_df.merge(ngram_df).rename(columns={'size': 'ngram_count'})

        # (N-1)-Gram counts C(w_1, ..., w_(n-1))
        n1gram_df = train_df[['n1gram']].groupby(train_df[['n1gram']].columns.tolist(), as_index=False).size()
        train_df = train_df.merge(n1gram_df).rename(columns={'size': 'n1gram_count'})

        # Create the conditional probabilities
        train_df = train_df.assign(prob = train_df['ngram_count']/train_df['n1gram_count'])

        # Drop duplicate N grams
        train_df = train_df.drop_duplicates()

        return train_df.drop(columns = ['ngram_count', 'n1gram_count'])
    

    def probability(self, words):

        # Get N grams of input words
        input_ngrams = list(zip(*[words[i:] for i in range(self.N)]))

        # Conditional probabilities of the N grams
        probability = 1
        for gram in input_ngrams:
            if gram in self.ngrams:
                probability *= self.mdl[self.mdl['ngram'] == gram]['prob'].values[0]
            else:
                return 0

        # Probabilities of mini grams that make up first N gram
        first_ngram = input_ngrams[0]
        prev = self
        for i in np.arange(self.N-2):
            prev = prev.prev_mdl
            first_ngram = first_ngram[:-1]
            probability *= prev.mdl[prev.mdl['ngram'] == first_ngram]['prob'].values[0]

        # Probability of initial unigram
        prev = prev.prev_mdl
        probability *= prev.mdl[first_ngram[0]]

        return probability
    

    def sample(self, M):

        # Recursive helper function to generate sample tokens of length `length`
        def token_generator(current, length):

            # If not starter token, get ending N-1 gram
            if length > 0:
                current = current[0][1:]

            # Find next options of N grams that start with the same N-1 gram as current
            next_options = self.mdl[self.mdl['n1gram'] == current]

            # Stop sentence if no possible next N grams
            if next_options.shape[0] == 0:
                return

            # Pick an N gram based on conditional probabilities
            probs = next_options['prob']/np.sum(next_options['prob'])
            next_ngram =  np.random.choice(np.array(next_options['ngram']), p=probs)

            # Find corresponding N-1 gram (ending N-1 gram)
            next_n1gram = self.mdl[self.mdl['ngram'] == next_ngram]['ngram'].values
            if len(next_n1gram) > 1:
                next_n1gram = next_n1gram[1:]

            # Get last token of N gram
            next_token = next_ngram[-1]

            if length == M-2:
                return next_token
            else:
                return next_token + ' ' + token_generator(next_n1gram, length+1)


        # Add each N gram's first token to DataFrame
        df = self.mdl.assign(first_token = [gram[0] for gram in self.mdl['ngram']])

        # Get starter token
        starter_token_options = df[df['first_token'] == '\x02']
        starter_probs = starter_token_options['prob']/np.sum(starter_token_options['prob'])
        starter_token = np.random.choice(starter_token_options['n1gram'], p=starter_probs)

        # Run and return recursive helper function to generate sentence
        tokens = '\x02 ' + token_generator(starter_token, 0) + ' \x03'
        return tokens


