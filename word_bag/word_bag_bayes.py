# Use word count to predict labels
import json
import pandas as pd
import numpy as np
import nltk
import string
import random
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from typing import Dict, List, Set

# read data from json file
with open('train_data/labeled/train.json', 'r') as fp:
    train_data = [json.loads(line) for line in fp.readlines()]

# split data into train and test
random.shuffle(train_data)
train_data = train_data[:int(len(train_data) * 0.8)]
test_data = train_data[int(len(train_data) * 0.8):]

# Bayes classification on word bag model
# Assume that for each word, the probability of it appearing in a document is independent of other words, 
# and the probability of it appearing in a document is proportional to the frequency of it in the document

# get all labels
label_columns = ['privilege-required', 'attack-vector', 'impact']
labels = {col: set() for col in label_columns}
for data in train_data:
    for col in label_columns:
        labels[col].add(data[col])
labels = {col: list(labels[col]) for col in label_columns}

word_count = {}
word_count_in_label = {col: {label: {} for label in labels[col]} for col in label_columns}
phrase_count = {}
phrase_count_in_label = {col: {label: {} for label in labels[col]} for col in label_columns}

label_count = {col: {label: 0 for label in labels[col]} for col in label_columns}

# get all words
def update_count(count: Dict[str, int], tokens: Set[str]):
    for word in tokens:
        count[word] = count.get(word, 0) + 1

def extract_words(description: str) -> Set[str]:
    tokens = word_tokenize(description)
    # filter out punctuation
    tokens = [word for word in tokens if word not in string.punctuation]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # filter out numbers
    tokens = [word for word in tokens if not word.isdigit()]
    # filter out words that letters are less than half of the length
    tokens = [word for word in tokens if sum(c.isalpha() for c in word) >= len(word) / 2]
    # lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return set(tokens)

none_phrase_grammar = r"""
NP: {<AT|DT|PP\$>?<RB>?<JJ>*<NN>(<IN><AT|DT|PP\$>?<RB>?<JJ>*<NN>)?}   # chunk determiner/possessive, adjectives and noun
    {<NNP>+}                                          # chunk sequences of proper nouns
"""

def extract_noun_phrases(description: str) -> Set[str]:
    pos_tagged = nltk.pos_tag(word_tokenize(description))

    cp = nltk.RegexpParser(none_phrase_grammar)
    tree = cp.parse(pos_tagged)
    noun_phrases = []
    for subtree in tree.subtrees():
        if subtree.label() == 'NP':
            t = ' '.join(word for word, tag in subtree.leaves())
            noun_phrases.append(t)
    # filter out phrases that are too short
    noun_phrases = [phrase for phrase in noun_phrases if len(phrase) >= 5]
    return set(noun_phrases)

verb_phrases_grammar = r"""
VP: {<VB.*><NP|PP|CLAUSE>+$}          # chunk verbs and their arguments
    {<VB.*><RB.?>*<TO><VB.*>} # chunk infinitival to
    {<RB.?>*<TO><VB.*>}          # chunk to verbs
"""

def extract_verb_phrases(description: str) -> Set[str]:
    pos_tagged = nltk.pos_tag(word_tokenize(description))
    cp = nltk.RegexpParser(verb_phrases_grammar)
    tree = cp.parse(pos_tagged)
    verb_phrases = []
    for subtree in tree.subtrees():
        if subtree.label() == 'VP':
            t = ' '.join(word for word, tag in subtree.leaves())
            verb_phrases.append(t)
    # filter out phrases that are too short
    verb_phrases = [phrase for phrase in verb_phrases if len(phrase) >= 5]
    return set(verb_phrases)

def extract_phrases(description: str) -> Set[str]:
    tokens = extract_noun_phrases(description)
    tokens.update(extract_verb_phrases(description))
    return tokens


# get word count, phras count and label count
for data in tqdm(train_data):
    phrases = extract_phrases(data['description'])
    words = extract_words(data['description'])
    update_count(word_count, words)
    update_count(phrase_count, phrases)
    for col in label_columns:
        label = data[col]
        label_count[col][label] += 1
        update_count(word_count_in_label[col][label], words)
        update_count(phrase_count_in_label[col][label], phrases)

# collect words that appear in at least 10 documents
all_words = [word for word, count in word_count.items() if count >= 10]
print('Number of known words: {}'.format(len(all_words)))

# collect phrases that appear in at least 5 documents
all_phrases = [phrase for phrase, count in phrase_count.items() if count >= 5]
print('Number of known phrases: {}'.format(len(all_phrases)))

# get word probability of appearing on each lable, arranged in order of all_words
word_prob = {col: {label: np.array([word_count_in_label[col][label].get(w, 0) for w in all_words]) / label_count[col][label] for label in labels[col]} for col in label_columns}
# get phrase probability of appearing on each lable, arranged in order of all_phrases
phrase_prob = {col: {label: np.array([phrase_count_in_label[col][label].get(p, 0) for p in all_phrases]) / label_count[col][label] for label in labels[col]} for col in label_columns}

# output word probability
with open('word_prob.json', 'w') as fp:
    word_prob_dict = {col: {label: {w: p for w, p in zip(all_words, word_prob[col][label])} for label in labels[col]} for col in label_columns}
    json.dump(word_prob_dict, fp, indent=4)
# output phrase probability
with open('phrase_prob.json', 'w') as fp:
    phrase_prob_dict = {col: {label: {p: v for p, v in zip(all_phrases, phrase_prob[col][label])} for label in labels[col]} for col in label_columns}
    json.dump(phrase_prob_dict, fp, indent=4)

# test on test data
eps = 2e-5  # epsilon avoids zero probability, that collaspe the log probability to -inf

print('test on test data')
correct = 0
total = 0

# predict label for each data by posterior probability
def predict(dataset):
    results = []
    for data in tqdm(dataset):
        phrases = extract_phrases(data['description'])
        words = extract_words(data['description'])
        # estimate P(label|description) = P(description|label) * P(label)
        # P(description|label) = P(word1|label) * P(word2|label) * ...
        # P(label) = count(label) / count(all)
        # we can ignore P(description) because it is the same for all labels
        # so we can just compare P(description|label) * P(label)
        # we use log to avoid underflow
        result = {'cve-number': data['cve-number'], 'description': data['description']}
        for col in label_columns:
            posteriors = []
            for label in labels[col]:
                posterior_word = np.log(label_count[col][label] / len(train_data)) + np.where([w in words for w in all_words], np.log(word_prob[col][label] + eps), np.log(1 - word_prob[col][label] + eps)).sum()
                posterior_phrase = np.log(label_count[col][label] / len(train_data)) + np.where([p in phrases for p in all_phrases], np.log(phrase_prob[col][label] + eps), np.log(1 - phrase_prob[col][label] + eps)).sum()
                posterior = posterior_word + posterior_phrase
                posteriors.append(posterior)
            result[col] = labels[col][np.argmax(posteriors)]
        results.append(result)
    return results

results = predict(test_data)
for col in label_columns:
    print(f'"{col}" acc:', sum(t[col] == r[col] for t, r in zip(test_data, results)) / len(test_data))
print("full acc:", sum(all(t[col] == r[col] for col in label_columns) for t, r in zip(test_data, results)) / len(test_data))

# predict test data
print('predict test data')
upper_case_map = {
    'nonprivileged': 'Nonprivileged',
    'non-remote': 'Non-remote',
    'privileged-gained(rce)': 'Privileged-Gained(RCE)',
    'dos': 'DoS'
}

with open('test_data/test_a.json', 'r') as fp:
    test_data = [json.loads(line) for line in fp.readlines()]

results = predict(test_data)
formated_results = []
for data in results:
    temp = {'CVE-Number': data['cve-number'], 'Description': data['description'], 'Privilege-Required': data['privilege-required'], 'Attack-Vector': data['attack-vector']}
    for level, impact_label in enumerate(data['impact'].split('_')):
        temp['Impact-level' + str(level + 1)] = impact_label

    formated_results.append({k: upper_case_map.get(v, v) for k, v in temp.items()})

# save as excel file
df = pd.DataFrame(formated_results)
df.to_excel('submissions/submit.xlsx', index=False, encoding='utf-8')
