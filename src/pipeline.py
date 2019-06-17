from sys import exit
from pprint import pprint

from nlp import NLP
from uses_multi import USES_MULTI
from alter_uses import ALTER_USES
from files import FILES
from domain import DOMAIN
from joblib import dump, load
from sklearn.svm import LinearSVC

import time
import warnings
warnings.filterwarnings("ignore")

#'5_class', 'teste', '2_class'

input_file = '2_class'
print('Input File: ', input_file)

modes = ['default','uses','alter_uses']

for mode in modes:
    print("\n")
    
    ini = time.time()

    issues = FILES.read(input_file)

    features = [t['text'] for t in issues]

    labels = [t['severity'] for t in issues]

    features = NLP.tokenizer(features)
    
    features = NLP.remove_numbers(features)
    features = NLP.remove_small_words(features)
    features = NLP.remove_stop_words(features, 'portuguese')
    features = NLP.lemmatizer(features, 'portuguese')
    features = NLP.remove_punctuation(features)

    if ("domain" in mode):
        severity_words = [t['severity_words'] for t in issues]

        severity_words = NLP.tokenizer(severity_words)
    
        severity_words = NLP.remove_numbers(severity_words)
        severity_words = NLP.remove_small_words(severity_words)
        severity_words = NLP.remove_stop_words(severity_words, 'portuguese')
        severity_words = NLP.lemmatizer(severity_words, 'portuguese')
        severity_words = NLP.remove_punctuation(severity_words)

    dict_words = set()
    
    if (mode == 'uses'):
        print('Mode: Uses')
        dict_words = USES_MULTI.get_dict(features, labels)
    elif (mode == 'alter_uses'):
        print('Mode: Alter Uses')
        dict_words = ALTER_USES.get_dict(features, labels)
    elif (mode == 'domain'):
        print('Mode: Domain')
        dict_words = DOMAIN.get_dict(severity_words)
    else :
        print('Mode: Default')        
            
            
    features = NLP.filter_features(features, dict_words)
        
    if (len(dict_words)):
        dict_words = [dict_words];
    else:
        dict_words = []

    features = NLP.text_to_numeric(features,[], 'tf')        


    #features = NLP.dim_reduction(features)

    results = NLP.test(features, labels)

    for classifier in results:
        print('Classifier: ', classifier)
        metrics = results[classifier];
        pprint(metrics['F1'])


    fim = time.time()
    print ("Tempo decorrido: ", fim-ini)

    model = LinearSVC(max_iter=10000, random_state = NLP.seed)
    model.fit(features, labels)
    dump(model, 'files/model.joblib')
    dump(dict_words, 'files/dict.joblib')
    
