from sys import exit
from pprint import pprint

from nlp import NLP
from uses_multi import USES_MULTI
from alter_uses import ALTER_USES
from files import FILES
from domain import DOMAIN


import time

#'5_class', 'teste', '2_class'

input_file = '5_class'
print('Input File: ', input_file)

modes = ['default','domain','uses','alter_uses']

for mode in modes:
    print("\n")
    
    ini = time.time()

    issues = FILES.read(input_file)
    
    features = [t['text'] for t in issues]

    labels = [t['severity'] for t in issues]

    severity_words = [t['severity_words'] for t in issues]

    features = NLP.tokenizer(features)

    if (mode != 'default'):
        if (mode == 'uses'):
            print('Mode: Uses')
            dict_words = USES_MULTI.get_dict(features, labels)
        elif (mode == 'alter_uses'):
            print('Mode: Alter Uses')
            dict_words = ALTER_USES.get_dict(features, labels)
        elif (mode == 'domain'):
            print('Mode: Domain')
            dict_words = DOMAIN.get_dict(severity_words)
            
        features = NLP.filter_features(features, dict_words)
        
    else :
        print('Mode: Default')
        

    features = NLP.remove_numbers(features)
    features = NLP.remove_small_words(features)
    features = NLP.remove_stop_words(features, 'portuguese')
    features = NLP.lemmatizer(features, 'portuguese')
    features = NLP.remove_punctuation(features)
    
    features = NLP.text_to_numeric(features)


    #features = NLP.dim_reduction(features)

    results = NLP.test(features, labels)

    better = {}
    better['Accuracy'] = 0
    better['Precision'] = 0
    better['Recall'] = 0
    better['F1'] = 0
    better_name = ''

    better = {}
    better['Accuracy'] = 0
    better['Precision'] = 0
    better['Recall'] = 0
    better['F1'] = 0
    better_name = ''

    for classifier in results:
        metrics = results[classifier];

        if (metrics['F1']['avg'] > better['F1']):
            better_name = classifier
            better['Accuracy'] = metrics['Accuracy']['avg']
            better['Precision'] = metrics['Precision']['avg']
            better['Recall'] = metrics['Recall']['avg']
            better['F1'] = metrics['F1']['avg']


    print(better_name)   
    print(better)       

    fim = time.time()
    print ("Tempo decorrido: ", fim-ini)
