from sys import exit
from pprint import pprint

from nlp import NLP
from uses_multi import USES_MULTI
from alter_uses import ALTER_USES


import time
import json

input_file = '5_class'
print('Input File: ', input_file)

modes = ['default','domain','uses','alter_uses']

for mode in modes:
    print("\n")
    
    ini = time.time()

    if (input_file == '5_class'):
        p_severities = ['1','2','3','4','5']
    elif (input_file == 'teste'):
        p_severities = ['0']     
    elif (input_file == '2_class'):
        p_severities = ['0','1']   

    dict_words = set()
    severities = {}
    for severity in p_severities:
        severities[severity] = []

        if (input_file == '5_class'):
            file = '../dataset/Multiclass/severity'+severity+'.txt'
        elif (input_file == 'teste'):
            file = '../dataset/teste/severity'+severity+'.txt'
        elif (input_file == '2_class'):
            file = '../dataset/2class/severity'+severity+'.txt'
            
        file = open(file, 'r', encoding="utf8")
        
        for line in file:
            if (line and line != '\n'):
                issue = json.loads(line);
                issue["text"] = '';
                issue["text"] = issue["description"].strip() + ' ' + issue["steps_reproduce"].strip() + ' ' + issue["expected_result"].strip() + ' ' + issue["real_result"].strip()            
                
                
                if (mode == 'domain'):
                    if (len(dict_words) == 0):
                        print('Mode: Domain')
                    severity_words = issue["severity_words"].strip().lower()
                    issue["text"] = severity_words
                    severity_words = severity_words.split()
                    dict_words |= set(severity_words)

                severities[severity].append(issue)
                

    severities = NLP.array_merge(severities.values())

    features = [t['text'] for t in severities]

    labels = [t['severity'] for t in severities]

    features = NLP.tokenizer(features)

    if (mode != 'default'):
        if (mode == 'uses'):
            print('Mode: USES')
            dict_words = USES_MULTI.get_dict(features, labels)
        elif (mode == 'alter_uses'):
            print('Mode: ALTER USES')
            dict_words = ALTER_USES.get_dict(features, labels)
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
        #print(classifier)
        metrics = results[classifier];
        #for metric in metrics:
        #    print(metric,': ', results[classifier][metric]['avg'])
        #print("\n")

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

