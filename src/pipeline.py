from sys import exit
from pprint import pprint

from nlp import NLP
from uses import USES

import time
import json

ini = time.time()

mode = '2_classes'

if (mode == 'default'):
    p_severities = ['1','2','3','4','5']
elif (mode == 'teste'):
    p_severities = ['0']     
elif (mode == '2_classes'):
    p_severities = ['0','1']   

print(mode)

severities = {}
for severity in p_severities:
    severities[severity] = []

    if (mode == 'default'):
        file = '../dataset/severity'+severity+'.txt'
    elif (mode == 'teste'):
        file = '../dataset/teste/severity'+severity+'.txt'
    elif (mode == '2_classes'):
        file = '../dataset/2_classes/severity'+severity+'.txt'
        
    file = open(file, 'r', encoding="utf8")
    
    for line in file:
        if (line and line != '\n'):
            issue = json.loads(line);
            issue["text"] = '';
            issue["text"] = issue["description"].strip() + ' ' + issue["steps_reproduce"].strip() + ' ' + issue["expected_result"].strip() + ' ' + issue["real_result"].strip()            
            severities[severity].append(issue)


severities = NLP.array_merge(severities.values())

features = [t['text'] for t in severities]
labels = [t['severity'] for t in severities]

features = NLP.tokenizer(features)
#features = NLP.remove_numbers(features)
#features = NLP.remove_small_words(features)
#features = NLP.remove_stop_words(features, 'portuguese')
#features = NLP.lemmatizer(features, 'portuguese')
#features = NLP.remove_punctuation(features)

uses = USES(features, labels)
features = uses.feature_selection(features, labels, 0.2, 0.9, 1000)
#pprint(features)
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
    print(classifier)
    metrics = results[classifier];
    for metric in metrics:
        print(metric,': ', results[classifier][metric]['avg'])
    print("\n")

    if (metrics['Accuracy']['avg'] > better['Accuracy']):
        better_name = classifier
        better['Accuracy'] = metrics['Accuracy']['avg']
        better['Precision'] = metrics['Precision']['avg']
        better['Recall'] = metrics['Recall']['avg']
        better['F1'] = metrics['F1']['avg']

print(better_name)   
print(better)       

fim = time.time()
print ("Tempo decorrido: ", fim-ini)
