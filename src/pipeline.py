from sys import exit
from pprint import pprint

from nlp import Tokenizer, Remove_Numbers, Remove_Small_Words, Remove_Stop_Words, Lemmatizer, Remove_Punctuation, Text_To_Numeric, Utils

import time
import json

from sklearn.pipeline import Pipeline

ini = time.time()

mode = 'default'

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
        issue = json.loads(line);
        issue["text"] = '';
        issue["text"] = issue["description"].strip() + ' ' + issue["steps_reproduce"].strip() + ' ' + issue["expected_result"].strip() + ' ' + issue["real_result"].strip()            
        severities[severity].append(issue)


severities = Utils.array_merge(severities.values())

features = [t['text'] for t in severities]
labels = [t['severity'] for t in severities]

pre_processing = Pipeline([
    ('tokenizer',Tokenizer()),
    ('remove_number',Remove_Numbers()),
    ('remove_small_words',Remove_Small_Words()),
    ('remove_stop_words',Remove_Stop_Words('portuguese')),    
    ('lemmatizer',Lemmatizer('portuguese')),
    ('remove_punctuation',Remove_Punctuation()),
    ('text_to_numeric',Text_To_Numeric())
])

features = pre_processing.transform(features)
results = Utils.test(features, labels)

for classifier in results:
    print(classifier)
    metrics = results[classifier];
    for metric in metrics:
        print(metric,': ', results[classifier][metric]['avg'])
    print("\n")

fim = time.time()
print ("Tempo decorrido: ", fim-ini)
