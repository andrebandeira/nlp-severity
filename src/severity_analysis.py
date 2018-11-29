from sys import exit
from pprint import pprint
import json

from nlp import nlp

p_severities = ['1','2','3','4','5']
#p_severities = ['0','1']

severities = {}
for severity in p_severities:
    severities[severity] = []
    file = '../dataset/severity'+severity+'.txt'
    #file = '../dataset/2_classes/severity'+severity+'.txt'
    file = open(file, 'r', encoding="utf8")
    for line in file:
        issue = json.loads(line);
        issue["text"] = issue["description"].strip() + ' ' + issue["steps_reproduce"].strip() + ' ' + issue["expected_result"].strip() + ' ' + issue["real_result"].strip()            
        severities[severity].append(issue)

n = nlp()

issues = n.array_merge(severities.values())

issues = n.get_tokens(issues, "text")

data = n.get_data(issues, "tokens", "severity")

results = n.test_data(data)
    
for classifier in results:
    print(classifier)
    metrics = results[classifier];
    for metric in metrics:
        print(metric,': ', results[classifier][metric]['avg'])
    print("\n")
