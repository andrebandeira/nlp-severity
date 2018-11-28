from sys import exit
from pprint import pprint
import json

from nlp import nlp

p_severities = ['1','2','3','4','5']

severities = {}
severities[0] = []
for severity in p_severities:
    severities[severity] = []
    file = 'dataset/severity'+severity+'.txt'
    file = open(file, 'r', encoding="utf8")
    for line in file:
        issue = json.loads(line);
        issue["text"] = issue["description"] + ' ' + issue["steps_reproduce"] + ' ' + issue["expected_result"] + ' ' + issue["real_result"]
        if (issue["severity"] == '5'):
            issue["severity"] = 1
            severity = 1
        else:
            issue["severity"] = 0
            severity = 0
            
        severities[severity].append(issue)

n = nlp()

issues = n.array_merge(severities.values())

issues = n.get_tokens(issues, "text")
data = n.get_data(issues, "tokens", "severity")

results = n.test_data(data, length = 100, number_times = 3)

for result in results:
    print("Classification rate for", result, ':', results[result]['avg_score'])
