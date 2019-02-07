from sys import exit
from pprint import pprint

from nlp import NLP

import json

class FILES:
    def read(input_file):
        if (input_file == '5_class'):
            p_severities = ['1','2','3','4','5']
        elif (input_file == 'teste'):
            p_severities = ['0']     
        elif (input_file == '2_class'):
            p_severities = ['0','1']   

        
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
                    line = json.loads(line);
                    issue = {}
                    
                    issue["text"] = '';
                    issue["severity_words"] = '';

                    if ("description" in line):
                        issue["text"] += line["description"].strip().lower() + ' '

                    if ("steps_reproduce" in line):
                        issue["text"] += line["steps_reproduce"].strip().lower() + ' '

                    if ("expected_result" in line):
                        issue["text"] += line["expected_result"].strip().lower() + ' '

                    if ("real_result" in line):
                        issue["text"] += line["real_result"].strip().lower() + ' '

                    if ("severity_words" in line):
                        issue["severity_words"] = line["severity_words"].strip().lower()

                    issue["severity"] = line["severity"]

                    issue["text"] = issue["text"].strip()
                    issue["severity_words"] = issue["severity_words"].strip()
                    
                    
                    severities[severity].append(issue)
                    

        return NLP.array_merge(severities.values())
