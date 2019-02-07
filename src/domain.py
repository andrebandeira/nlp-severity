from sys import exit
from pprint import pprint

class DOMAIN:
        
    def get_dict(severity_words):
        dict_words = set()
        
        size = len(severity_words)
        
        for i in range(0,size):            
            words = severity_words[i].split()
            for word in words:
                dict_words.add(word)
        
        return dict_words
