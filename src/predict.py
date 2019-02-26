from sys import exit
from pprint import pprint
from joblib import dump, load
from nlp import NLP
import sys

import time
import warnings
warnings.filterwarnings("ignore")

ini = time.time()

if (len(sys.argv) < 2):
    print('Defect not found. Check the path to the defect file.')
    exit()
    
file = sys.argv[1]

with open(file,'r') as i:
    lines = i.readlines()

defect = ' '.join(lines)

defect = defect.lower().strip().replace("\n", "")

features = NLP.tokenizer([defect])

features = NLP.remove_numbers(features)
features = NLP.remove_small_words(features)
features = NLP.remove_stop_words(features, 'portuguese')
features = NLP.lemmatizer(features, 'portuguese')
features = NLP.remove_punctuation(features)
    
dict_words = load('files/dict.joblib')
dict_words = dict_words[0]

features = NLP.filter_features(features, dict_words)

features = NLP.text_to_numeric(features, [dict_words])

model = load('files/model.joblib')

labels = model.predict(features)

if (labels[0] == '1'):
    print('High Severity')
else:
    print('Low Severity')
    

fim = time.time()
#print ("Tempo decorrido: ", fim-ini)
