from sys import exit
from pprint import pprint
import numpy as np
import random
from sklearn.metrics import f1_score

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from nlp import NLP

class USES:
    words = {}
    num_label = {}

    def __init__(self, features, labels):
        size = len(features)
        for i in range(0,size):
            item = features[i]
            label = labels[i]
            
            if label not in self.num_label:
                    self.num_label[label] = 0

            self.num_label[label] = self.num_label[label] + 1
                
            sizeItem = len(item)
            for j in range(0,sizeItem):
                feature = item[j]

                if feature not in self.words:
                    self.words[feature] = {}

                if label not in self.words[feature]:
                    self.words[feature][label] = 0

                self.words[feature][label] = self.words[feature][label] + 1
                
    def affinity_score(self, label, word):
        label = str(label)
        if label in self.words[word]:
            defect_word_label = self.words[word][label]
        else:
            defect_word_label = 0
            
        defect_label = self.num_label[label]
        defect_word = sum(self.words[word].values())
        
        affinity = defect_word_label / (defect_label + defect_word -defect_word_label)

        return affinity;

    def feature_score(self):
        score = {}
        for word in self.words:
            score[word] = self.affinity_score(1, word) -  self.affinity_score(0, word)

        score = sorted(score.items(), key=lambda x: x[1])
        
        return score

    def random_items(self, items, number):
        new_items = {}
        
        keys = random.sample(list(items), number)
        
        for key in keys:
            new_items[key] = items[key]

        return new_items
            
        
    def feature_selection(self, features, labels, percentage_words, percentage_build, iterations):
        number_words = len(self.words)
        number_selected_words = int(number_words * percentage_words)
        features_score = self.feature_score()
        
        positive_candidates = features_score[-number_selected_words:]    
        negative_candidates = features_score[:number_selected_words]

        candidates = {}
        candidates.update(positive_candidates)
        candidates.update(negative_candidates)

        number_features = len(features)
        number_build_features = int(number_features * percentage_build)

        pprint(len(features))
        pprint(len(labels))
        
        features_build = features[:number_build_features]
        labels_build = labels[:number_build_features]

        pprint(len(features_build))
        pprint(len(labels_build))
        
        features_validate = features[number_build_features:]        
        labels_validate = labels[number_build_features:]

        pprint(len(features_validate))
        pprint(len(labels_validate))
        
        best_FMeasure = 0
        actual_iteration = 0
        selected_features = {}

        while actual_iteration < iterations and best_FMeasure > 0:
            actual_candidates = self.random_items(candidates, number_selected_words)
            
            filtered_features_build = self.filter_features(features_build, actual_candidates)
            
            filtered_features_validate = self.filter_features(features_validate, actual_candidates)
           
            try:
                FMeasure = self.classifier(
                    filtered_features_build,
                    labels_build,
                    filtered_features_validate,
                    labels_validate
                )

                if (FMeasure > best_FMeasure):
                    best_FMeasure = FMeasure
                    selected_features = actual_candidates
            except:
                FMeasure = 0            

            actual_iteration = actual_iteration + 1

        return self.filter_features(features, selected_features)
            
    def classifier(self, features_build, labels_build, features_validate, labels_validate):
        features_build = NLP.text_to_numeric(features_build)
        features_validate = NLP.text_to_numeric(features_validate)
        
        svc = MultinomialNB()
        
        svc.fit(features_build, labels_build)
                
        labels_predict = svc.predict(features_validate)

        
        return f1_score(labels_validate, labels_predict, average="macro")
        

    def filter_features(self, features, selected_features):
        size = len(features)
        for i in range(0,size):
            item = features[i]
            item = [t for t in item if t in selected_features]
            features[i] = item

        return features
    
        
        
        




    
    
    

        
                
            
        
        

  
