from sys import exit
from pprint import pprint
import numpy as np
import random
from sklearn.metrics import f1_score

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from nlp import NLP

class USES:
        
    def feature_selection(features, labels, percentage_words, percentage_build, iterations):        
        words_label = USES.words_label(features, labels)

        number_words = len(words_label)
        number_selected_words = int(round(number_words * percentage_words))

        features_score = USES.feature_score(words_label, labels)
        
        positive_candidates = features_score[-number_selected_words:]    
        negative_candidates = features_score[:number_selected_words]

        candidates = {}
        candidates.update(positive_candidates)
        candidates.update(negative_candidates)
        
        number_features = len(features)
        number_build_features = int(number_features * percentage_build)

        features_build = features[:number_build_features]
        labels_build = labels[:number_build_features]
        
        features_validate = features[number_build_features:]        
        labels_validate = labels[number_build_features:]
        
        best_FMeasure = 0
        actual_iteration = 0
        selected_features = {}

        while (actual_iteration < iterations) or (best_FMeasure == 0):               
            
            actual_candidates = USES.random_items(candidates, number_selected_words)
            
            filtered_features_build = USES.filter_features(features_build, actual_candidates)
            
            filtered_features_validate = USES.filter_features(features_validate, actual_candidates)
            
            try:
                FMeasure = USES.classifier(
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

        return USES.filter_features(features, selected_features)

    def features_label(features, labels):
        features_label = []
        size = len(features)
        for i in range(0,size):
            features_label.append({
                'features': features[i],
                'labels': labels[i]
            })

        return features_label
    
    def words_label(features, labels):
        words_label = {}
        
        size = len(features)
        for i in range(0,size):
            item = features[i]
            label = labels[i]
                            
            sizeItem = len(item)
            for j in range(0,sizeItem):
                feature = item[j]

                if feature not in words_label:
                    words_label[feature] = {}

                if label not in words_label[feature]:
                    words_label[feature][label] = 0

                words_label[feature][label] = words_label[feature][label] + 1

        return words_label
        
    def feature_score(words_label, labels):
        score = {}
        for word in words_label:
            score[word] = USES.affinity_score(1, word, words_label, labels) -  USES.affinity_score(0, word, words_label, labels)

        score = sorted(score.items(), key=lambda x: x[1])
        
        return score            
                
    def affinity_score(label, word, words_label, labels):
        label = str(label)
        if label in words_label[word]:
            defect_word_label = words_label[word][label]
        else:
            defect_word_label = 0
            
        defect_label =  labels.count(label)
        defect_word  = sum(words_label[word].values())
               
        affinity = defect_word_label / (defect_label + defect_word -defect_word_label)

        return affinity;

    def random_items(items, number):
        new_items = {}
        
        keys = random.sample(list(items), number)
        
        for key in keys:
            new_items[key] = items[key]

        return new_items
                   
    def classifier(features_build, labels_build, features_validate, labels_validate):
        features_build = NLP.text_to_numeric(features_build)
        features_validate = NLP.text_to_numeric(features_validate)
        
        svc = MultinomialNB()
        
        svc.fit(features_build, labels_build)
                
        labels_predict = svc.predict(features_validate)
        
        return f1_score(labels_validate, labels_predict, average="macro")
        
    def filter_features(features, selected_features):
        size = len(features)
        for i in range(0,size):
            item = features[i]
            item = [t for t in item if t in selected_features]
            features[i] = item

        return features

    def shuflle_features(features, labels):
        features_label = USES.features_label(features, labels)
        np.random.shuffle(features_label)

        features = []
        labels = []
        
        size = len(features_label)
        for i in range(0,size):
            features.append(
                features_label[i]['features']
            )

            labels.append(
                features_label[i]['labels']
            )

        return {
            'features': features,
            'labels': labels
        }
    
        
        
        




    
    
    

        
                
            
        
        

  
