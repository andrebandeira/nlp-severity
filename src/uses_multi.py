from sys import exit
from pprint import pprint
import numpy as np
import random
from sklearn.metrics import f1_score

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from nlp import NLP

class USES_MULTI:
        
    def feature_selection(features, labels, percentage_words, percentage_build, iterations):
        real_features = features.copy()
        
        words_label = USES_MULTI.words_label(features, labels)

        number_words = len(words_label)
        number_selected_words = int(round(number_words * percentage_words))
        features_score = USES_MULTI.feature_score(words_label, labels)
        
        candidates = {}
        for label in features_score:
            negative_candidates = features_score[label][-number_selected_words:]            
            positive_candidates = features_score[label][:number_selected_words]
            candidates.update(positive_candidates)
            candidates.update(negative_candidates)

        number_features = len(features)
        number_build_features = int(number_features * percentage_build)
        
        results = {}
        actual_iteration = 0
        
        while (actual_iteration < iterations) or (len(results) == 0):
            if (actual_iteration % iterations == 0):
                suffle_features = USES_MULTI.suffle_features(features, labels)
                features = suffle_features['features']
                labels = suffle_features['labels']
            
            actual_candidates = USES_MULTI.random_items(candidates, number_selected_words)

            filtered_features = USES_MULTI.filter_features(features, actual_candidates)
        
            try:
                #pprint(actual_iteration)
                FMeasure = USES_MULTI.classifier(
                    filtered_features,
                    labels,
                    number_build_features
                )
                #pprint(FMeasure)
            except:
                #except Exception as e:
                #pprint(e)
                FMeasure = 0            

            results.update({
                FMeasure: actual_candidates
            })
            
            actual_iteration = actual_iteration + 1

        results = sorted(results.items(), key=lambda x: x[0], reverse=True)

        best_FM = results[0][0]
        selected_features = results[0][1]
        
        filtered_features = USES_MULTI.filter_features(real_features, selected_features)
        pprint('BEST')
        pprint(best_FM)
        
        return filtered_features

    def valid_candidates(features, labels, features_score, actual_candidates):
        candidates = list(actual_candidates.keys())
        size = len(features)
        for i in range(0,size):
            item = features[i]
            valid = False
            for feature in item:
                if feature in candidates:
                    valid = True
                    break
                
            if (valid == False):
                scores = features_score[labels[i]]
                for score in scores:
                    if score[0] in item:
                        candidates.append(score[0])
                        actual_candidates.update({
                            score[0]: score[1]
                        })
                        break
                    
        return actual_candidates
        
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
        distinct_labels = list(set(labels))
        score_label = 0;
        score_others_labels = 0

        score = {}
        
        for label in distinct_labels:
            others_labels = [t for t in distinct_labels if t != label]
            for word in words_label:
                score_label = USES_MULTI.affinity_score(label, word, words_label, labels)
                score_others_labels = 0
                for other in others_labels:
                    score_others_labels += USES_MULTI.affinity_score(other, word, words_label, labels)

                if  label not in score:
                    score[label] = {}
                    
                score[label][word] = score_label - score_others_labels

        
        for label in score:
            score[label] = sorted(score[label].items(), key=lambda x: x[1], reverse=True)
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
        items_copy = items.copy()
        
        new_items = {}
        
        keys = random.sample(list(items_copy), number)
        
        for key in keys:
            new_items[key] = items_copy[key]

        return new_items
                   
    def classifier(features, labels, number_build_features):
        features_copy = features.copy()
        labels_copy = labels.copy()
        
        features_copy = NLP.text_to_numeric(features_copy)
        
        features_build = features_copy[:number_build_features]
        labels_build = labels_copy[:number_build_features]
        
        features_validate = features_copy[number_build_features:]        
        labels_validate = labels_copy[number_build_features:]
             
        svc = LinearSVC(max_iter=10000)
        
        svc.fit(features_build, labels_build)
                
        labels_predict = svc.predict(features_validate)
        
        return f1_score(labels_validate, labels_predict, average="macro")
        
    def filter_features(features, selected_features):
        features_copy = features.copy()
        size = len(features_copy)
        for i in range(0,size):
            item = features_copy[i]
            item = [t for t in item if t in selected_features]
            features_copy[i] = item

        return features_copy

    def suffle_features(features, labels):
        features_copy = features.copy()
        labels_copy = labels.copy()
        
        positions = np.arange(0, len(features_copy)).tolist()
        np.random.shuffle(positions)
        newFeatures = []
        newLabels = []
        
        i = 0        
        for position in positions:
            newFeatures.insert(i, features_copy[position])
            newLabels.insert(i, labels_copy[position])
            i = i + 1
        
        return {
           'features': newFeatures,
           'labels': newLabels,
        }
        





    
    
    

        
                
            
        
        

  
