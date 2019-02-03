from sys import exit
from pprint import pprint
import numpy as np
import random
from sklearn.metrics import f1_score

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from nlp import NLP

class ALTER_USES:
        
    def feature_selection(features, labels, max_words = 100):
        real_features = features.copy()
        real_labels = labels.copy()

        words_label = ALTER_USES.words_label(features, labels)
        number_words = len(words_label)

        if (number_words < max_words):
            max_words = number_words
            
        features_score = ALTER_USES.feature_score(words_label, labels)        
        
        results = {}
        number_words = 1
        
        while (number_words <= max_words) or (len(results) == 0):
            if (number_words % max_words == 0):
                suffle_features = ALTER_USES.suffle_features(features, labels)
                features = suffle_features['features']
                labels = suffle_features['labels']
            
            filtered_features = ALTER_USES.filter_features(features, labels, features_score, number_words)
        
            try:
                pprint(number_words)
                FMeasure = ALTER_USES.classifier(
                    filtered_features,
                    labels
                )
                pprint(FMeasure)
            except Exception as e:
                pprint(e)
                FMeasure = 0            

            results.update({
                FMeasure: number_words
            })
            
            number_words = number_words + 1

        results = sorted(results.items(), key=lambda x: x[0], reverse=True)

        best_FM = results[0][0]
        number_words = results[0][1]
        
        filtered_features = ALTER_USES.filter_features(real_features, real_labels, features_score, number_words)
        print('Best: ', best_FM)
        print('Number_Words: ', number_words)
        
        return filtered_features
        
    def words_label(features, labels):
        words_label = {}
        
        size = len(features)
        for i in range(0,size):
            item = features[i]
            label = labels[i]
            label = str(label)
                            
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
                score_label = ALTER_USES.affinity_score(label, word, words_label, labels)
                score_others_labels = 0
                for other in others_labels:
                    score_others_labels += ALTER_USES.affinity_score(other, word, words_label, labels)

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

        affinity = defect_word_label / (defect_label + defect_word - defect_word_label)
            
        return affinity;
                   
    def classifier(features, labels):
        features_copy = features.copy()
        labels_copy = labels.copy()
        
        features_copy = NLP.text_to_numeric(features_copy)

        results = NLP.test(features_copy, labels)
        betterFM = 0

        for classifier in results:
            metrics = results[classifier];
            if (metrics['F1']['avg'] > betterFM):
                betterFM= metrics['F1']['avg']

        return betterFM;
        
    def filter_features(features, labels, features_score, number_words):
        features_copy = features.copy()
        size = len(features_copy)
        for i in range(0,size):
            item = features_copy[i]
            score = features_score[labels[i]]

            size_score = len(score)
            item_update = []
            for j in range(0,size_score):
                word = score[j][0]
                if word in item:
                    item_update.append(word)
                    if (len(item_update) == number_words):
                        break
            features_copy[i] = item_update

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
        





    
    
    

        
                
            
        
        

  
