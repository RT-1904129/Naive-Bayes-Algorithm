import csv
import json
import numpy as np


def Import_data():
    X=np.genfromtxt("train_X_nb.csv",delimiter='\n',dtype=str)
    Y=np.genfromtxt("train_Y_nb.csv",delimiter='\n',dtype=str)
    return X,Y


def preprocessing(train_X):
    for i in range(len(train_X)):
        s=train_X[i].split()
        str1=""
        for word in s:
            word=list([val for val in word if val.isalpha()])
            if len(word)!=0 :
                str1=str1+("".join(word)).lower()+ " "
        train_X[i]= str1
    
    return train_X
        
    
def class_wise_words_frequency_dict(X, Y):
    class_sentence=dict()
    for i in range(len(Y)):
        if Y[i] not in class_sentence:
            class_sentence[Y[i]]=""
        class_sentence[Y[i]]=class_sentence[Y[i]]+" "+X[i]
        
    class_wise_words_frequency=dict()
    for class_name in class_sentence.keys():
        class_word_dict=dict()
        sentence=class_sentence[class_name].split()
        for word in sentence:
            if word not in class_word_dict:
                class_word_dict[word]=0
            class_word_dict[word]+=1
        class_wise_words_frequency[class_name]=class_word_dict
        
    return  class_wise_words_frequency


def compute_prior_probabilities(Y):
    total_number=len(Y)
    classes_freq=dict()
    prior_probabilities_dict=dict()
    for i in Y:
        if i not in classes_freq.keys():
            classes_freq[i]=0
        classes_freq[i]+=1
    for i in classes_freq.keys():
        prior_probabilities_dict[i]=classes_freq[i]/total_number
    return prior_probabilities_dict


def get_class_wise_denominators_likelihood(X, Y):
    class_wise_words_frequency=class_wise_words_frequency_dict(X,Y)
    Total_string=""
    for i in X:
        Total_string=Total_string+" "+i
    total_unique_word=len(list(set(Total_string.split())))
    sum_word_in_class_dict=dict()
    for classes in class_wise_words_frequency.keys():
        Total_sum_word_in_class=sum(list(class_wise_words_frequency[classes].values()))
        # considering smmothing hyperparmeter alpha=1
        sum_word_in_class_dict[classes]=Total_sum_word_in_class+total_unique_word
    return sum_word_in_class_dict


def train_model(train_X,train_Y) :
    dictionary ={
        "class_wise_frequency_dict" : class_wise_words_frequency_dict(train_X,train_Y),
        "prior_probabilities" :compute_prior_probabilities(train_Y),
        "class_wise_denominators":get_class_wise_denominators_likelihood(train_X,train_Y),
    }

    with open("MODEL_FILE.json", "w") as outfile:
        json.dump(dictionary, outfile)

    

if __name__=="__main__":
    X,Y=Import_data()
    X=preprocessing(X)
    train_model(X,Y)