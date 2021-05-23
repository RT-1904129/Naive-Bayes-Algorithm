import numpy as np
import csv
import sys
import json

from validate import validate


def import_data_and_model(test_X_file_path, model_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter='\n', dtype=str)
    with open(model_file_path, "r") as read_file:
        model = json.load(read_file)
    return test_X, model

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
  


def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()

def compute_likelihood(test_X, classes,class_wise_frequency_dict, class_wise_denominators):
    likelihood = 0
    words = test_X.split()
    for word in words:
        count = 0
        words_frequency = class_wise_frequency_dict[classes]
        if word in words_frequency:
            count = class_wise_frequency_dict[classes][word]
        # considering smmothing hyperparmeter alpha=1
        likelihood += np.log((count + 1)/class_wise_denominators[classes])
    return likelihood


def predict_target_values(test_X, model):
    class_wise_frequency_dict=model["class_wise_frequency_dict"]
    class_wise_denominators=model["class_wise_denominators"]
    prior_probabilities=model["prior_probabilities"]
    total_class=list(set(prior_probabilities.keys()))
    predicted_value_list=[]
    for test_string in test_X :
        best_predicted_value=0
        best_prior_multiply_likelihood=-9999999
        for classes in total_class:
            likelihood=compute_likelihood(test_string,classes,class_wise_frequency_dict, class_wise_denominators)
            probability_exact_prior_likelihood=np.log(prior_probabilities[classes])+likelihood
            if (probability_exact_prior_likelihood>best_prior_multiply_likelihood):
                best_predicted_value=classes
                best_prior_multiply_likelihood=probability_exact_prior_likelihood
        predicted_value_list.append(best_predicted_value)
    return np.array(predicted_value_list)


def predict(test_X_file_path):
    test_X, model = import_data_and_model(test_X_file_path, "./MODEL_FILE.json")
    test_X=preprocessing(test_X)
    pred_Y = predict_target_values(test_X, model)
    write_to_csv_file(pred_Y, "predicted_test_Y_nb.csv")    


if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    # Uncomment to test on the training data
    # validate(test_X_file_path, actual_test_Y_file_path="train_Y_nb.csv") 