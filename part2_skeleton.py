import matplotlib.pyplot as plt
import numpy as np
import statistics
import pandas as pd
import math
import random

""" 
    Helper functions
    (You can define your helper functions here.)
"""


def read_dataset(filename):
    """
        Reads the dataset with given filename.
    """

    df = pd.read_csv(filename, sep=',', header = 0)
    return df

def getNumberOf10s(dataset, anime):
    result = 0
    for index, row in dataset.iterrows():
        rating = row[anime]
        if rating == 10:
            result += 1
    return result

def most10Rated(dataset):
    top10 = {}
    max_score = 0
    result = ""
    headers = dataset.columns.values.tolist()
    for header in headers:
        if header == "user_id":
            continue
        score = getNumberOf10s(dataset, header)
        if max_score < score:
            result = header
            max_score = score
    return result
def exponential_experiment_helper(dataset, epsilon):
    top10 = {}
    headers = dataset.columns.values.tolist()
    #query = number of rating 10s
    total = 0
    for header in headers:
        if header == "user_id":
            continue
        score = getNumberOf10s(dataset, header)
        #top10[header] = (epsilon * score) / (2 * 1)
        #total += (epsilon * score) / (2 * 1)
        top10[header] = np.exp((epsilon * score) / ( 2 * 1))
        total += top10[header]
    for header in headers:
        if header == "user_id":
            continue
        top10[header] = top10[header] / total
    return top10

### HELPERS END ###


''' Functions to implement '''

# TODO: Implement this function!
def get_histogram(dataset, chosen_anime_id="199"):
    histogram= {-1: 0, 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10:0}
    for index, row in dataset.iterrows():
        rating = row[chosen_anime_id]
        if not pd.isna(rating):
            histogram[rating] += 1
        else:
            histogram[-1] += 1
    lst = list(np.hstack([np.repeat(i, histogram[i]) for i in histogram.keys()]))
    plt.bar(list(histogram.keys()), list(histogram.values()), align='center')
    plt.xticks(list(histogram.keys()))
    plt.ylabel("Counts")
    plt.title("Rating Counts for Anime id=" + str(chosen_anime_id))
    plt.show()
    result = list(np.hstack([histogram[i]] for i in sorted(histogram.keys())))
    return result
                
                
        


# TODO: Implement this function!
def get_dp_histogram(counts, epsilon: float):
    ratings = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    b = 1 / epsilon
    lst = []
    result = []
    for idx in range(len(ratings)):
        noise = np.random.laplace(0, 1 / epsilon)
        result.append(counts[idx] + noise)
    return result
    

# TODO: Implement this function!
def calculate_average_error(actual_hist, noisy_hist):
    total = 0
    for idx in range(len(actual_hist)):
        total += abs(actual_hist[idx] - noisy_hist[idx])
    return total / len(actual_hist)


# TODO: Implement this function!
def calculate_mean_squared_error(actual_hist, noisy_hist):
    total = 0
    for idx in range(len(actual_hist)):
        total += (actual_hist[idx] - noisy_hist[idx])**2
    return total / len(actual_hist)


# TODO: Implement this function!
def epsilon_experiment(counts, eps_values: list):
    result_mse = []
    result_ae = []
    errors_mse = []
    errors_ae = []
    for e in eps_values:
        for i in range(40):
            noisy = get_dp_histogram(counts, e)
            errors_mse.append(calculate_mean_squared_error(counts, noisy))
            errors_ae.append(calculate_average_error(counts, noisy))
        result_mse.append(sum(errors_mse) / len(errors_mse))
        errors_mse.clear()
        result_ae.append(sum(errors_ae) / len(errors_ae))
        errors_ae.clear()
    return result_ae, result_mse


# FUNCTIONS FOR LAPLACE END #
# FUNCTIONS FOR EXPONENTIAL START #


# TODO: Implement this function!
def most_10rated_exponential(dataset, epsilon):
    top10 = {}
    headers = dataset.columns.values.tolist()
    #query = number of rating 10s
    total = 0
    for header in headers:
        if header == "user_id":
            continue
        score = getNumberOf10s(dataset, header)
        #top10[header] = (epsilon * score) / (2 * 1)
        #total += (epsilon * score) / (2 * 1)
        top10[header] = np.exp((epsilon * score) / ( 2 * 1))
        total += top10[header]
    for header in headers:
        if header == "user_id":
            continue
        top10[header] = top10[header] / total
    return np.random.choice(list(top10.keys()), 1, list(top10.values()))


            
    

# TODO: Implement this function!
def exponential_experiment(dataset, eps_values: list):
    ans = most10Rated(dataset)
    experiments = {}
    count = 0
    for e in eps_values:
        tmp = exponential_experiment_helper(dataset, e)
        for i in range(1000):
            result = np.random.choice(list(tmp.keys()), 1, list(tmp.values()))
            if result == ans:
                count += 1
        experiments[e] = (count/1000) * 100
        count = 0
    plt.plot(experiments.keys(), experiments.values(), "r.")
    plt.xlabel("ε")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs ε")
    plt.show()
    return list(experiments.values())
            


# FUNCTIONS TO IMPLEMENT END #

def main():
    filename = "anime-dp.csv"
    dataset = read_dataset(filename)

    counts = get_histogram(dataset)
    
    print("**** LAPLACE EXPERIMENT RESULTS ****")
    eps_values = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 1.0]
    error_avg, error_mse = epsilon_experiment(counts, eps_values)
    print("**** AVERAGE ERROR ****")
    for i in range(len(eps_values)):
        print("eps = ", eps_values[i], " error = ", error_avg[i])
    print("**** MEAN SQUARED ERROR ****")
    for i in range(len(eps_values)):
        print("eps = ", eps_values[i], " error = ", error_mse[i])


    print ("**** EXPONENTIAL EXPERIMENT RESULTS ****")
    eps_values = [0.001, 0.005, 0.01, 0.03, 0.05, 0.1]
    exponential_experiment_result = exponential_experiment(dataset, eps_values)
    for i in range(len(eps_values)):
        print("eps = ", eps_values[i], " accuracy = ", exponential_experiment_result[i])


if __name__ == "__main__":
    main()

