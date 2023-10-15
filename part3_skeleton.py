import math, random
import matplotlib.pyplot as plt

""" Globals """

DOMAIN = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
LABELS = ["front page", "news", "tech", "local", "opinion", "on-air", "misc", "weather", "msn-news", "health", "living", "business", "msn-sports", "sports", "summary", "bbs", "travel"]
N = 0
  

""" Helpers """
def calculate_average_error(actual_hist, noisy_hist):
    total = 0
    for idx in range(len(actual_hist)):
        total += abs(actual_hist[idx] - noisy_hist[idx])
    return total / len(actual_hist)

def read_dataset(filename):
    """
        Reads the dataset with given filename.
    """

    result = []
    with open(filename, "r") as f:
        for line in f:
            result.append(int(line))
    N = len(result)
    return result


# You can define your own helper functions here. #

### HELPERS END ###

""" Functions to implement """


# GRR

# TODO: Implement this function!
def perturb_grr(val, epsilon):
    p = math.exp(epsilon) / (math.exp(epsilon) + len(DOMAIN) - 1)
    coin = random.random()
    if coin <= p:
        return val
    else:
        ans = random.randint(1, 17)
        while ans == val:
            ans = random.randint(1, 17)
        return ans


# TODO: Implement this function!
def estimate_grr(perturbed_values, epsilon):
    p = math.exp(epsilon) / (math.exp(epsilon) + len(DOMAIN) - 1)
    q = (1 - p) / (len(DOMAIN) - 1)
    res = []
    
    for row in perturbed_values:
        nv = row
        Iv = (nv * p) + ((N - nv) * q)
        estimation = (Iv - (N*q)) / (p-q)
        res.append(estimation)
    plt.bar(DOMAIN, res)
    plt.xticks(DOMAIN)
    plt.show()
    return res


# TODO: Implement this function!
def grr_experiment(dataset, epsilon):
    histogram = dict.fromkeys(DOMAIN, 0)
    real = dict.fromkeys(DOMAIN, 0)
    for row in dataset:
        perturbed_val = perturb_grr(row, epsilon)
        real[row] += 1
        histogram[perturbed_val] += 1
    return calculate_average_error(list(real.values()), estimate_grr(list(histogram.values()), epsilon))        
    
        
        

# RAPPOR

# TODO: Implement this function!
def encode_rappor(val):
    bit_vec = [0] * len(DOMAIN)
    bit_vec[val - 1] = 1
    return bit_vec


# TODO: Implement this function!
def perturb_rappor(encoded_val, epsilon):
    p = (math.exp(epsilon / 2)) / (math.exp(epsilon / 2) + 1)
    q = 1 / (math.exp(epsilon / 2) + 1)
    res = [0] * len(encoded_val)
    for i in range(len(encoded_val)):
        coin = random.random()
        if coin <= p:
            res[i] = encoded_val[i]
        else:
            if encoded_val[i] == 0:
                res[i] = 1
            else:
                res[i] = 0
    return res


# TODO: Implement this function!
def estimate_rappor(perturbed_values, epsilon):
    p = (math.exp(epsilon / 2)) / (math.exp(epsilon / 2) + 1)
    q = 1 / (math.exp(epsilon / 2) + 1)
    res = []
    cum = [0] * len(DOMAIN)
    for i in range(len(cum)):
        for row in perturbed_values:
            cum[i] += row[i]
    for row in cum:
        nv = row
        Iv = (nv * p) + ((N - nv) * q)
        estimation = (Iv - (N*q)) / (p-q)
        res.append(estimation)
    plt.bar(DOMAIN, res)
    plt.xticks(DOMAIN)
    plt.show()
    return res


# TODO: Implement this function!
def rappor_experiment(dataset, epsilon):
    pert_vals = []
    real = [0] * len(DOMAIN)
    for row in dataset:
        encoded = encode_rappor(row)
        pert_vals.append(perturb_rappor(encoded, epsilon))
        for i in range(len(real)):
            real[i] += encoded[i]
    return calculate_average_error(real, estimate_rappor(pert_vals, epsilon))        


# OUE

# TODO: Implement this function!
def encode_oue(val):
    bit_vec = [0] * len(DOMAIN)
    bit_vec[val - 1] = 1
    return bit_vec


# TODO: Implement this function!
def perturb_oue(encoded_val, epsilon):
    p = 1 / (math.exp(epsilon) + 1)
    res = [0] * len(encoded_val)
    for i in range(len(encoded_val)):
        if encoded_val[i] == 1:
            if random.randint(0, 1) == 0:
                res[i] = 1
            else:
                res[i] = 0
        else:
            if random.random() <= p:
                res[i] = 1
            else:
                res[i] = 0
    return res


# TODO: Implement this function!
def estimate_oue(perturbed_values, epsilon):
    n = len(perturbed_values)
    res = [0] * len(DOMAIN)
    for i in range(len(res)):
        C_hat = 0
        for row in perturbed_values:
            if row[i] == 1:
                C_hat += 1
        C = (2 * (((math.exp(epsilon) + 1) * C_hat) - n)) / (math.exp(epsilon) - 1)
        res[i] = C
    return res


# TODO: Implement this function!
def oue_experiment(dataset, epsilon):
    pert_vals = []
    real = [0] * len(DOMAIN)
    for row in dataset:
        encoded = encode_oue(row)
        pert_vals.append(perturb_oue(encoded, epsilon))
        for i in range(len(real)):
            real[i] += encoded[i]
    return calculate_average_error(real, estimate_rappor(pert_vals, epsilon))  


def main():
    dataset = read_dataset("msnbc-short-ldp.txt")
    
    print("GRR EXPERIMENT")
    for epsilon in [0.1, 0.5, 1.0, 2.0, 4.0, 6.0]:
        error = grr_experiment(dataset, epsilon)
        print("e={}, Error: {:.2f}".format(epsilon, error))
    
    print("*" * 50)

    print("RAPPOR EXPERIMENT")
    for epsilon in [0.1, 0.5, 1.0, 2.0, 4.0, 6.0]:
        error = rappor_experiment(dataset, epsilon)
        print("e={}, Error: {:.2f}".format(epsilon, error))

    print("*" * 50)

    print("OUE EXPERIMENT")
    for epsilon in [0.1, 0.5, 1.0, 2.0, 4.0, 6.0]:
        error = oue_experiment(dataset, epsilon)
        print("e={}, Error: {:.2f}".format(epsilon, error))


if __name__ == "__main__":
    main()

