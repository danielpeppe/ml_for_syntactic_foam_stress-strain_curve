
def normalise_gaussian(x, mu, sigma, denormalise=False, denormalise_MSE=False):
    if denormalise:
        return x * sigma + mu
    elif denormalise_MSE:
        return x * sigma ** 2
    else:
        return (x - mu) / sigma


def normalise_minmax(x, min_val, max_val, denormalise=False, denormalise_MSE=False):
    if denormalise:
        return x * (max_val - min_val) + min_val
    elif denormalise_MSE:
        return x * (max_val - min_val) ** 2
    else:
        return (x - min_val) / (max_val - min_val)