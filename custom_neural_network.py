from network import Network
from tqdm import tqdm
import logging

# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    filename='log.txt'
)

def train_networks(networks, dataset):
    """Train each network.

    Args:
        networks (list): Current population of networks
        dataset (str): Dataset to use for training/evaluating
    """
    pbar = tqdm(total=len(networks))
    for network in networks:
        network.train(dataset)
        pbar.update(1)
    pbar.close()

def print_networks(networks):
    """Print a list of networks.

    Args:
        networks (list): The population of networks

    """
    logging.info('-'*80)
    for network in networks:
        network.print_network()
        print(network.accuracy)
        


def get_average_accuracy(networks):
    """Get the average accuracy for a group of networks.

    Args:
        networks (list): List of networks

    Returns:
        float: The average accuracy of a population of networks.

    """
    total_accuracy = 0
    for network in networks:
        total_accuracy += network.accuracy

    return total_accuracy / len(networks)


def generate(networks, dataset):
    # train network
    train_networks(networks, dataset)

    # Get the average accuracy for this generation.
    average_accuracy = get_average_accuracy(networks)
    logging.info("Generation average: %.2f%%" % (average_accuracy * 100))
    logging.info('-'*80)

    # Sort our final population.
    networks = sorted(networks, key=lambda x: x.accuracy, reverse=True)
    # Print out the top 5 networks.
    print_networks(networks)
    logging.info('-'*80)

nn_param_choices = {
        'nb_neurons': [64, 128, 256, 512, 768, 1024],
        'nb_layers': [1, 2, 3, 4],
        'activation': ['relu', 'elu', 'tanh', 'sigmoid'],
        'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad','adadelta', 'adamax', 'nadam'],
    }


networkParams = [{'nb_neurons': 64, 'nb_layers': 1, 'activation': 'relu', 'optimizer': 'sgd'},\
        {'nb_neurons': 128, 'nb_layers': 1, 'activation': 'relu', 'optimizer': 'sgd'},\
        {'nb_neurons': 256, 'nb_layers': 1, 'activation': 'relu', 'optimizer': 'sgd'},\
        {'nb_neurons': 512, 'nb_layers': 1, 'activation': 'relu', 'optimizer': 'sgd'},\
        {'nb_neurons': 768, 'nb_layers': 1, 'activation': 'relu', 'optimizer': 'sgd'},\
        {'nb_neurons': 1024, 'nb_layers': 1, 'activation': 'relu', 'optimizer': 'sgd'}]

# create networks and set their params
networkList = []
for params in networkParams:
    newNetwork = Network()
    newNetwork.create_set(params)
    networkList.append(newNetwork)


dataset = 'cifar10'
logging.info(f'Dataset: {dataset}')
generate(networkList,dataset)

# train_networks(networkList)

# # Get the average accuracy for this generation.
# average_accuracy = get_average_accuracy(networkList)

# # Sort our final population.
# networks = sorted(networkList, key=lambda x: x.accuracy, reverse=True)

# # Print out the networks.
# print_networks(networkList)


