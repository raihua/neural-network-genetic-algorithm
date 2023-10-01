import unittest
from optimizer import Optimizer
from network import Network

class TestOptimiser(unittest.TestCase):
    def setUp(self):
        self.nn_param_choices = {
        'nb_neurons': [64, 128, 256, 512, 768, 1024],
        'nb_layers': [1, 2, 3, 4],
        'activation': ['relu', 'elu', 'tanh', 'sigmoid'],
        'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad','adadelta', 'adamax', 'nadam'],
        }

        self.optimizer = Optimizer(self.nn_param_choices)
        # self.network = Network(self.nn_param_choices)

        self.network1 = {'nb_neurons': 1024, 'nb_layers': 1, 'activation': 'relu', 'optimizer': 'sgd'}
        self.network2 = {'nb_neurons': 64, 'nb_layers': 3, 'activation': 'tanh', 'optimizer': 'adagrad'}

    def test_create_population(self):
        networkList = self.optimizer.create_population(1)

        #check type
        self.assertIsInstance(networkList[0], Network, "Population not created.")

        #check keys
        self.assertEqual(networkList[0].network.keys(), self.nn_param_choices.keys(), "Keys not equal.")

    def test_fitness(self):
        networkList = self.optimizer.create_population(1)
        networkList[0].accuracy = 20
        self.assertEqual(self.optimizer.fitness(networkList[0]), 20, "Fitness doesn't match.")

    def test_grade(self):
        networkList = self.optimizer.create_population(2)
        manualMeanFitnessPopulation = (networkList[0].accuracy + networkList[1].accuracy) / 2

        self.assertEqual(self.optimizer.grade(networkList), manualMeanFitnessPopulation, "Average does not match.")

    def test_breed(self):
        
        offspring = self.optimizer.breed(self.network1, self.network2)

        self.assertEqual(len(offspring), 2, "Not a list of 2 objects.")
        self.assertEqual(type(offspring[0]), Network, "Offspring not a network.")
        self.assertNotEqual(offspring[0], self.network1, "Offspring still the same as parent.")
        
    def test_mutate(self):
        self.optimizer.mutate_chance = 0.99
        network1original = {'nb_neurons': 1024, 'nb_layers': 1, 'activation': 'relu', 'optimizer': 'sgd'}
        mutatedNetwork = self.optimizer.mutate(self.network1)
        self.assertNotEqual(mutatedNetwork.network.items(), network1original.items(), "Failed to mutate")

    

if __name__ == '__main__':
    unittest.main()