"""
Class that holds a genetic algorithm for evolving a network.

Credit:
    A lot of those code was originally inspired by:
    http://lethain.com/genetic-algorithms-cool-name-damn-simple/
"""
from functools import reduce
from operator import add
import random
from network import Network
from statistics import mean

class Optimizer():
    """Class that implements genetic algorithm for MLP optimization."""

    def __init__(self, nn_param_choices, retain=0.4, random_select=0.1, mutate_chance=0.2):
        """Create an optimizer.

        Args:
            nn_param_choices (dict): Possible network paremters
            retain (float): Percentage of population to retain after
                each generation
            random_select (float): Probability of a rejected network
                remaining in the population
            mutate_chance (float): Probability a network will be
                randomly mutated

        """
        self.mutate_chance = mutate_chance
        self.random_select = random_select
        self.retain = retain
        self.nn_param_choices = nn_param_choices

    def create_population(self, count):
        """Create a population of random networks.

        Args:
            count (int): Number of networks to generate, aka the
                size of the population

        Returns:
            (list): population of network objects

        """
        population = [Network(self.nn_param_choices) for x in range(0,count)]
        for network in population:
            network.create_random()

        return population

    @staticmethod
    def fitness(network):
        """Return the accuracy, which is our fitness function."""
        return network.accuracy

    def grade(self, population):
        """Find average fitness for a population.

        Args:
            population (list): The population of networks

        Returns:
            (float): The average accuracy of the population

        """

        return float(mean(child.accuracy for child in population))

    def breed(self, mother, father):
        """Make two children as parts of their parents.

        Args:
            mother (dict): Network parameters
            father (dict): Network parameters

        Returns:
            (list): Two network objects
        
        """
        childA, childB = Network(), Network()

        crossOverPoint = random.randint(1, len(mother)-1)
        childAParams = {**{k: mother[k] for k in list(mother.keys())[:crossOverPoint]}, **{k: father[k] for k in list(father.keys())[crossOverPoint:]}}
        childBParams = {**{k: father[k] for k in list(father.keys())[:crossOverPoint]}, **{k: mother[k] for k in list(mother.keys())[crossOverPoint:]}}

        
        childA.create_set(childAParams)
        childB.create_set(childBParams)


        return [childA, childB]

    def mutate(self, network):
        """Randomly mutate one part of the network.

        Args:
            network (dict): The network parameters to mutate

        Returns:
            (Network): A randomly mutated network object

        """
        mutateChance = random.random()
        hyperparameter_to_mutate = random.choice(list(self.nn_param_choices.keys()))


        if mutateChance <= self.mutate_chance:
            network[hyperparameter_to_mutate] = random.choice(self.nn_param_choices[hyperparameter_to_mutate])

        return network

    def evolve(self, population):
        """Evolve a population of networks.

        Args:
            population (list): A list of network parameters

        Returns:
            (list): The evolved population of networks

        """
        originalLength = len(sortedPopulation)
        retainLength = self.retain * len(population)
        numNewPopulationLength = originalLength - retainLength

        print(population + "is here")
        # evaluate population fitness and sort
        sortedPopulation = sorted(population, key = lambda item: item.fitness())





        # while not 2 parents appended

            # select parent from population using select chance

            # breed two parents

            # mutate offspring

            # evaluate fitness of offspring

            # replace parents with offspring




        return
