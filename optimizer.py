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
        population = [Network(self.nn_param_choices) for _ in range(0,count)]
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
        
        childAParams = {k: mother[k] for k in list(mother.keys())[:crossOverPoint]} \
                    | {k: father[k] for k in list(father.keys())[crossOverPoint:]}
        childBParams = {k: father[k] for k in list(father.keys())[:crossOverPoint]} \
                    | {k: mother[k] for k in list(mother.keys())[crossOverPoint:]}

        
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

        mutatedNetwork = Network()
        mutatedNetwork.create_set(network)

        return mutatedNetwork

    def evolve(self, population):
        """Evolve a population of networks.

        Args:
            population (list): A list of network objects

        Returns:
            (list): The evolved population of networks objects

        """
        originalLength = len(population)
        retainLength = int(self.retain * originalLength)

        sortedPopulation = sorted(population, key=lambda x: x.accuracy, reverse=True)
        evolvedPopulation = sortedPopulation[:retainLength]

        # random select from rejected by chance:
        for i in range(retainLength, originalLength):
            if random.random() <= self.random_select:
                evolvedPopulation.append(sortedPopulation[i])

        numChildrenNeeded = originalLength - len(evolvedPopulation)

        while numChildrenNeeded > 0:
            # select two parents:
            parent1, parent2 = evolvedPopulation[0], evolvedPopulation[1]
            [child1, child2] = self.breed(parent1.network, parent2.network)

            if numChildrenNeeded == 1:
                evolvedPopulation.append(child1)
                numChildrenNeeded -= 1

            evolvedPopulation.append(child1)
            evolvedPopulation.append(child2)
            numChildrenNeeded -= 2

        # Sort the evolved population by fitness
        evolvedPopulation = sorted(evolvedPopulation, key=lambda x: x.accuracy, reverse=True)

        # Truncate the population to the original size
        evolvedPopulation = evolvedPopulation[:originalLength]

        return evolvedPopulation
