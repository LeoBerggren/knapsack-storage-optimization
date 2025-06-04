import random
import pandas as pd
import numpy as np
import time

random.seed(438579088)


df = pd.read_excel('data/kidr_activity.xlsx', header=None)
df = df.drop([0,1,2]) #Drops the rows that are not part of the data.
df.columns = df.iloc[0].to_list() #Designates the relevant row as the header.
df = df[1:]
#print(df.head())
#print(df.columns)

var = ['SND number','Size','Days passed','#Canceled','Number of Requests', 'Access Granted*','Visits'] #relevant variables
#The size of the datasets are given in MB
df = df[var]
df = df.reset_index(drop=True)
columns_to_fill = ['Number of Requests', 'Access Granted*','#Canceled','Visits'] 
df[columns_to_fill] = df[columns_to_fill].fillna(0) #Filling in the empty rows as zeros
df['Size'] = df['Size'].astype('float') #Since the class of 'Size' is initially strings we convert it to float

df_ned = pd.read_excel('data/nedladdningar.xlsx', header=None)
df_ned.columns = df_ned.iloc[0].to_list() #Designates the relevant row as the header.
df_ned = df_ned[1:]

var_ned = ['Dataset','Filtyp']
df_ned = df_ned[var_ned]

data = np.zeros(len(df)) #Vector of 62 zeros
doc = np.zeros(len(df))

for index_1, value_1 in df['SND number'].items():
    index_1_int: int = int(index_1)  # Explicitly cast to int
    for index_2, value_2 in df_ned['Dataset'].items():
        index_2_int: int = int(index_2)  # Explicitly cast to int
        filtyp_value2: str = df_ned['Filtyp'][index_2_int] #explicitly declare that the value is a string.
        if value_2 == value_1:
                if df_ned['Filtyp'][index_2_int] == 'dokumentation':
                    doc[index_1_int] += 1
                elif df_ned['Filtyp'][index_2_int]== 'data':
                    data[index_1_int] += 1
                else:
                    print('something wrong')
        else:
            pass

data = data.astype(int)


#Constructing the hyperparameters
A = 2158/233
B = 5264/2158
C = 10662/5264
df['data downloads'] = data
df['doc downloads'] = doc
#Constructing the hyperparameters
Weights = (df['Size']*1000).tolist() #converts from MB to KB
Values = ((A*B*C*(3/2*df['Access Granted*']+(df['Number of Requests']-1/2*df['#Canceled']))+(B*C*df['data downloads']+C*df['doc downloads'])+df['Visits']))/df['Days passed'].astype(float).tolist()
Max_capacity = int(0.5*1000*1000*1000) #Max capacity is set to 500GB. Other values might be appropriate

def genetic_knapsack(weights, values, max_capacity, population_size, generations, mutation_rate):
    """
    Solves the 0/1 Knapsack problem using a genetic algorithm.

    Args:
        weights: A list of the weights of the items.
        values: A list of the values of the items.
        max_capacity: The maximum capacity of the knapsack.
        population_size: the number of solutions (aka "chromosomes") generated for each generation
        generations: the number of generations that the algorithm iterates through
        mutation_rate: the likelyhood of a random change in a chromosome

    Returns:
        A tuple containing:
            - The best fitness aka maximum value that can be carried in the knapsack.
            - A list of the indices of the items included in the optimal solution.
            - The optimal solutions used capacity
    """
    n = len(weights)

    def generate_chromosome():
        """
        Generates a random chromosome/"solution"

        Returns:
            a list of length n of random integers of either zeros or ones.
        
        """
        return [random.randint(0, 1) for _ in range(n)]

    def calculate_fitness(chromosome):
        """
        Calculates a fitness value of a solution

        Args: 
            - chromosome: a possible solution for the problem
        
        Returns: 
            Either 0 if the proposed solution has an higher total weight than the max capacity,
            or total_value if it does not.
        """
        total_weight = sum(weights[i] for i in range(n) if chromosome[i] == 1)
        total_value = sum(values[i] for i in range(n) if chromosome[i] == 1)
        #print(f"Total weight: {total_weight}, Total value: {total_value}")

        if total_weight <= max_capacity:
            return total_value
        else:
            return 0  # Return 0 if capacity is exceeded

    def selection(population, fitness_values):
        """
        Selects a number of parents equal to the population from the population, semirandomly with chromosomes with higher fitness_values being more likely to be chosen. It is done with replacement.

        Args:
            - population: a list of chromosomes or solutions 
            - fitness_values: corresponding fitness values of the population
        Returns:
            A list of "parents" equal in size to the population
        """

        selected_parents = random.choices(population, weights=fitness_values, k=len(population))
        return selected_parents

    def crossover(parent1, parent2):
        """
        splits two parents randomly and adds the different parts with each other, making a "child"
        
        Args:
            - parent1: first parent
            - parent2: second parent

        Returns
            two "children" each made up of one different part of each parent.
        """
        crossover_point = random.randint(1, n - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2

    def mutate(chromosome, mutation_rate):
        """
        Randomly mutates a chromosome depending on some mutation rate

        Args:
            - chromosome: the chromosome to be or not be mutaded
            - mutation_rate: the likelyhood of mutation

        Returns:
            The chromosome, mutated or not.
        """
        
        for i in range(n):
            if random.random() < mutation_rate:
                chromosome[i] = 1 - chromosome[i]
        return chromosome

    population = [generate_chromosome() for _ in range(population_size)]

    for generation in range(generations):
        fitness_values = [calculate_fitness(chromosome) for chromosome in population]
        if np.count_nonzero(fitness_values) == 0:
            print("Dead start, try again")
            return 0, [], 0  # Return placeholder values or raise an exception.
        #print(f"Generation {generation}: Fitness values = {fitness_values}")
        parents = selection(population, fitness_values)
        new_population = []
        for i in range(0, population_size, 2):
            if i + 1 < population_size:
                parent1, parent2 = parents[i], parents[i + 1]
                child1, child2 = crossover(parent1, parent2)
                child1 = mutate(child1, mutation_rate)
                child2 = mutate(child2, mutation_rate)
                new_population.extend([child1, child2])
        population = new_population

    # Find the best chromosome
    fitness_values = [calculate_fitness(chromosome) for chromosome in population]
    best_chromosome = population[fitness_values.index(max(fitness_values))]
    best_fitness = max(fitness_values)

    selected_items = [i for i in range(n) if best_chromosome[i] == 1]
    used_capacity = sum(weights[i] for i in selected_items)

    return best_fitness, selected_items, used_capacity

# Example Usage
population_size = 200
generations = 250
mutation_rate = 0.012

time_start = time.process_time()
best_fitness, selected_items, used_capacity = genetic_knapsack(Weights, Values, Max_capacity, population_size, generations, mutation_rate)
time_end = time.process_time()

print('Time spent calculating: ', time_end-time_start)
print(sum(Values[i] for i in selected_items))
print("Best fitness:", best_fitness)
print("Selected items:", selected_items)
print("Used capacity:", used_capacity)
a = sum(Weights[i] for i in selected_items)
Total_weight = sum(Weights)
print(Total_weight)
print(a)
print(a/Max_capacity)

#Currently best found value: 17.68837... with Hpar: Mrate = 0.012, Pop=200, Gen=250 

print('Total Value of All Items: ', sum(Values))
print('Total Weight of All Items: ', sum(Weights))