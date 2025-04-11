import random
import pandas as pd
import numpy as np
random.seed(3.14159)


df = pd.read_excel('../data/kidr_activity.xlsx', header=None)
df = df.drop([0,1,2]) #Drops the rows that are not part of the data.
df.columns = df.iloc[0].to_list() #Designates the relevant row as the header.
df = df[1:]
#print(df.head())
#print(df.columns)

var = ['SND number','Size','Days passed','#Canceled','Number of Requests', 'Access Granted*'] #relevant variables
#The size of the datasets are given in MB
df = df[var]
df = df.reset_index(drop=True)
columns_to_fill = ['Number of Requests', 'Access Granted*','#Canceled'] 
df[columns_to_fill] = df[columns_to_fill].fillna(0) #Filling in the empty rows as zeros
df['Size'] = df['Size'].astype('float') #Since the class of 'Size' is initially strings we convert it to float
"""
print(df.head())
print(df.loc[1, 'Size'])
print(type(df.loc[1, 'Size']))
print(type(df.loc[1, 'Days passed']))
print(type(df.loc[1, 'Number of Requests']))
"""
#Constructing the hyperparameters
Weights = (df['Size']*1000).tolist() #MB
#print(Weights)
Values = (1000*(df['Access Granted*']+1/2*(df['Number of Requests']-1/2*df['#Canceled'])+1))/df['Days passed'].astype(float).tolist()
#print(Values)
#print(sum(Weights))
Max_capacity = int(0.1*1000*1000*1000) #Max capacity is 70TB = tot cap of KI 
#We test different constructed max capacities to restrain the knapsack more

def genetic_knapsack(weights, values, max_capacity, population_size, generations, mutation_rate):
    n = len(weights)

    def generate_chromosome():
        return [random.randint(0, 1) for _ in range(n)]

    def calculate_fitness(chromosome):
        total_weight = sum(weights[i] for i in range(n) if chromosome[i] == 1)
        total_value = sum(values[i] for i in range(n) if chromosome[i] == 1)
        #print(f"Total weight: {total_weight}, Total value: {total_value}")

        if total_weight <= max_capacity:
            return total_value
        else:
            return 0  # Return 0 if capacity is exceeded

    def selection(population, fitness_values):
        selected_parents = random.choices(population, weights=fitness_values, k=len(population))
        return selected_parents

    def crossover(parent1, parent2):
        crossover_point = random.randint(1, n - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2

    def mutate(chromosome, mutation_rate):
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
weights = [2, 5, 10, 5]
values = [40, 30, 50, 10]
max_capacity = 16
population_size = 100
generations = 100
mutation_rate = 0.01

best_fitness, selected_items, used_capacity = genetic_knapsack(Weights, Values, Max_capacity, population_size, generations, mutation_rate)

print(sum(Values[i] for i in selected_items))
print("Best fitness:", best_fitness)
print("Selected items:", selected_items)
print("Used capacity:", used_capacity)
a = sum(Weights[i] for i in selected_items)
Total_weight = sum(Weights)
print(Total_weight)
print(a)
print(a/Max_capacity)
