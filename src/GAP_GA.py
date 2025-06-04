import random
import numpy as np
import heapq
import pandas as pd
import time
# Set random seed #42, 49, #112
random.seed(42)



df = pd.read_excel('data/kidr_activity.xlsx', header=None)
df = df.drop([0,1,2]) #Drops the rows that are not part of the data.
df.columns = df.iloc[0].to_list() #Designates the relevant row as the header.
df = df[1:]

var = ['SND number','Size','Days passed','#Canceled','Number of Requests', 'Access Granted*','Visits'] #relevant variables
#The size of the datasets are given in MB
df = df[var]
df = df.reset_index(drop=True)
columns_to_fill = ['Number of Requests', 'Access Granted*','#Canceled','Visits'] 
df[columns_to_fill] = df[columns_to_fill].fillna(0) #Filling in the empty rows as zeros
df['Size'] = df['Size'].astype('float')

df_ned = pd.read_excel('data/nedladdningar.xlsx', header=None)
df_ned.columns = df_ned.iloc[0].to_list() #Designates the relevant row as the header.
df_ned = df_ned[1:]

var_ned = ['Dataset','Filtyp']
df_ned = df_ned[var_ned]

#pseudocode
data = np.zeros(len(df)) #Vector of 62 zeros
doc = np.zeros(len(df))


#for each dataset (those in df) this will go through each download in df_ned for that dataset and count it as data
# or as document depending on its 'filtyp'. 
# if it is not marked as 'document' or 'data' it will raise an keyerror since something is wrong with the spreadsheet
# If it does not find anymore or any it will go to the next item. 

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
#We test different constructed max capacities to restrain the knapsack more

#First results with pop=100, gen=300, mut=0.05
def genetic_algorithm(values, weights, capacities, population_size=100, generations=300, mutation_rate=0.05):
    num_agents = len(values)
    num_tasks = len(values[0])

    def create_individual():
        return [random.choice(range(-1, num_agents)) for _ in range(num_tasks)]

    def fitness(individual):
        total_value = 0
        agent_loads = [0] * num_agents

        for t, agent in enumerate(individual):
            if agent == -1:
                continue
            task_weight = weights[agent][t]
            task_value = values[agent][t]
            if agent_loads[agent] + task_weight <= capacities[agent]:
                agent_loads[agent] += task_weight
                total_value += task_value
            else:
                return 0  # Heavy penalty for invalid solutions

        return total_value

    def mutate(individual):
        for i in range(num_tasks):
            if random.random() < mutation_rate:
                individual[i] = random.choice(range(-1, num_agents))
        return individual

    def crossover(parent1, parent2):
        point = random.randint(1, num_tasks - 2)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]        
        return child1, child2

    # Initialize population
    population = [create_individual() for _ in range(population_size)]

    best_solution = None
    best_fitness = -float('inf')

    for gen in range(generations):
        population.sort(key=fitness, reverse=True)

        if fitness(population[0]) > best_fitness:
            best_fitness = fitness(population[0])
            best_solution = population[0]
            print(f"Gen {gen}: New best value: {best_fitness:.4f}")

        next_gen = population[:10]  # Elitism: keep top 10

        while len(next_gen) < population_size:
            parent1, parent2 = random.choices(population[:50], k=2)  # Select top half /50
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            next_gen.extend([child1,child2])

        population = next_gen

    return best_fitness, best_solution

Capacities = [0.1*Max_capacity, 1.8*Max_capacity]
Multi_values = [1*Values, [0.5 * v for v in Values]]
Multi_weights = [Weights, Weights]

st = time.process_time_ns()
Max_val, Assignment = genetic_algorithm(Multi_values, Multi_weights, Capacities)
et = time.process_time_ns()
time = et-st
print('Time: ', time)
print("Max total value (GA):", Max_val)
print("Task assignments (task -> agent):", Assignment)