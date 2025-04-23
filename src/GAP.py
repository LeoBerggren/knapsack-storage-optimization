import random
import numpy as np
import heapq
import pandas as pd
import time

random.seed(42) # Sets random seed

### DATA CONSTRUCTION ###
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

### PARAMETER CONSTRUCTION ###
A = 2158/233
B = 5264/2158
C = 10662/5264
df['data downloads'] = data
df['doc downloads'] = doc
Weights = (df['Size']*1000).tolist() #converts from MB to KB
Values = ((A*B*C*(3/2*df['Access Granted*']+(df['Number of Requests']-1/2*df['#Canceled']))+(B*C*df['data downloads']+C*df['doc downloads'])+df['Visits']))/df['Days passed'].astype(float).tolist()

Max_capacity = int(0.1*1000*1000*1000) #Max capacity is 70TB = tot cap of KI 
#We test different constructed max capacities to restrain the knapsack more

## Specs of capacities, weights & values of knapsacks/agents ##
Capacities = [0.5*Max_capacity, Max_capacity]
Multi_values = [1*Values, [0.8 * v for v in Values]]
Multi_weights = [Weights, Weights]

### ALGORITHMS ###

## GENETIC ALGORITHM ##
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
        return parent1[:point] + parent2[point:]

    # Initialize population
    population = [create_individual() for _ in range(population_size)]

    best_solution = None
    best_fitness = -float('inf')

    for gen in range(generations):
        population.sort(key=fitness, reverse=True)

        if fitness(population[0]) > best_fitness:
            best_fitness = fitness(population[0])
            best_solution = population[0]
            #print(f"Gen {gen}: New best value: {best_fitness:.4f}")

        next_gen = population[:10]  # Elitism: keep top 10

        while len(next_gen) < population_size:
            parent1, parent2 = random.choices(population[:50], k=2)  # Select top 50
            child = crossover(parent1, parent2)
            child = mutate(child)
            next_gen.append(child)

        population = next_gen

    return best_fitness, best_solution

## BRANCH AND BOUND ALGORITHM ##
def gap_branch_and_bound(values, weights, capacities):
    num_agents = len(values)
    num_tasks = len(values[0])

    # Sort tasks by best value-to-weight ratio (across all agents)
    ratios = [max(values[a][t] / weights[a][t] if weights[a][t] > 0 else 0 for a in range(num_agents)) for t in range(num_tasks)]
    sorted_indices = sorted(range(num_tasks), key=lambda t: -ratios[t])

    # Initial state: (negative bound, current value, task index, agent loads, assignment)
    heap = [(-float('inf'), 0, 0, [0] * num_agents, [-1] * num_tasks)]
    best_value = 0
    best_assignment = None
    max_iterations = 10000000
    iteration_count = 0

    while heap:
        iteration_count += 1
        if iteration_count > max_iterations:
            print("🔴 Max iteration limit reached.")
            break

        _ , total_value, task_idx, agent_loads, assignment = heapq.heappop(heap)

        if task_idx >= num_tasks:
            if total_value > best_value:
                print(f"✅ New best: value={total_value:.4f}, iterations={iteration_count}")
                best_value = total_value
                best_assignment = assignment
            continue

        task = sorted_indices[task_idx]

        for agent in range(num_agents):
            task_weight = weights[agent][task]
            task_value = values[agent][task]
            if agent_loads[agent] + task_weight <= capacities[agent]:
                new_value = total_value + task_value
                new_agent_loads = agent_loads.copy()
                new_agent_loads[agent] += task_weight
                new_assignment = assignment.copy()
                new_assignment[task] = agent

                # Estimate upper bound from this state
                est_bound = new_value
                for next_task_idx in range(task_idx + 1, num_tasks):
                    next_task = sorted_indices[next_task_idx]
                    #print('next tasks: ',range(task_idx+1,num_tasks))                    
                    #print('next task: ',next_task)
                    best_ratio = 0
                    for a in range(num_agents):
                        if weights[a][next_task] > 0:
                            ratio = values[a][next_task] / weights[a][next_task]
                            if new_agent_loads[a] + weights[a][next_task] <= capacities[a]:
                                best_ratio = max(best_ratio, ratio)
                    est_bound += best_ratio * 1  # Assume we can pick the best agent for this task
                eps = 3
                if est_bound > best_value-eps:
                    heapq.heappush(heap, (-est_bound, new_value, task_idx + 1, new_agent_loads, new_assignment))
                #else:
                    #print(f"✘ Pruned (low bound): task {task} -> agent {agent}, est_bound={est_bound:.4f}, current best={best_value:.4f}")
            #else:
                #print(f"✘ Pruned (capacity): task {task} -> agent {agent}, load={agent_loads[agent]:,.2f}, task_weight={task_weight:,.2f}, cap={capacities[agent]:,.2f}")

        # Also consider skipping the task entirely (unassigned)
        est_bound = total_value
        for next_task_idx in range(task_idx + 1, num_tasks):
            next_task = sorted_indices[next_task_idx]
            for a in range(num_agents):
                if weights[a][next_task] > 0 and agent_loads[a] + weights[a][next_task] <= capacities[a]:
                    est_bound += values[a][next_task]
                    break  # Assign to first eligible agent

        heapq.heappush(heap, (-est_bound, total_value, task_idx + 1, agent_loads.copy(), assignment))

    return best_value, best_assignment

### RESULTS ###
st = time.process_time_ns()
Max_val_bnb, Assignment_bnb = gap_branch_and_bound(Multi_values, Multi_weights, Capacities)
et = time.process_time_ns()
time_bnb = et - st

st = time.process_time_ns()
Max_val_gen, Assignment_gen = genetic_algorithm(Multi_values, Multi_weights, Capacities)
et = time.process_time_ns()
time_gen = et - st

print('Algorithms done.')

## Double checking Values, and measuring used storage ##

Value_gen = 0 #Double check of returned value
Ws_gen = 0 #Warm storage used
Ls_gen = 0 #Lukewarm storage used
for ind, val in enumerate(Assignment_gen):
    if val == 1: #Lukewarm storage
        Value_gen += 0.8*Values[ind]
        Ls_gen += Weights[ind]
    elif val == 0: #Warm storage
        Value_gen += Values[ind]
        Ws_gen += Weights[ind]

Value_bnb = 0
Ws_bnb = 0
Ls_bnb = 0
for ind, val in enumerate(Assignment_bnb):
    if val == 1: #Lukewarm storage
        Value_bnb += 0.8*Values[ind]
        Ls_bnb += Weights[ind]
    elif val == 0: #Warm storage
        Value_bnb += Values[ind]
        Ws_bnb += Weights[ind]

## Measuring choices in common and objects included ##

set_diff = 0 # counts the instances the algs sends the object to different storage
set_int = 0 # count the instances the algs sends the object to the same storage
set_inc = 0 #Count the instances both algs include the object, with one in warm and one in lukewarm storage
W_st = 0 # Counts the instances objects are sorted to warm storage by both algs
LW_st = 0 # Counts the instance objects are sorted to lukewarm storage by both algs.
for i_bnb, v_bnb in enumerate(Assignment_bnb):
    for i_gen, v_gen in enumerate(Assignment_gen):
        if i_gen == i_bnb:
            if v_bnb == v_gen:
                set_int += 1
                if v_bnb == 1:
                    W_st += 1
                elif v_bnb == 0:
                    LW_st += 1
            else:
                if v_bnb + v_gen > 0:
                    set_inc += 1              
        else:
            pass

corr = set_int/len(Assignment_bnb) #percentage of choices in common
corr_wide = (set_int+set_inc)/len(Assignment_bnb) #percentage of choices in common if LW=W
inclusions = (W_st+LW_st+set_inc)/len(Assignment_bnb) #Percentage of items included in knapsacks


### PRESENTING RESULTS ###
print('BRANCH AND BOUND RESULTS')
print("Max total value:", Max_val_bnb)
print('Double check of value: ', Value_bnb)
print('Warm storage used: ', Ws_bnb, ' | Lukewarm storage used: ', Ls_bnb)
print('CPU Time required (nanoseconds): ', time_bnb)
#print("Task assignments (task -> agent):", Assignment_bnb)

print('GENETIC ALGORITHM RESULTS')
print("Max total value (GA):", Max_val_gen)
print('Double check of value: ', Value_gen)
print('Warm storage used: ', Ws_gen, ' | Lukewarm storage used: ', Ls_gen)
print('CPU Time required (nanoseconds): ', time_gen)
#print("Task assignments (task -> agent):", Assignment_gen)

print('OTHER RESULTS')
print('Percentage of same choices: ', corr)
print('Percentage of same choices LW=W: ', corr_wide)
print('percentage of items included in either LW or W: ', inclusions)