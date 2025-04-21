import pandas as pd
import numpy as np
import heapq
import random
import time
random.seed(438579088)


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
#print('datapoints: ', len(df))


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

#print("data downloads: ", data)
#print('Total data downloads: ', sum(data))
#print('document downloads: ', doc)
#print("Total document downloads: ", sum(doc))
#print('total downloads:',np.sum(doc)+np.sum(data))            


### PARAMETER CONSTRUCTION ###
A = 2158/233
B = 5264/2158
C = 10662/5264
df['data downloads'] = data
df['doc downloads'] = doc
#Constructing the hyperparameters
Weights = (df['Size']*1000).tolist() #converts from MB to KB
#print(Weights)
Values = ((A*B*C*(3/2*df['Access Granted*']+(df['Number of Requests']-1/2*df['#Canceled']))+(B*C*df['data downloads']+C*df['doc downloads'])+df['Visits']))/df['Days passed'].astype(float).tolist()
#print(Values)
#print(sum(Weights))
Max_capacity = int(0.1*1000*1000*1000) #Max capacity is 70TB = tot cap of KI 
#We test different constructed max capacities to restrain the knapsack more

# DP SPECIFIC PARAMETERS #
dp_weights = [w/1000+1 for w in Weights] #makes sure that we won't have a zero when we round the weights 
dp_weights = [round(w) for w in dp_weights] #also converts the weights to Mb
Max_capacity_dp = int(Max_capacity/1000) #Converts max_capacity to Mb

#REVERSE DP SPECIFIC PARAMETERS #
Round_values = round(10000*Values) #Integer conversion of values, different powers of 10 gives varying specificity
#print(Round_values)
V_SUM_MAX = int(sum(Round_values))  # Maximum possible value sum
N_MAX = len(Values)
W_MAX = float('inf')  # Use infinity for large weights
dp = [[W_MAX for _ in range(N_MAX)] for _ in range(V_SUM_MAX + 1)]

# GENETIC ALGORITHM SPECIFIC PARAMETERS #
population_size = 200
generations = 250
mutation_rate = 0.012


### ALGORITHMS ###

# GREEDY APPROACH #
def Greedy_knapsack(Max_capacity, Weights, Values):

    df = pd.DataFrame({'values': Values, 'weights': Weights})

    df['ratios'] = df['values'] / df['weights']
    df_sorted = df.sort_values(by='ratios', ascending = False)

    cum_weight = 0 #hehe #stands for cumulative weight #OBS should remove before submission
    Selected_items = []
    for index, row in df_sorted.iterrows():
        if cum_weight+row['weights'] <= Max_capacity:
            Selected_items.append(index)
            cum_weight += row['weights']
        else:
            pass
    return Selected_items

# DYNAMIC APPROACH #
def dyn_knapsack(Weights, Values, Max_capacity):
    """
    Solves the 0/1 Knapsack problem using dynamic programming.

    Args:
        capacity: The maximum capacity of the knapsack.
        weights: A list of the weights of the items.
        values: A list of the values of the items.

    Returns:
        A tuple containing:
            - The maximum value that can be carried in the knapsack.
            - A list of the indices of the items included in the optimal solution.
    """

    n = len(Values)  # Number of items

    # Create a 2D table (n+1) x (capacity+1) to store results of subproblems
    dp = [[0 for _ in range(Max_capacity + 1)] for _ in range(n + 1)]

    # Build the dp table in bottom-up manner
    for i in range(n + 1):
        for w in range(Max_capacity + 1):
            if i == 0 or w == 0:
                dp[i][w] = 0
            elif Weights[i - 1] <= w:
                # Convert Weights[i-1] to int for indexing:
                dp[i][w] = max(
                    Values[i - 1] + dp[i - 1][w - int(Weights[i - 1])], dp[i - 1][w]
                )
            else:
                dp[i][w] = dp[i - 1][w]

    # Find the maximum value
    max_value = dp[n][Max_capacity]

    # Backtrack to find the selected items
    selected_items = []
    i = n
    w = Max_capacity
    while i > 0 and w > 0:
        if dp[i][w] != dp[i - 1][w]:
            selected_items.append(i - 1)  # Item indices are 0-based
            w -= int(Weights[i - 1]) # Convert here as well
        i -= 1

    return max_value, selected_items[::-1]  # Reverse to get the correct order

#REVERSE DYNAMIC PROGRAMMING
def knapsack_large_weights(weights, values, capacity):
    for i in range(N_MAX):
        dp[0][i] = 0

    for v in range(1, V_SUM_MAX + 1):
        for i in range(N_MAX):
            if i == 0:
                if v <= values[0]:
                    dp[v][0] = weights[0]
                else:
                    dp[v][0] = W_MAX
            else:
                if v <= values[i]:
                    dp[v][i] = min(weights[i], dp[v][i-1])
                else:
                    dp[v][i] = min(dp[v][i-1], weights[i] + dp[v-int(values[i])][i-1])
    # Find the maximum value
    max_value = 0
    for v in range(V_SUM_MAX, -1, -1):
        if dp[v][N_MAX - 1] <= capacity:
            max_value = v
            break
    
    # Backtracking to find selected items and used capacity
    v = max_value
    i = N_MAX - 1
    selected_items = []
    used_capacity = 0
    Tot_val = 0
    while v > 0 and i >= 0:
        if i > 0 and dp[v][i] != dp[v][i - 1]:
            selected_items.append(i)
            used_capacity += weights[i]
            Tot_val += values[i]
            v -= int(values[i])
        i -= 1
    Tot_val = Tot_val/1000 #return to original scale
    return Tot_val, selected_items, used_capacity


# GENETIC APPROACH #
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

# BRANCH AND BOUND #
def knapsack_branch_and_bound(values, weights, W):
    def calculate_bound(value, weight, index, values, weights, W):
        if weight >= W:
            return 0
        bound = value
        total_weight = weight
        for i in range(index, len(values)):
            if total_weight + weights[i] <= W:
                bound += values[i]
                total_weight += weights[i]
            else:
                remaining_capacity = W - total_weight
                bound += values[i] * (remaining_capacity / weights[i])
                break
        return bound

    n = len(values)
    max_value = 0
    best_items = []

    # Priority queue: (neg_value, weight, bound, index, items_included)
    queue = []
    heapq.heappush(queue, (-0, 0, 0, -1, []))

    while queue:
        neg_value, curr_weight, bound, index, included = heapq.heappop(queue)
        curr_value = -neg_value

        if curr_weight <= W and curr_value > max_value:
            max_value = curr_value
            best_items = included

        if index + 1 < n:
            # Include the next item
            next_index = index + 1
            new_weight = curr_weight + weights[next_index]
            new_value = curr_value + values[next_index]
            new_included = included + [next_index]

            if new_weight <= W:
                new_bound = calculate_bound(new_value, new_weight, next_index, values, weights, W)
                if new_bound > max_value:
                    heapq.heappush(queue, (-new_value, new_weight, new_bound, next_index, new_included))

            # Exclude the next item
            new_bound = calculate_bound(curr_value, curr_weight, next_index, values, weights, W)
            if new_bound > max_value:
                heapq.heappush(queue, (-curr_value, curr_weight, new_bound, next_index, included))

    return max_value, best_items

### CALCULATIONS OF RESULTS ###
st_dp = time.process_time_ns()
max_dp_val, selected_dp_items = dyn_knapsack(dp_weights, Values, Max_capacity_dp)
et_dp = time.process_time_ns()
dp_time = et_dp-st_dp

st_rdp = time.process_time_ns()
max_rdp_val, selected_rdp_items, used_capacity = knapsack_large_weights(Weights, Round_values, Max_capacity)
et_rdp = time.process_time_ns()
rdp_time = et_rdp-st_rdp

st_gr = time.process_time_ns()
gr_selected_items = Greedy_knapsack(Max_capacity, Weights, Values)
et_gr = time.process_time_ns()
gr_time = et_gr - st_gr 

st_gen = time.process_time_ns()
max_gen_val, selected_gen_items, used_capacity = genetic_knapsack(Weights, Values, Max_capacity, population_size, generations, mutation_rate)
et_gen = time.process_time_ns()
gen_time = et_gen - st_gen

st_BnB = time.process_time_ns()
max_BnB_val, selected_BnB_items = knapsack_branch_and_bound(Values, Weights, Max_capacity)
et_BnB = time.process_time_ns()
BnB_time = et_BnB - st_BnB

# Comparison of in common selected items #

#Note that these sets only count the selected items
#Thus we have that Iij = intersection of sets i and j is only those items
#selected in both sets.
# union of i j count every selected item in either sets, which gives that:
# Total objects -  the union is the items that were not selected by both sets.
# which means we want to calculate (Total - Union + Intersection)/T to get our percentage
#Intersections
I12 = len(list(set(selected_dp_items) & set(selected_rdp_items)))
I13 = len(list(set(selected_dp_items) & set(gr_selected_items)))
I14 = len(list(set(selected_dp_items) & set(selected_gen_items)))
I15 = len(list(set(selected_dp_items) & set(selected_BnB_items)))

I23 = len(list(set(selected_rdp_items) & set(gr_selected_items)))
I24 = len(list(set(selected_rdp_items) & set(selected_gen_items)))
I25 = len(list(set(selected_rdp_items) & set(selected_BnB_items)))

I34 = len(list(set(gr_selected_items) & set(selected_gen_items)))
I35 = len(list(set(gr_selected_items) & set(selected_BnB_items)))

I45 = len(list(set(selected_gen_items) & set(selected_BnB_items)))

#Unions

U12 = len(list(set(selected_dp_items).union(set(selected_rdp_items))))
U13 = len(list(set(selected_dp_items).union(set(gr_selected_items))))
U14 = len(list(set(selected_dp_items).union(set(selected_gen_items))))
U15 = len(list(set(selected_dp_items).union(set(selected_BnB_items))))

U23 = len(list(set(selected_rdp_items).union(set(gr_selected_items))))
U24 = len(list(set(selected_rdp_items).union(set(selected_gen_items))))
U25 = len(list(set(selected_rdp_items).union(set(selected_BnB_items))))

U34 = len(list(set(gr_selected_items).union(set(selected_gen_items))))
U35 = len(list(set(gr_selected_items).union(set(selected_BnB_items))))

U45 = len(list(set(selected_gen_items).union(set(selected_BnB_items))))

Total = len(Values)
#Percentage
C12 = (Total-U12+I12)/Total
C13 = (Total-U13+I13)/Total
C14 = (Total-U14+I14)/Total
C15 = (Total-U15+I15)/Total

C23 = (Total-U23+I23)/Total
C24 = (Total-U24+I24)/Total
C25 = (Total-U25+I25)/Total

C34 = (Total-U34+I34)/Total
C35 = (Total-U35+I35)/Total

C45 = (Total-U45+I45)/Total
C = {
    (1, 2): C12,
    (1, 3): C13,
    (1, 4): C14,
    (1, 5): C15,
    (2, 3): C23,
    (2, 4): C24,
    (2, 5): C25,
    (3, 4): C34,
    (3, 5): C35,
    (4, 5): C45
}

# Initialize matrix with 100% on the diagonal
n = 5  # number of entities (1 to 5)
matrix = pd.DataFrame(100.0, index=range(1, n+1), columns=range(1, n+1))

# Fill in C-values symmetrically
for (i, j), c_val in C.items():
    matrix.at[i, j] = c_val*100 #percentage
    matrix.at[j, i] = c_val*100  # Symmetry


### PRESENTING RESULTS ###

print('RESULTS OF DYNAMIC KNAPSACK')
print('VALUE: ', sum(Values[i] for i in selected_dp_items), '    Storage used: ', sum(Weights[i] for i in selected_dp_items))
print('Processing time: ', dp_time)

print('RESULTS OF REVERSE DYNAMIC KNAPSACK')
print('VALUE: ', sum(Values[i] for i in selected_rdp_items), '    Storage used: ', sum(Weights[i] for i in selected_rdp_items))
print('Processing time: ', rdp_time) 

print('RESULTS OF GREEDY KNAPSACK')
print('VALUE: ', sum(Values[i] for i in gr_selected_items),'    Storage used: ', sum(Weights[i] for i in gr_selected_items))
print('Processing time: ', gr_time)

print('RESULTS OF GENETIC KNAPSACK')
print('VALUE: ', max_gen_val, '    Storage used: ', sum(Weights[i] for i in selected_gen_items))
print('Processing time: ', gen_time)

print('RESULTS OF BRANCHH AND BOUND KNAPSACK')
print('VALUE: ', max_BnB_val, '    Storage used: ', sum(Weights[i] for i in selected_BnB_items))
print('Processing time: ', BnB_time)

# Display the matrix
print(matrix.round(2))

print('Total Value of All Items: ', sum(Values))
print('Total Weight of All Items: ', sum(Weights))
