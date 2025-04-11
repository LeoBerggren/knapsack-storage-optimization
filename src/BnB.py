import heapq
import pandas as pd
import numpy as np
import random

# Seed for reproducibility
random.seed(42)

df = pd.read_excel('data/kidr_activity.xlsx', header=None)
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
Weights = (df['Size']*1000).tolist() #converts from MB to KB
#print(Weights)
Values = (df['Access Granted*']+1/2*(df['Number of Requests']-1/2*df['#Canceled'])+1)/df['Days passed'].astype(float).tolist()
#print(Values)
#print(sum(Weights))
Max_capacity = int(0.1*1000*1000*1000) #Max capacity is 70TB = tot cap of KI 
#We test different constructed max capacities to restrain the knapsack more




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

"""
n = 10
values = [random.randint(10, 100) for _ in range(n)]
weights = [random.randint(5, 50) for _ in range(n)]
W = 100
"""

#print("Values of items:", Values)
#print("Weights of items:", Weights)
print("Knapsack capacity:", Max_capacity)
print("Total Value: ", sum(Values))
print("Total Weight: ", sum(Weights))

max_value, selected_indices = knapsack_branch_and_bound(Values, Weights, Max_capacity)
print("Maximum value in knapsack =", max_value)
print("Used capacity: ", [sum(Weights[i] for i in selected_indices)])
print("Selected item indices =", selected_indices)
#print("Selected values =", [Values[i] for i in selected_indices])
#print("Selected weights =", [Weights[i] for i in selected_indices])