import heapq
import random
import pandas as pd
import numpy as np
import time

# Sets random seed
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
#Constructing the hyperparameters #
Weights = (df['Size']*1000).tolist() #converts from MB to KB
Values = ((A*B*C*(3/2*df['Access Granted*']+(df['Number of Requests']-1/2*df['#Canceled']))+(B*C*df['data downloads']+C*df['doc downloads'])+df['Visits']))/df['Days passed'].astype(float).tolist()
Max_capacity = int(0.5*1000*1000*1000) #This is 500GB, different values might be appropriate

def gap_branch_and_bound(values, weights, capacities):
    """
    Solves the generalized assignment problem using a branch and bound algorithm. Instead of the classical example of agents choosing jobs, we view it as objects in a multiple knapsacks.

    Args:
        - values: the values of the objects, depending on both the object and which knapsack it is put in
        - weights: the weights of the objects, depending on both the object and which knapsack it is put in
        - capacities: the maximum capacities of the different knapsacks.

    Returns:
        The best value achieved from the assignment of objects to the knapsacks.
        A tuple of the assignments of the objects to the different knapsacks.
    """

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
                eps = 2/3 #decides the fineness of our search
                if est_bound > eps*(best_value): 
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



Capacities = [0.1*Max_capacity, 1.8*Max_capacity]
Multi_values = [1*Values, [0.5 * v for v in Values]]
Multi_weights = [Weights, Weights] #In our case I don't think there is any need to change both capacity and weights. 
#They stand for different things, but since we assume that it will affect it linearly, the change is already applied to capacity
#Alternatively we can think that from a budget perspective it is cheaper to buy the storage(capacity) but it
#is still a storage perspective rather than cost.

st = time.process_time_ns()
Max_val, Assignment = gap_branch_and_bound(Multi_values, Multi_weights, Capacities)
et = time.process_time_ns()
print('Time: ', et-st)
print("Max total value:", Max_val)
print("Task assignments (task -> agent):", Assignment)
print(type(Assignment))

Value = 0
Ws = 0
Ls = 0
for ind, val in enumerate(Assignment):
    if val == 1: #Lukewarm storage
        Value += 0.1*Values[ind]
        Ls += Weights[ind]
    elif val == 0: #Warm storage
        Value += Values[ind]
        Ws += Weights[ind]

print('Value double check: ', Value)
print('Weights in warm storage: ', Ws)
print('Weights in Lukewarm storage: ', Ls)

