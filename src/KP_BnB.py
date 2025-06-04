import heapq
import pandas as pd
import numpy as np

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

#Constructing the hyperparameters
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
#print(Values)
#print(sum(Weights))
Max_capacity = int(1*1000*1000*1000)



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

    # Sort items by value-to-weight ratio
    value_weight_ratio = [v / w if w > 0 else 0 for v, w in zip(values, weights)]
    sorted_items = sorted(enumerate(value_weight_ratio), key=lambda x: x[1], reverse=True)
    sorted_indices = [i for i, _ in sorted_items]

    values_sorted = [values[i] for i in sorted_indices]
    weights_sorted = [weights[i] for i in sorted_indices]

    n = len(values)
    max_value = 0
    best_items = []

    # Priority queue: (neg_value, weight, bound, index, items_included)
    queue = []
    heapq.heappush(queue, (-0, 0, 0, -1, []))

    visited = {} #Establishes a dictionary of visited states

    iteration = 0
    while queue:
        iteration += 1
        if iteration % 10000 == 0:
            print(f"⏳ Iteration {iteration:,}, Queue size: {len(queue)}, Best value so far: {max_value:.2f}")

        neg_value, curr_weight, bound, index, included = heapq.heappop(queue)
        curr_value = -neg_value

        # Prune by state revisiting
        key_weight = int(curr_weight // 1)  # round to whole numbers
        state_key = (index, key_weight)
        if state_key in visited and visited[state_key] >= curr_value:
            continue
        visited[state_key] = curr_value

        if curr_weight <= W and curr_value >= max_value:
            max_value = curr_value
            best_items = included


        if index + 1 < n:
            next_index = index + 1
            
            # Include the next item
            new_weight = curr_weight + weights_sorted[next_index]
            new_value = curr_value + values_sorted[next_index]
            new_included = included + [next_index]

            if new_weight <= W:
                new_bound = calculate_bound(new_value, new_weight, next_index, values_sorted, weights_sorted, W)
                if new_bound > max_value:
                    heapq.heappush(queue, (-new_value, new_weight, new_bound, next_index, new_included))

            # Exclude the next item
            new_bound = calculate_bound(curr_value, curr_weight, next_index, values_sorted, weights_sorted, W)
            if new_bound > max_value:
                heapq.heappush(queue, (-curr_value, curr_weight, new_bound, next_index, included))
   
    # Map back to original indices
    best_items_original = sorted([sorted_indices[i] for i in best_items])

    return max_value, best_items_original


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