import pandas as pd
import numpy as np

df = pd.read_excel('data/kidr_activity.xlsx', header=None)
df = df.drop([0,1,2]) #Drops the rows that are not part of the data.
df.columns = df.iloc[0].to_list() #Designates the relevant row as the header.
df = df[1:]
#print(df.head())
#print(df.columns)

var = ['SND number','Size','Days passed','#Canceled','Number of Requests', 'Access Granted*', 'Visits'] #relevant variables
#The size of the datasets are given in MB
df = df[var]
df = df.reset_index(drop=True)
columns_to_fill = ['Number of Requests', 'Access Granted*','#Canceled','Visits'] 
df[columns_to_fill] = df[columns_to_fill].fillna(0) #Filling in the empty rows as zeros
df['Size'] = df['Size'].astype('float') #Since the class of 'Size' is initially strings we convert it to float


#Constructing the hyperparameters
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
Values = ((A*B*C*(3/2*df['Access Granted*']+(df['Number of Requests']-1/2*df['#Canceled']))+(B*C*df['data downloads']+C*df['doc downloads'])+df['Visits']))/df['Days passed'].astype(float).tolist()
Max_capacity = int(0.5*1000*1000*1000) #Tot cap of KI is 70Tb
#We test different constructed max capacities to restrain the knapsack more

# DP SPECIFIC PARAMETERS #
dp_weights = [w/1000+1 for w in Weights] #makes sure that we won't have a zero when we round the weights 
dp_weights = [round(w) for w in dp_weights] #also converts the weights to Mb
Max_capacity_dp = int(Max_capacity/1000) #Converts max_capacity to Mb

#REVERSE DP SPECIFIC PARAMETERS #
Round_values = round(10000*Values) #Integer conversion of values, different powers of 10 gives varying specificity
V_SUM_MAX = int(sum(Round_values))  # Maximum possible value sum
N_MAX = len(Values)
W_MAX = float('inf')  # Use infinity for large weights
dp = [[W_MAX for _ in range(N_MAX)] for _ in range(V_SUM_MAX + 1)]

"""The disparity between the weights, i.e. that some are very big and some very small, makes this problem 
fundamentally unsuited for an dynamic programming approach. Since Scaling the weights such that there are no
decimals would require a unit size of 10 bytes. This becomes problematic when you have some datasets in the scale
of TB as well as a max capacity of several TB. Since it effectively requires a dp-table with one dimension having
a length of 100*1000^3. Requiring massive amount of memory to calculate 
"""

#Dynamic approach
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




print('THE DYNAMIC SOLUTION')
max_value, selected_items = dyn_knapsack(dp_weights, Values, Max_capacity_dp)
print("Maximum value:", max_value)
print("Selected items (indices):", selected_items)
"""
# Example to print item details:
print("Selected items:")
for index in selected_items:
    print(f"Item (Index {index}): Weight = {Weights[index]}, Value = {Values[index]}")
"""

"""An alternative that is still dynamic prgramming is to flip the table so to speak, and instead of asking 
what is the greatest value given a capacity, ask what is the minimum weight to achieve certain values, ranging 
from the absolute maximum possible value and down."""

## Reverse dynamic algorithm ##
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

print('THE REVERSE DYNAMIC SOLUTION')
Tot_val, dp_selected_items, used_capacity = knapsack_large_weights(Weights, Round_values, Max_capacity)

print("Selected items (indices):", np.sort(dp_selected_items))
print("Number of selected items:", len(dp_selected_items))
print("Used capacity:", used_capacity / 1000, "MB, out of:", Max_capacity/1000, "MB")  # Print used capacity in MB
print("Max value:", Tot_val)  # max value achieved
print("Total Value ", sum(Values[i] for i in dp_selected_items))
#for index in dp_selected_items:
 #   print(f"Item (Index {index}): Weight = {Weights[index] / 1000} MB, Value = {Values[index]}")


#Greedy approach
def Greedy_knapsack(Max_capacity, Weights, Values):

    df = pd.DataFrame({'values': Values, 'weights': Weights})

    df['ratios'] = df['values'] / df['weights']
    df_sorted = df.sort_values(by='ratios', ascending = False)

    cum_weight = 0 #OBS Cumulative weight...#...hehe
    Selected_items = []
    for index, row in df_sorted.iterrows():
        if cum_weight+row['weights'] <= Max_capacity:
            Selected_items.append(index)
            cum_weight += row['weights']
        else:
            pass
    return Selected_items

print('THE GREEDY SOLUTION')
gr_selected_items = Greedy_knapsack(Max_capacity, Weights, Values)
print("Selected item indices:", np.sort(gr_selected_items))
print("Amount of items: ", len(gr_selected_items))
print("Greedy alg total value: ", sum(Values[i] for i in gr_selected_items))
print("Greedy alg used capacity: ", sum(Weights[i] for i in gr_selected_items))

#To display the selected items and their values and weights.
"""
df = pd.DataFrame({'values': Values, 'weights': Weights})
df['ratios'] = df['values'] / df['weights']
for index in gr_selected_items:
    print(df.iloc[index])
""" 
