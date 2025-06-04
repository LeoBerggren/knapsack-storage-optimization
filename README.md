# knapsack-storage-optimization
Applying the knapsack and GAP problem to research data storage optimization, with data from Karolinska Institutet (KI)

A combination of different algorithms are tried and compared when tackling the 0/1 Knapsack Problem  with unique items as well as the Generalized Assignment Problem (GAP). These are for the Knapsack Problem: dynamic programming; dynamic programming reformulated as to use values instead of weights for the memory matrix; Greedy approach; Genetic approach and lastly Branch and Bound. 

For the GAP we implement only a Genetic approach as well as Branch and Bound.

There are a number of different code files for so that each algorithm may be tested and changed separately, but there is one file with all the algorithms designed for the knapsack implementation in one (KP_all.py) and one with all (both) algorithms for the generalized assignment problem implementation (GAP_all.py). They can be run as they are, as the data that is used is provided (kidr_activity.xlsx; nedladdningar.xlsx) and is loaded in the code. 

If one is interested in changing any of the parameters or hyperparameters  (such as value or max capacity) used in the knapsack and GAP, you may pay attention to the ###Parameter construction### section. Note that while if the code is run as is it won't give any output of which datasets the algorithms select, each algorithm has it as an output, so if one wishes to view them you can just print it.

* ett testset med testresultat för att se att man har lyckats köra din kod rätt och så man kan se formatet av input filerna
The following part presents what datasets the algorithms choose given the following specifics:
V = value of datasets in warm storage

**Knapsack**:
max capacity = 500GB

Genetic approach specific parameters:
Random seed = 438579088
Population = 200
Mutation rate = 0.012
Generations = 250


**GAP**:
warm storage capacity = 50GB
Lukewarm storage capacity = 900GB
Value of datasets in lukewarm storage = 0.5*V

Genetic approach specific parameters:
Random seed = 42
Population = 100
Mutation rate = 0.05
Generations = 300


**Applied to the knapsack problem**:
(OBS! The lists presented are a list of the datasets that the algorithms "place" in hot storage, i.e if the lists have an 28 as an entry, then that means dataset 28 is placed in hot storage)

Dynamic (weights as memory matrix): [0, 1, 4, 6, 7, 8, 9, 11, 13, 14, 15, 16, 21, 22, 26, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]

Dynamic (values as memory matrix): [ 1  4  6  7  8  9 11 13 14 15 16 21 22 26 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60]

Greedy algorithm:  [ 0  1  4  6  7  8  9 10 11 13 14 15 16 21 22 26 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60]

Genetic approach (GA): [1, 6, 7, 9, 11, 13, 15, 21, 22, 26, 29, 30, 31, 32, 33, 35, 38, 39, 40, 41, 43, 44, 45, 47, 48, 49, 50, 56, 57, 59, 60]

Branch and Bound (BnB): [0, 1, 4, 6, 7, 8, 9, 11, 13, 14, 15, 16, 21, 22, 26, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]

**Applied to the Generalized Assignment Problem (GAP)**:
(OBS! the output of the GAP algorithms differ from the KP ones. Here the index of an entry indicates which dataset it is (i.e. dataset 0 is the first entry in the list and so on, its storage is indicated by the number, 1=lukewarm storage, 0=warm storage, -1= cold storage)

Genetic approach: [0, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, -1, 1, -1, -1, 0, 0, 1, -1, -1, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

Branch and Bound: [-1, -1, -1, 0, 0, -1, 0, -1, -1, 0, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, 1, -1, 0, -1, -1, -1, -1, 0, 0, -1, -1, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, -1]







