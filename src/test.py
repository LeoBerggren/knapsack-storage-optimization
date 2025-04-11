import itertools
import numpy as np
import pandas as pd
import random
import heapq


# Small test case data: 4 tasks and 3 agents
values = [
    [60, 100, 120, 80],  # Values of tasks for Agent 0
    [50, 90, 110, 70],   # Values of tasks for Agent 1
    [40, 80, 100, 60]    # Values of tasks for Agent 2
]

weights = [
    [10, 20, 30, 40],    # Weights of tasks for Agent 0
    [12, 22, 32, 42],    # Weights of tasks for Agent 1
    [15, 25, 35, 45]     # Weights of tasks for Agent 2
]

capacities = [50, 100, 150]  # Capacities of agents
num_agents = 3
num_tasks = 4

# Brute Force optimal solution for GAP (check all combinations of assignments)
def brute_force_gap(values, weights, capacities):
    n = len(values[0])
    best_value = 0
    best_assignment = []
    
    # Generate all possible assignments of tasks to agents
    all_assignments = list(itertools.product(range(num_agents), repeat=n))
    
    for assignment in all_assignments:
        agent_loads = [0] * num_agents
        total_value = 0
        
        # Calculate the total value and check if the assignment is valid
        valid_assignment = True
        for task_idx, agent in enumerate(assignment):
            agent_loads[agent] += weights[agent][task_idx]
            if agent_loads[agent] > capacities[agent]:  # Exceeds capacity
                valid_assignment = False
                break
            total_value += values[agent][task_idx]
        
        if valid_assignment and total_value > best_value:
            best_value = total_value
            best_assignment = assignment
    
    return best_value, best_assignment

# Run brute force to get the optimal solution
optimal_value, optimal_assignment = brute_force_gap(values, weights, capacities)

print(f"Optimal Value: {optimal_value}")
print(f"Optimal Assignment (task -> agent): {optimal_assignment}")

# Now, we assume the current Branch and Bound algorithm is implemented for GAP
# Let's use your existing Branch and Bound function for comparison

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
    max_iterations = 100000
    iteration_count = 0

    while heap:
        iteration_count += 1
        if iteration_count > max_iterations:
            print("🔴 Max iteration limit reached.")
            break

        neg_bound, total_value, task_idx, agent_loads, assignment = heapq.heappop(heap)
        print('Heap: ',heap)
        # Debugging: print the current task and assignment being processed
        print(f"Task {task_idx} | Assignment: {assignment} | Total Value: {total_value} | Bound: {-neg_bound}")

        if task_idx >= num_tasks:
            if total_value > best_value:
                print(f"✅ New best: value={total_value:.4f}, iterations={iteration_count}, Assignment = {assignment}")
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
                    #best_ratio = 0
                    best_assigned_value = 0
                    for a in range(num_agents):
                        if weights[a][next_task] > 0 and new_agent_loads[a] + weights[a][next_task] <= capacities[a]:
                            ratio = values[a][next_task] / weights[a][next_task]
                            best_assigned_value = max(best_assigned_value,  values[a][next_task])
                            #best_ratio = max(best_ratio, ratio)
                    est_bound += best_assigned_value
                    # est_bound += best_ratio * 1  # Assume we can pick the best agent for this task
                if est_bound > best_value:
                    heapq.heappush(heap, (-est_bound, new_value, task_idx + 1, new_agent_loads, new_assignment))
                else:
                    #print(f"✘ Pruned (low bound): task {task} -> agent {agent}, est_bound={est_bound:.4f}, current best={best_value:.4f}")
                    continue

            else:
                #print(f"✘ Pruned (capacity): task {task} -> agent {agent}, load={agent_loads[agent]:,.2f}, task_weight={task_weight:,.2f}, cap={capacities[agent]:,.2f}")
                continue
        # Do not allow skipping tasks if a valid assignment is possible
        est_bound = total_value
        for next_task_idx in range(task_idx + 1, num_tasks):
            next_task = sorted_indices[next_task_idx]
            best_assigned_value = 0
            for a in range(num_agents):
                if weights[a][next_task] > 0 and agent_loads[a] + weights[a][next_task] <= capacities[a]:
                    best_assigned_value = max(best_assigned_value, values[a][next_task])
            est_bound += best_assigned_value  # Add the best possible value from future tasks
        if est_bound > best_value:
            heapq.heappush(heap, (-est_bound, total_value, task_idx + 1, agent_loads.copy(), assignment))

    return best_value, best_assignment

# Run Branch and Bound for GAP
branch_and_bound_value = gap_branch_and_bound(values, weights, capacities)

print(f"Branch and Bound Value: {branch_and_bound_value}")

#################################

