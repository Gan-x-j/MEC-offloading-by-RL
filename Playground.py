from Environment import EdgeEnv
from collections import defaultdict
from functools import cmp_to_key
import numpy as np
import matplotlib.pyplot as plt


# learning_parameters
actions = [x for x in range(4, 100) if x % 20 == 4]
episodes = 300
Q = defaultdict(lambda: np.zeros(len(actions)))
alpha = 0.1
epsilon = 0.1
time_of_a_day = 1000
discount_factor = 0.5

# simulate_parameters
server_cpu_speed = 4e9
avg_job_cycles = 2e9

user_number = 100
user_job_arrive_prob = 0.3
user_job_arrive_rate = 1
user_cpu_speed = 2e8
user_time_weight = 0.9
user_energy_weight = 0.1
change_points = (0, int(time_of_a_day * 0.2), int(time_of_a_day * 0.7))
user_pos_distribution = [50, 20, 80]
user_job_arrive_probs = [0.1, 0.5, 1]
user_pos_prob_distribution = [
    [0.9, 0.1, 0],
    [0.2, 0.7, 0.1],
    [0.4, 0.3, 0.3],
]


def greedy_policy(state):
    if np.random.rand() > epsilon:
        return np.argmax(Q[state])
    else:
        return np.random.randint(0, len(Q[state]))


env = EdgeEnv(user_number, server_cpu_speed, user_job_arrive_rate, user_cpu_speed,
              avg_job_cycles, user_time_weight, user_energy_weight, user_job_arrive_prob)
rewards = []

for episode in range(episodes):
    day_reward = []
    for i in range(time_of_a_day):
        for j in range(len(change_points)):
            if change_points[j] == i:
                cur_state = env.reset(j, user_pos_distribution, user_pos_prob_distribution[j], user_job_arrive_probs[j])
        cur_action = greedy_policy(cur_state)
        cur_price = actions[cur_action]
        new_state, reward = env.step(cur_state[0], cur_price)
        day_reward.append(reward)
        Q[cur_state][cur_action] += alpha * (reward + discount_factor * Q[new_state].max() - Q[cur_state][cur_action])
        cur_state = new_state
    rewards.append(day_reward)

for k in list(Q.keys()):
    if 0 in Q[k]:
        del Q[k]


def compare_to(a, b):
    if a[0] > b[0]:
        return 1
    elif a[0] < b[0]:
        return -1
    else:
        if a[1] > b[1]:
            return 1
        elif a[1] < b[1]:
            return -1
        else:
            return 0


ans = sorted(Q.items(), key=cmp_to_key(compare_to))
for l in ans:
    print(str(l[0]) + " : " + str(actions[np.argmax(l[1])]))

means = []
for i in range(len(rewards)):
    means.append(np.mean(rewards[i]))

plt.plot(means)
plt.show()
