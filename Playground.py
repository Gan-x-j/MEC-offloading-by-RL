from Environment import EdgeEnv
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# learning_parameters
actions = [x for x in range(4, 17) if x % 4 == 0]
episodes = 50
Q = defaultdict(lambda: np.zeros(len(actions)))
alpha = 0.1
epsilon = 0.1
time_of_a_day = 1000

# simulate_parameters
server_cpu_speed = 4e9
avg_job_cycles = 2e9

user_number = 100
user_job_arrive_prob = 0.2
user_job_arrive_rate = 1
user_cpu_speed = 1e8
user_time_weight = 0.9
user_energy_weight = 0.1


def greedy_policy(state):
    if np.random.rand() > epsilon:
        return np.argmax(Q[state])
    else:
        return np.random.randint(0, len(Q[state]))


env = EdgeEnv(user_number, server_cpu_speed, user_job_arrive_rate, user_cpu_speed,
              avg_job_cycles, user_time_weight, user_energy_weight, user_job_arrive_prob)
rewards = []

for episode in range(episodes):
    cur_state = env.reset()
    day_reward = []
    for i in range(time_of_a_day):
        cur_action = greedy_policy(cur_state)
        cur_price = actions[cur_action]
        new_state, reward = env.step(cur_price)
        day_reward.append(reward)
        Q[cur_state][cur_action] += alpha * (reward + 0.9 * Q[new_state].max() - Q[cur_state][cur_action])
        if Q.__len__() > 50:
            break
        cur_state = new_state
    rewards.append(day_reward)

ans = sorted(Q.items())
for l in ans:
    print(str(l[0]) + " : " + str(actions[np.argmax(l[1])]))
