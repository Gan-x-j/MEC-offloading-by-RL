from Environment import EdgeEnv
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

actions = [x for x in range(10, 51) if x % 10 == 0]
episodes = 20
Q = defaultdict(lambda: np.zeros(len(actions)))
alpha = 0.9
epsilon = 0.1
tens_of_a_day = 5000


def greedy_policy(state):
    if np.random.rand() > epsilon:
        return np.argmax(Q[state])
    else:
        return np.random.randint(0, len(Q[state]))


def exploit_policy(state):
    return np.argmax(Q[state])


env = EdgeEnv(200)
rewards = []

for episode in range(episodes):
    cur_state = env.reset()
    for s in range(tens_of_a_day):
        cur_action = greedy_policy(cur_state)
        cur_price = actions[cur_action]
        new_state, reward, _ = env.step(cur_price)
        Q[cur_state][cur_action] += alpha * (reward + 0 * Q[new_state].max() - Q[cur_state][cur_action])
        cur_state = new_state

ans = sorted(Q.items())
for l in ans:
    print(str(l[0]) + " : " + str(actions[np.argmax(l[1])]))

# plt.plot(rewards)
# plt.show()