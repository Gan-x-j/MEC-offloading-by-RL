from User import User
import numpy as np


class EdgeEnv:
    def __init__(self, num_user):
        self.num_user = num_user
        self.buffer_using = 0
        self.user_job_arrive_rate = 0.002
        self.SERVER_CPU_SPEED = 3e9
        self.MEC_job_finish_rate = 3e9 / 2.5e10
        self.users = []
        self.reset()

    def reset(self):
        self.users.clear()
        for i in range(self.num_user):
            new_user = User(self.user_job_arrive_rate, np.random.randint(10, 201), self.SERVER_CPU_SPEED)
            self.users.append(new_user)
        self.buffer_using = 0
        return self.buffer_using

    def step(self, new_price):
        if self.buffer_using - 10 * self.SERVER_CPU_SPEED > 0:
            self.buffer_using -= 10 * self.SERVER_CPU_SPEED
        else:
            self.buffer_using = 0
        reward = 0
        cur_delay = 1 / (2 * self.MEC_job_finish_rate) + np.sqrt(new_price / (0.9 * self.MEC_job_finish_rate) + 1 / (4 * np.power(self.MEC_job_finish_rate, 2)))
        for i in range(0, self.num_user):
            true_delay = 100 * self.buffer_using / self.SERVER_CPU_SPEED
            cur_user_decision = self.users[i].make_offload_decision(new_price, true_delay)
            if cur_user_decision == 'EDGE':
                self.buffer_using += self.users[i].get_load()
                reward += self.users[i].get_sum_cost(true_delay, 'EDGE')
            elif cur_user_decision == 'LOCAL':
                reward += self.users[i].get_sum_cost(true_delay, 'LOCAL')
            elif cur_user_decision == 'NO_TASK':
                continue
        return self.buffer_using, reward, true_delay

    def update_user_job_arrive_rate(self, new_arrive_rate):
        self.user_job_arrive_rate = new_arrive_rate

    def user_distribution(self, time):
        if 9 <= time < 17:
            return (0.8, 0.1, 0.1)
        if 17 < time <= 21:
            return (0.2, 0.6, 0.2)
        return (0.2, 0, 0.8)
