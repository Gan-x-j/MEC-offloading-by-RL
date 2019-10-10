from User import User
import numpy as np


class EdgeEnv:
    def __init__(self, num_user, server_cpu_speed, job_arrive_rate, user_cpu_speed,
                 avg_job_cycles, time_weight, energy_weight, job_arrive_prob):
        self.num_user = num_user
        self.SERVER_CPU_SPEED = server_cpu_speed
        self.avg_job_cycles = avg_job_cycles

        self.user_job_arrive_rate = job_arrive_rate
        self.user_cpu_speed = user_cpu_speed
        self.user_time_weight = time_weight
        self.user_energy_weight = energy_weight
        self.user_job_arrive_prob = job_arrive_prob

        self.MEC_job_finish_rate = self.SERVER_CPU_SPEED / self.avg_job_cycles
        self.buffer_using = 0
        self.users = []

        self.reset()

    def reset(self):
        self.users.clear()
        for i in range(self.num_user):
            new_user = User(self.user_cpu_speed, self.avg_job_cycles, self.SERVER_CPU_SPEED,
                            self.user_job_arrive_prob, self.user_job_arrive_rate, np.random.randint(10, 101),
                            self.user_time_weight, self.user_energy_weight)
            self.users.append(new_user)
        self.buffer_using = 0
        return self.buffer_using

    def step(self, new_price):
        terminal = False
        if self.buffer_using - 10 * self.SERVER_CPU_SPEED > 0:
            self.buffer_using -= 10 * self.SERVER_CPU_SPEED
        else:
            self.buffer_using = 0
        reward = 0
        pred_delay = 1 / (2 * self.MEC_job_finish_rate) + np.sqrt(new_price / (0.9 * self.MEC_job_finish_rate) + 1 / (4 * np.power(self.MEC_job_finish_rate, 2)))
        for i in range(0, self.num_user):
            true_delay = self.buffer_using / self.SERVER_CPU_SPEED
            cur_user_decision = self.users[i].make_offload_decision(new_price, pred_delay)
            if cur_user_decision == 'EDGE':
                self.buffer_using += self.users[i].get_load()
                reward += self.users[i].get_sum_cost(true_delay, 'EDGE')
            elif cur_user_decision == 'LOCAL':
                reward += self.users[i].get_sum_cost(true_delay, 'LOCAL')
            elif cur_user_decision == 'NO_TASK':
                continue
        return self.buffer_using, reward
