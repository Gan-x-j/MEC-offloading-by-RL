import numpy as np


class User:
    def __init__(self, rate, position, server_cpu_speed):
        self.trans_power = 100
        self.SNR = -40
        self.cpu_speed = 5e8
        self.path_loss_exp = 3.5
        self.avg_job_cycles = 2.5e10
        self.SERVER_CPU_SPEED = server_cpu_speed

        self.job_arrive_rate = rate
        self.data_offload_rate = 100
        self.position = position
        self.time_weight = 0.9
        self.energy_weight = 0.1
        self.buffer_using = 0
        self.job_arrive_prob = 1
        self.trans_speed = 1 * 360/8 * np.log2(1 + (self.trans_power * np.power(self.position, -1 * self.path_loss_exp)) / np.power(10, self.SNR / 10))

    def get_load(self):
        return self.avg_job_cycles * self.job_arrive_rate

    def local_power(self):
        return 1e-27 * self.cpu_speed**2 * self.avg_job_cycles

    def local_time(self):
        return self.avg_job_cycles / self.cpu_speed + self.buffer_using / self.cpu_speed

    def edge_time(self, delay):
        return (self.avg_job_cycles / self.SERVER_CPU_SPEED) + (self.data_offload_rate / self.trans_speed) + delay

    def edge_power(self):
        return self.trans_power / 1000 * self.data_offload_rate / self.trans_speed

    def make_offload_decision(self, price, delay):
        if self.buffer_using - 10 * self.cpu_speed > 0:
            self.buffer_using -= 10 * self.cpu_speed
        else:
            self.buffer_using = 0

        if np.random.rand() > self.job_arrive_prob:
            return 'NO_TASK'
        else:
            if self.get_user_profit(price, delay) > 0:
                return 'EDGE'
            else:
                self.buffer_using += self.get_load()
                return 'LOCAL'

    def get_user_profit(self, price, delay):
        local_power = self.local_power()
        local_time = self.local_time()
        edge_time = self.edge_time(delay)
        edge_power = self.edge_power()
        user_profit = self.time_weight * (local_time - edge_time) + self.energy_weight * (local_power - edge_power) - price
        return user_profit

    def get_sum_cost(self, delay, judge):
        if judge == 'LOCAL':
            return -1 * (self.time_weight * (self.local_time()) + self.energy_weight * (self.local_power()))
        elif judge == 'EDGE':
            return -1 * (self.time_weight * (self.edge_time(delay)) + self.energy_weight * (self.edge_power()))