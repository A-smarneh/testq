import simpy
import numpy as np
from mealpy.physics_based import MVO, SA
from scipy.stats import expon
import matplotlib.pyplot as plt
import random
from mealpy import TransferBinaryVar, TWO, EO, CircleSA
import pandas as pd

# this run started at 13:20
SIM_TIME = 100
SERVER_PROCESSING_CAPABILITIES = 15  # in GHZ= billion clock cycle per second
BANDWIDTH = 0.1  # in Gbps
env = simpy.Environment()
EXECUTION_MODE = 2  # 0: All local execution, 1: all offloading, 2:optimized

'''
Problem:
we dont send tasks to the queue, we generate them one by one that is why when only one user the queue is empty
Notes:
- queueing delay is not utilized now, but could be used later on to obtain additional factor to the optimization
algorithm. 
- issue time was not used too
To-DO:
- update user creation to make it auto
5. implement server stats
what about using a priority queue instead, with the priority set to issue time so that older tasks gets executed first 
https://stackoverflow.com/questions/54638792/how-can-i-change-the-priority-of-a-resource-request-in-simpy-after-yielding-the
or this approach
- update the avg delay post deadline to include only task that are delayed post deadline only 
ES plan:
-change the resource to a priority queue
if i cant cancel the request outside the ue lets create it inside the ES

Important Note:
- when calculating the delays in for the solutions we are not considering the actual queueing delays. 
- in real life, the MEC server doesn't dedicate all of its resources to one task, instead, each task gets a portion of 
the resources. Therefore, we might want to consider using 5 GHZ instead of 20 since the processor can server 4
 - what about making the penalty a percentage of the delay beyond the deadline
Future Considerations:
- Idle power consumption
- Update power consumption model to count for the distance
- do we have to discard delayed tasks or just execute them
- confidence interval 

*******************************************
 Evaluation Metrics:
 - Input size effect on offloading decision 
 -
'''


class Task:
    def __init__(self):
        """
        CPU_cycles: the CPU cycles required to complete the task in Giga cycle
        input_size The size of the input data required for the task in gigabit
        deadline: the deadline by which the output of the task must be transmitted back to UE
        issue_time: the time at which the task was issued.
        """
        self.CPU_cycles = None
        self.deadline = None
        self.input_size = None
        self.issue_time = None
        self.queueing_delay = None
        self.priority = None

    def create_task(self, ue_capability):
        """
        This method create a task with CPU_cycles following a Gamma distribution and with
        :return:
        """
        sanity_check = False
        while not sanity_check:
            self.deadline = env.now + random.uniform(0.01, 0.3)
            self.input_size = random.uniform(0.01, 0.1)
            self.CPU_cycles = random.uniform(0.15, 3)
            # print(self.deadline, self.get_input_size(), self.get_task_cpu_cycles())
            sanity_check = self.task_sanity_check(ue_capability)
            # print("task sanity", sanity_check)

    def get_deadline(self):
        return self.deadline

    def task_sanity_check(self, ue_capability):
        local_exec_deadline = env.now + (self.get_task_cpu_cycles() / ue_capability)
        offloading_exec_delay = env.now + (self.get_task_cpu_cycles() / SERVER_PROCESSING_CAPABILITIES) + \
                                (self.get_input_size() / BANDWIDTH)

        if self.get_deadline() > local_exec_deadline or self.get_deadline() > offloading_exec_delay:
            return True
        else:
            return False

    def get_input_size(self):
        return self.input_size

    def set_queueing_delay(self, delay):
        self.queueing_delay = delay

    def get_queueing_delay(self):
        return self.queueing_delay

    def get_task_issue_time(self):
        return self.issue_time

    def set_issue_time(self, issue_time):
        self.issue_time = issue_time

    def get_priority(self):
        return self.priority

    def set_priority(self, priority):
        self.priority = priority

    def set_task_cpu_cycles(self, cpu_cycles):
        self.CPU_cycles = cpu_cycles

    def get_task_cpu_cycles(self):
        return self.CPU_cycles


class ES:

    def __init__(self, processing_capabilities):
        """
        :param processing_capabilities: Array  that defines the processing capability of each resource the ES has
        """
        self.processor = simpy.PriorityResource(env, capacity=1)
        self.processing_capabilities = processing_capabilities  # this will be used to calculate
        # how long the task will take
        self.total_tasks_executed = 0
        self.total_discarded_tasks = 0  # this counts the discarded tasks from all users
        self.total_execution_delays = list()
        self.total_queuing_delays = list()
        self.x_pos = None
        self.y_pos = None

    def add_to_execution_delay(self, delay):
        self.total_execution_delays.append(delay)

    def add_to_queuing_delays(self, delay):
        self.total_queuing_delays.append(delay)

    def set_position(self, x, y):
        self.x_pos = x
        self.y_pos = y

    def get_avg_delay_q(self):
        avg_sum = sum(self.total_delays)
        return avg_sum / len(self.total_delays)

    def update_executed_tasks(self):
        self.total_tasks_executed = self.total_tasks_executed + 1

    def get_total_tasks_executed(self):
        return self.total_tasks_executed

    def status(self):
        print("Statistics")
        print("------------------------------------")
        print(f' Queued events:  {len(self.processor.put_queue)}')
        print(f' Total served tasks: {self.total_tasks_executed}')
        print(f' Average delay: {self.get_avg_delay_q()}')


class UE:

    def __init__(self, name, poisson_mean, processing_energy_consumption_per_second,
                 transmission_energy_consumption_per_second, processing_capability,
                 ):
        """
        :param name: unique identifier of the UE
        :param processing_capability The processing capability of the UE in GHZ
        """
        self.processor = simpy.PriorityResource(env, capacity=1)
        self.name = str(name)
        self.generated_tasks_counter = 0
        self.poisson_mean = poisson_mean
        self.x_pos = None
        self.y_pos = None
        self.transmission_energy_consumption = 0
        self.loca_processing_energy_consumption = 0
        self.processing_energy_consumption_per_second = processing_energy_consumption_per_second
        self.processing_capability = processing_capability
        self.transmission_energy_consumption_per_second = transmission_energy_consumption_per_second
        self.locally_executed_subtasks = 0
        self.offloaded_task_counter = 0  # executed only
        self.total_idle_time = 0
        self.total_transmission_time = 0
        self.total_local_processing_time = 0
        self.user_load_stat = list()
        self.local_execution_delays = list()
        self.offloading_execution_delays = list()  # This list store the queueing delays for remote execution
        self.complete_task_list = list()
        self.current_application_tasks_list = list()
        self.total_energy_consumption = list()
        self.total_execution_delays = list()
        self.avg_delay_post_deadline = None
        self.qualified_tasks_ratio = 0  # this is the ratio of qualified tasks

    def export_ue_stat(self):
        self.export_energy_consumption()
        df = pd.DataFrame(self.user_load_stat, columns=['time', 'CPU Cycles(Giga Cycles)', 'Input data size(Gigabit)'])
        df2 = pd.DataFrame(self.total_execution_delays, columns=['execution time', 'deadline', 'diff'])
        df.to_excel('Users Loads' + self.name + '.xlsx', sheet_name=self.name)
        df2.to_excel('time' + self.name + '.xlsx', sheet_name=self.name)

    def print_stat(self):
        self.calculate_avg_delay_post_deadline()
        print("EU ", self.name, "executed ", str(self.locally_executed_subtasks))
        print("server executed : ", self.get_offloaded_task_counter())
        print(f' Queued events:  {len(self.processor.put_queue)}')
        print(f' Total Energy Consumption: {self.sum_total_energy_consumption()}')
        print(f' Average delay post deadline: {self.avg_delay_post_deadline}')
        print(f' Qualified tasks ratio: {self.qualified_tasks_ratio}')
        print("----------------------------------------------------------")

    def calculate_avg_delay_post_deadline(self):
        total = 0
        qualified_tasks = 0
        for r in self.total_execution_delays:

            if r[2] < 0:
                qualified_tasks = qualified_tasks + 1
            else:
                total = abs(r[2]) + total
        if qualified_tasks != len(self.total_execution_delays):
            self.avg_delay_post_deadline = total / (len(self.total_execution_delays) - qualified_tasks)
        else:
            self.avg_delay_post_deadline = total
        self.qualified_tasks_ratio = qualified_tasks / len(self.total_execution_delays)

    def get_avg_delay_post_deadline(self):
        self.calculate_avg_delay_post_deadline()
        return self.avg_delay_post_deadline

    def get_qualified_tasks(self):
        self.calculate_avg_delay_post_deadline()
        return self.qualified_tasks_ratio

    def sum_total_energy_consumption(self):
        energy_consumption = 0
        for point in self.total_energy_consumption:
            energy_consumption = energy_consumption + point[1]
        return energy_consumption

    def set_coordinates(self, x, y):
        self.x_pos = x
        self.y_pos = y

    def update_locally_executed_tasks(self):
        self.locally_executed_subtasks = self.locally_executed_subtasks + 1

    def get_coordinates(self):
        return self.x_pos, self.y_pos

    def update_offloaded_tasks_counter(self):
        self.offloaded_task_counter = self.offloaded_task_counter + 1

    def get_offloaded_task_counter(self):
        return self.offloaded_task_counter

    def get_name(self):
        return self.name

    def update_generated_tasks(self, tasks):
        self.generated_tasks_counter = self.generated_tasks_counter + tasks

    def get_poisson_mean(self):
        return self.poisson_mean

    def update_energy_consumption(self, time, offload):
        # this is based on the execution type
        if offload:
            total_energy_consumption = self.transmission_energy_consumption_per_second * time
        else:
            total_energy_consumption = self.processing_energy_consumption_per_second * time
        self.total_energy_consumption.append((env.now, total_energy_consumption))

    def export_energy_consumption(self):
        i = 0
        slot = int(len(self.total_energy_consumption) / 50)
        energy_sum = 0
        new_list = list()
        for point in self.total_energy_consumption:
            if i < slot:
                energy_sum = energy_sum + point[1]
                i = i + 1
            elif i == slot:
                energy_sum = energy_sum + point[1]
                i = 0
                new_list.append((point[0], energy_sum))

        df = pd.DataFrame(new_list, columns=['time', 'Energy'])
        df.to_excel('Energy Consumption' + self.name + '.xlsx', sheet_name=self.name)

    def update_local_processing_time(self, time):
        self.total_local_processing_time = self.total_local_processing_time + time

    def process_task_locally(self, tasks):

        for task in tasks:
            env.process(self.generate_local_task(task))

    def generate_offloaded_tasks(self, task):
        start_waiting = env.now
        # transmit the data
        # print("before transmission:", task.get_deadline() - env.now)
        transmission_time = task.get_input_size() / BANDWIDTH
        yield env.timeout(transmission_time)
        priority = task.get_deadline() - env.now
        task.set_priority(priority)
        # print("p after transmission ", priority)
        # es1.optimized_execution()
        print("server queue: ", len(es1.processor.put_queue))
        with es1.processor.request(priority=priority) as req:
            req.obj = task
            task_processing_time = req.obj.get_task_cpu_cycles() / SERVER_PROCESSING_CAPABILITIES
            end_waiting = env.now
            waiting_time = end_waiting - start_waiting
            self.offloading_execution_delays.append(waiting_time)  # update delays total delays and offloading
            yield env.timeout(task_processing_time)
            self.total_execution_delays.append((env.now, task.deadline, env.now - task.deadline))
            self.update_offloaded_tasks_counter()  # update offloaded task counter
            # update energy consumption
            self.update_energy_consumption(transmission_time, offload=True)

    def generate_local_task(self, task):
        # process the task
        # add logging
        # update energy levels
        # note that some tasks don't get generated because each application sends the requests sequentially.
        # might need to update this behavior to make them queued instead of having empty queue
        start_waiting = env.now
        priority = task.deadline - env.now
        task.set_priority(priority)

        with self.processor.request(priority=priority) as req:
            req.obj = task
            end_waiting = env.now
            waiting_time = end_waiting - start_waiting
            self.local_execution_delays.append(waiting_time)  # not utilized
            task_processing_time = req.obj.get_task_cpu_cycles() / self.processing_capability

            yield env.timeout(task_processing_time)
            self.total_execution_delays.append((env.now, task.deadline, env.now - task.deadline))
            self.update_locally_executed_tasks()  # increment by one
            self.update_energy_consumption(task_processing_time, offload=False)  # These 2 are redundant
            self.update_local_processing_time(task_processing_time)

    def offload_tasks(self, tasks):
        """
        This method should be updated such that the server handles the execution and update these info
        :param tasks:
        :return:
        """
        for task in tasks:
            env.process(self.generate_offloaded_tasks(task))

    def generate_ue_tasks(self):
        """
        This method creates a set of tasks that form an application, with inter-arrival time between applications
        following a poisson distribution
        """
        delay_critical = 1
        while True:
            self.current_application_tasks_list = list()
            next_request = expon.rvs(scale=self.poisson_mean, size=1)
            yield env.timeout(next_request.item(0))  # delay between events
            number_of_subtasks = random.randint(5, 15)

            self.update_generated_tasks(number_of_subtasks)
            for i in range(number_of_subtasks):

                task = Task()
                task.create_task(
                    self.processing_capability)  # This method sets the required CPU
                # cycles and input data size
                self.current_application_tasks_list.append(task)
                self.complete_task_list.append(task)
            delay_critical = 1 - delay_critical
            # print("delay critical ",delay_critical)
            # print("APP subtask: ", len(self.current_application_tasks_list))
            self.log_application_load(self.current_application_tasks_list)
            if EXECUTION_MODE == 0:
                self.process_task_locally(self.current_application_tasks_list)
            elif EXECUTION_MODE == 1:
                self.offload_tasks(self.current_application_tasks_list)
            elif EXECUTION_MODE == 2:
                offloading_decision = self.optimize_offloading()
                local_exec_list = list()
                offloading_list = list()
                index = 0
                for task in self.current_application_tasks_list:
                    if offloading_decision[index] == 0:
                        local_exec_list.append(task)
                    else:
                        offloading_list.append(task)
                    index = index + 1

                self.process_task_locally(local_exec_list)
                self.offload_tasks(offloading_list)

        # to do: call optimizer here to choose which tasks to offload -> done
        # perform local execution
        # transmit remote execution tasks

    def log_application_load(self, app_list):
        cpu_sum = 0
        input_sum = 0
        for task in app_list:
            cpu_sum = cpu_sum + task.get_task_cpu_cycles()
            input_sum = input_sum + task.get_input_size()
        self.user_load_stat.append([env.now, cpu_sum, input_sum])

    def objective_multi(self, solution):

        def energy(solution_array):
            solution_energy_consumption = 0
            i = 0
            for task in self.current_application_tasks_list:
                if solution_array[i] == 0:  # local execution
                    task_execution_energy_consumption = self.processing_energy_consumption_per_second * (
                            task.get_task_cpu_cycles() / self.processing_capability)

                else:  # energy consumption of offloaded tasks = transmission energy consumption of the task
                    # data = (input data size /Bandwidth ) * transmission power consumption
                    task_execution_energy_consumption = self.transmission_energy_consumption_per_second * \
                                                        (task.get_input_size() / BANDWIDTH)

                i = i + 1
                # print("en: ", i, task_execution_energy_consumption)
                solution_energy_consumption = solution_energy_consumption + task_execution_energy_consumption

            # print("energy values: "," ", solution_energy_consumption)
            return solution_energy_consumption

        def delay(solution_array):
            solution_execution_delay = 0
            i = 0
            for task in self.current_application_tasks_list:
                if solution_array[i] == 0:  # local execution
                    task_execution_delay = task.get_task_cpu_cycles() / self.processing_capability
                    # print(task.get_deadline())
                    if (task_execution_delay + env.now) > task.get_deadline():
                        task_execution_delay = task_execution_delay * 2
                        # print(" delayed task expected")
                else:  # execution delay on the MEC server
                    task_execution_delay = (task.get_input_size() / BANDWIDTH) + task.get_task_cpu_cycles() / \
                                           SERVER_PROCESSING_CAPABILITIES
                    if (task_execution_delay + env.now) > task.get_deadline():
                        task_execution_delay = task_execution_delay * 2
                solution_execution_delay = solution_execution_delay + task_execution_delay
                i = i + 1
                # print("delay:" ,i, " ", task_execution_delay)
            # print("delay values: ", solution_execution_delay)

            return solution_execution_delay

        return [energy(solution), delay(solution)]

    def optimize_offloading(self):
        """
        This method takes a list of tasks, and run Tug of War Optimization (TWO) algorithm to determine
        the subset of subtasks to offload, where 0 indicates local execution and 1 indicates offloading
        Don't forget to add the constraint to the optimization

        :return:
        """
        problem_multi = {
            "obj_func": self.objective_multi,
            "bounds": TransferBinaryVar(n_vars=len(self.current_application_tasks_list), name="delta",
                                        tf_func="vstf_04"),
            "minmax": "min",
            "obj_weights": [1, 1],  # Define it or default value will be [1, 1, 1]
            "log_to": 'log.txt'
        }

        # model = CircleSA.OriginalCircleSA(epoch=60, pop_size=50, c_factor=0.8) # this one doing good
        # model = EO.AdaptiveEO(epoch=50, pop_size=50)
        model = TWO.OriginalTWO(epoch=40, pop_size=35)
        g_best = model.solve(problem_multi)
        # print("-------------------------------------------------")
        # print(f"Best Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
        return g_best.solution


def generate_ue(num_of_users):
    x_min = 0
    x_max = 1
    y_min = 0
    y_max = 1
    # rectangle dimensions
    x_delta = x_max - x_min
    y_delta = y_max - y_min
    area_total = x_delta * y_delta  # area of rectangle

    # Point process parameters
    # lambda0 = 50  # intensity (ie mean density) of the Poisson process

    # Simulate a Poisson point process
    # numbPoints = np.random.poisson(lambda0 * area_total)  # Poisson number of points

    xx = x_delta * np.random.uniform(0, 1, num_of_users) + x_min  # y coordinates of Poisson points
    yy = y_delta * np.random.uniform(0, 1, num_of_users) + y_min  # y coordinates of Poisson points
    user_list = list()
    for i in range(num_of_users):
        poisson_mean = random.randint(1, 6) * 0.5
        processing_capability = random.randint(1, 2)
        ue = UE(name=i, poisson_mean=poisson_mean, processing_energy_consumption_per_second=0.6,
                transmission_energy_consumption_per_second=0.7,
                processing_capability=processing_capability)
        ue.set_coordinates(xx[i], yy[i])
        user_list.append(ue)
    return user_list


es1 = ES(SERVER_PROCESSING_CAPABILITIES)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    es1.set_position(0.5, 0.5)
    users_list = generate_ue(num_of_users=40)
    processes_list = list()
    for user in users_list:
        p = env.process(user.generate_ue_tasks())
        processes_list.append(p)
    post_exec_avg = 0
    qualified_avg = 0
    env.run(until=SIM_TIME)
    for user in users_list:
        user.print_stat()
        user.export_ue_stat()
        post_exec_avg = post_exec_avg + user.get_avg_delay_post_deadline()
        qualified_avg = qualified_avg + user.get_qualified_tasks()

    print("**************")
    print("avg qualified: \n", qualified_avg / len(users_list))
    print("avg delay: \n", post_exec_avg / len(users_list))

