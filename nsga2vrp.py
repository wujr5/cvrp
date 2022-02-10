from copy import deepcopy
import os
import io
import random
import numpy
import csv
from functools import cmp_to_key
import datetime
from json import load
from deap import base, creator, tools

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class nsgaAlgo():

    def __init__(self, popSize, mutProb, numGen, type):
        self.json_instance = self.load_instance('./data/json/RC104.json')
        self.speed = self.load_speed('./data/speed.csv')
        self.ind_size = self.json_instance['Number_of_customers']
        self.pop_size = popSize
        self.mut_prob = mutProb
        self.num_gen = numGen
        self.type = type
        self.toolbox = base.Toolbox()

        self.A = 0.5  # 顾客时间窗左端（即 ready time）=0
        self.B = 1 - self.A  # 其余则为B类，A + B = 1

        # self.logbook, self.stats = self.createStatsObjs()
        self.createCreators()

    # 加载 json 文件
    def load_instance(self, json_file):
        if os.path.exists(path=json_file):
            with io.open(json_file, 'rt', newline='') as file_object:
                return load(file_object)
        return None

    # 读取速度文件
    def load_speed(self, csv_file):
        if os.path.exists(path=csv_file):
            with io.open(csv_file, 'rt', newline='') as f:
                f_csv = list(csv.reader(f))
                speed_config = {}
                for i in range(1, len(f_csv)):
                    timestr = f_csv[i][0]
                    speedstr = float(f_csv[i][1].replace(' ', ''))

                    timestrGap = timestr.split('-')

                    # 记录最早的时间点
                    if i == 1:
                        speed_config[-1.0] = timestrGap[0].split(':')

                    end = timestrGap[1].split(':')

                    # 计算跟最早时间点的时间段
                    timegap = datetime.timedelta(
                        hours=int(end[0]), minutes=int(end[1])) - (datetime.timedelta(
                            hours=int(speed_config[-1][0]), minutes=int(speed_config[-1][1])))

                    speed_config[timegap.total_seconds() / 60] = speedstr

                speed = {0.0: 0.0}
                time_keys = list(speed_config.keys())
                time_keys.remove(-1)
                time_keys.sort()

                last_time = 0
                last_distance = 0
                for t in time_keys:
                    for k in range(last_time, int(t)):
                        speed[k + 1] = round(last_distance +
                                             speed_config[t] *
                                             (k + 1 - last_time) / 60, 2)

                    last_time = int(t)
                    last_distance = speed[int(t)]

                return speed
        return None

    # 设置 deap 库需要使用的各个函数
    def createCreators(self):
        # 适应值函数
        creator.create('FitnessMulti', base.Fitness, weights=(1.0, -1.0))
        # 创建个体容器
        creator.create('Individual', list, fitness=creator.FitnessMulti)

        # 初始化种群方式：随机
        if self.type == 1:
            self.toolbox.register('indexes', random.sample, range(
                1, self.ind_size + 1), self.ind_size)

        # 初始化种群方式：随机抽取顾客点后，根据距离长短
        else:
            self.toolbox.register('indexes', self.initPopulation, range(
                1, self.ind_size + 1), self.ind_size)

        # 个体初始化函数
        self.toolbox.register('individual', tools.initIterate,
                              creator.Individual, self.toolbox.indexes)

        # 种群初始化函数
        self.toolbox.register('population', tools.initRepeat,
                              list, self.toolbox.individual)

        # 设置适应值计算函数
        self.toolbox.register('evaluate', self.evaluate, unit_cost=1)

        # 设置选择算法
        self.toolbox.register("select", tools.selNSGA2)

        # 设置交配算法
        self.toolbox.register("mate", self.crossOverVrp)

        # 设置变异算法
        self.toolbox.register("mutate", self.mutation)

    # 定制的种群初始化方式
    def initPopulation(self, customers, size):
        result = []
        customers = list(customers)
        instance = self.json_instance
        capacity = instance['vehicle_capacity']

        # 升序排列
        def cmp(a, b):
            return a['d'] - b['d']

        while len(customers) > 0:
            k = random.sample(customers, 1)[0]
            result.append(k)
            customers.remove(k)

            distance_arr = [{'c': i, 'd': instance['distance_matrix'][k][i]}
                            for i in customers]
            distance_arr.sort(key=cmp_to_key(cmp))

            sub_capacity = instance['customer_' + str(k)]['demand']

            i = 0
            while len(customers) > 0 and sub_capacity + distance_arr[i]['d'] <= capacity:
                result.append(distance_arr[i]['c'])
                customers.remove(distance_arr[i]['c'])
                sub_capacity += distance_arr[i]['d']
                i = i + 1

        return result

    # 初始化适应值
    def generatingPopFitness(self):
        self.pop = self.toolbox.population(n=self.pop_size)

        for ind in self.pop:
            ind.fitness.values = self.toolbox.evaluate(ind)

        # self.recordStat(self.invalid_ind, self.logbook,
        #                 self.pop, self.stats, gen=0)

    # 运行入口
    def runGenerations(self):
        best_fitness = 0
        best_fitness_count = 0
        type_config = {1: '随机', 2: '定向'}

        for gen in range(self.num_gen):

            # 选择用于交配的后代
            self.parents = tools.selTournament(
                self.pop, int(len(self.pop) / 2), int(len(self.pop) / 2))

            self.offsprings = []

            # 交配与变异操作
            for i in range(int(len(self.parents) / 2)):
                ind1 = self.parents[i]
                ind2 = self.parents[i * 2 + 1]

                # 交配
                new1, new2 = self.toolbox.mate(ind1, ind2)
                self.offsprings += [new1, new2]

                # 变异
                new3 = self.toolbox.mutate(new1)
                new4 = self.toolbox.mutate(new2)
                self.offsprings += [new3, new4]

            for i in range(len(self.offsprings)):
                # 2-opt操作
                # self.offsprings[i] = self.operate2opt(self.offsprings[i])
                # 重新计算适应值
                self.offsprings[i].fitness.values = self.toolbox.evaluate(
                    self.offsprings[i])

            # 使用 nsga2 算法，重新选择种群
            self.pop = self.toolbox.select(
                self.pop + self.offsprings, self.pop_size)

            best_individual = tools.selBest(self.pop, 1)[0]

            if best_fitness == best_individual.fitness.values[1]:
                best_fitness_count += 1
            else:
                best_fitness = best_individual.fitness.values[1]
                best_fitness_count = 0

            print(
                f'迭代：{gen + 1}，变异：{self.mut_prob}，类型：{type_config[self.type]}，满意：{best_individual.fitness.values[0]}，成本：{best_individual.fitness.values[1]}，相同次数：{best_fitness_count}')

            # 生成日志
            # self.recordStat(self.offsprings, self.logbook,
            #                 self.pop, self.stats, gen + 1)

    # 2-opt 算法
    def operate2opt(self, ind):
        subroute = self.routeToSubroute(ind)
        result = []

        # 升序排列
        def cmp(a, b):
            return a['d'] - b['d']

        for i in range(len(subroute)):
            distance_arr = [{'c': customer, 'd': self.json_instance['distance_matrix'][customer][0]}
                            for customer in subroute[i]]
            distance_arr.sort(key=cmp_to_key(cmp))

            # 距离配送中心最近的
            result.append(distance_arr[0]['c'])
            subroute[i].remove(result[-1])

            while len(subroute[i]) > 0:
                distance_arr = [{'c': k, 'd': self.json_instance['distance_matrix'][result[-1]][k]}
                                for k in subroute[i]]
                distance_arr.sort(key=cmp_to_key(cmp))

                result.append(distance_arr[0]['c'])
                subroute[i].remove(result[-1])

        return creator.Individual(list(result))

    # 交配算法
    def crossOverVrp(self, input_ind1, input_ind2):
        # 升序排列
        def cmp(a, b):
            return a[1] - b[1]

        # cross over 方式交配
        def cross(item1, item2, a, b):
            newitem = [0 for i in item1]

            for i in range(a, b + 1):
                newitem[i] = item1[i]
                item2.remove(item1[i])

            for i in item2:
                newitem[newitem.index(0)] = i

            return newitem

        # 选取两个切片位置，并保证 a < b
        a, b = random.sample(range(self.ind_size), 2)
        if a > b:
            a, b = b, a

        # 存放子代
        new1 = cross(deepcopy(input_ind1), deepcopy(input_ind2), a, b)
        new2 = cross(deepcopy(input_ind2), deepcopy(input_ind1), a, b)

        return creator.Individual(list(new1)), creator.Individual(list(new2))

    # 变异算法
    def mutation(self, individual):
        size = len(individual)
        ind = deepcopy(individual)

        for i in range(size):
            if random.random() < self.mut_prob:
                swap_indx = random.randint(0, size - 2)
                if swap_indx >= i:
                    swap_indx += 1
                ind[i], ind[swap_indx] = ind[swap_indx], ind[i]

        return creator.Individual(list(ind))

    # 根据输入的速度数据获取消耗的时间
    def getTimeCostByInputSpeed(self, startTime, distance):
        d = distance / 10
        find_d = 0
        time_cost = 1

        while find_d < d:
            find_d = self.speed[startTime + time_cost] - \
                self.speed[float(startTime)]
            time_cost += 1

        return time_cost

    # 满意度函数
    def getSatisfaction(self, individual):
        left_edge = 20  # 可容忍早到时间
        right_edge = 20  # 可容忍迟到时间

        all_sub_route = self.routeToSubroute(individual)
        A_Customer = []
        B_Customer = []

        for sub_route in all_sub_route:
            last_customer_id = 0
            sub_time_cost = 0

            for customer_id in sub_route:
                customer_satisfaction = 0
                customer = self.json_instance["customer_" + str(customer_id)]
                distance = self.json_instance["distance_matrix"][last_customer_id][customer_id]

                sub_time_cost = sub_time_cost + self.getTimeCostByInputSpeed(
                    sub_time_cost, distance)

                # 早到 left_edge 分钟内
                if sub_time_cost >= (customer['ready_time'] - left_edge) and sub_time_cost < customer['ready_time']:
                    customer_satisfaction += 100 * \
                        (1 - (customer['ready_time'] -
                         sub_time_cost) / left_edge)
                # 早到
                elif sub_time_cost < customer['ready_time']:
                    customer_satisfaction += 0
                # 刚好
                elif sub_time_cost >= customer['ready_time'] and sub_time_cost <= customer['due_time']:
                    customer_satisfaction += 100
                # 迟到 right_edge 分钟内
                elif sub_time_cost > customer['due_time'] and sub_time_cost <= (customer['due_time'] + right_edge):
                    customer_satisfaction += 100 * \
                        (1 - (sub_time_cost -
                         customer['due_time']) / right_edge)
                # 迟到
                elif sub_time_cost > customer['due_time']:
                    customer_satisfaction += 0

                # 加上服务时间
                sub_time_cost += 3

                if customer['ready_time'] == 0:
                    A_Customer.append(customer_satisfaction)
                else:
                    B_Customer.append(customer_satisfaction)

                last_customer_id = customer_id

            # 加权计算平均满意度
            A_Satisfaction = 0
            B_Satisfaction = 0

            for s in A_Customer:
                A_Satisfaction += s
            for s in B_Customer:
                B_Satisfaction += s

        return round(A_Satisfaction * self.A / len(A_Customer) + B_Satisfaction * self.B / len(B_Customer), 2)

    # 返回带子路径的二维数组
    def routeToSubroute(self, individual):
        instance = self.json_instance
        route = []
        sub_route = []
        vehicle_load = 0
        vehicle_capacity = instance['vehicle_capacity']

        for customer_id in individual:
            demand = instance[f"customer_{customer_id}"]["demand"]
            updated_vehicle_load = vehicle_load + demand

            if(updated_vehicle_load <= vehicle_capacity):
                sub_route.append(customer_id)
                vehicle_load = updated_vehicle_load
            else:
                route.append(sub_route)
                sub_route = [customer_id]
                vehicle_load = demand

        if sub_route != []:
            route.append(sub_route)

        return route

    def printRoute(self, route, merge=False):
        route_str = '0'
        sub_route_count = 0
        for sub_route in route:
            sub_route_count += 1
            sub_route_str = '0'
            for customer_id in sub_route:
                sub_route_str = f'{sub_route_str} - {customer_id}'
                route_str = f'{route_str} - {customer_id}'
            sub_route_str = f'{sub_route_str} - 0'
            if not merge:
                print(f'  Vehicle {sub_route_count}\'s route: {sub_route_str}')
            route_str = f'{route_str} - 0'
        if merge:
            print(route_str)

    # 计算个体的车辆数

    def getVehicleNum(self, individual):
        all_sub_route = self.routeToSubroute(individual)
        num_of_vehicles = len(all_sub_route)
        return num_of_vehicles

    # 计算个体的距离成本

    def getRouteCost(self, individual, unit_cost=1):
        # 总成本
        total_cost = 0

        all_sub_route = self.routeToSubroute(individual)

        for sub_route in all_sub_route:
            sub_route_distance = 0
            # 从配送点出发
            last_customer_id = 0

            for customer_id in sub_route:
                distance = self.json_instance["distance_matrix"][last_customer_id][customer_id]
                sub_route_distance += distance
                last_customer_id = customer_id

            # 回到配送点
            sub_route_distance = sub_route_distance + \
                self.json_instance["distance_matrix"][last_customer_id][0]

            # 乘以单位成本
            sub_route_transport_cost = unit_cost * sub_route_distance

            total_cost = total_cost + sub_route_transport_cost

        return total_cost

    # 计算个体的每个路径的距离成本

    def getSubRouteCost(self, individual, unit_cost=1):
        # 单条路径和成本
        route_cost = []

        all_sub_route = self.routeToSubroute(individual)

        for sub_route in all_sub_route:
            sub_route_distance = 0
            # 从配送点出发
            last_customer_id = 0

            for customer_id in sub_route:
                distance = self.json_instance["distance_matrix"][last_customer_id][customer_id]
                sub_route_distance += distance
                last_customer_id = customer_id

            # 回到配送点
            sub_route_distance = sub_route_distance + \
                self.json_instance["distance_matrix"][last_customer_id][0]

            # 乘以单位成本
            sub_route_transport_cost = unit_cost * sub_route_distance

            route_cost.append((sub_route, sub_route_transport_cost))

        return route_cost

    # 计算适应值
    def evaluate(self, individual, unit_cost=1):

        # 用车成本
        vehicles = self.getVehicleNum(individual)

        # 路程成本
        total_cost = self.getRouteCost(individual, unit_cost)

        # 获取满意度
        satisfaction = self.getSatisfaction(individual)

        # return (1 / satisfaction * 100, vehicles + route_cost)
        return (satisfaction, vehicles * 30 + total_cost / 10 * 20)

    # Statistics and Logging

    def createStatsObjs(self):
        # Method to create stats and logbook objects
        """
        Inputs : None
        Outputs : tuple of logbook and stats objects.
        """
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", numpy.mean, axis=0)
        stats.register("std", numpy.std, axis=0)
        stats.register("min", numpy.min, axis=0)
        stats.register("max", numpy.max, axis=0)

        # Methods for logging
        logbook = tools.Logbook()
        logbook.header = "Generation", "evals", "avg", "std", "min", "max", "best_one", "fitness_best_one"
        return logbook, stats

    def recordStat(self, invalid_ind, logbook, pop, stats, gen):
        """
        Inputs : invalid_ind - Number of children for which fitness is calculated
                logbook - Logbook object that logs data
                pop - population
                stats - stats object that compiles statistics
        Outputs: None, prints the logs
        """
        record = stats.compile(pop)
        best_individual = tools.selBest(pop, 1)[0]
        record["best_one"] = best_individual
        record["fitness_best_one"] = best_individual.fitness
        logbook.record(Generation=gen, evals=len(invalid_ind), **record)
        print(
            f'迭代：{gen}，车辆：{best_individual.fitness.values[0]}，距离：{best_individual.fitness.values[1]}')

    def getBestInd(self):
        self.best_individual = tools.selBest(self.pop, 1)[0]

        # Printing the best after all generations
        # print(f"最好：{self.best_individual}")
        # print(
        #     f"最好的样本：满意：{self.best_individual.fitness.values[0]}，成本：{self.best_individual.fitness.values[1]}")

        # Printing the route from the best individual
        # self.printRoute(self.routeToSubroute(self.best_individual))

    def doExport(self):
        csv_file_name = f"{self.json_instance['instance_name']}_" \
            f"pop{self.pop_size}" \
            f"_mutProb{self.mut_prob}_numGen{self.num_gen}.csv"
        self.exportCsv(csv_file_name, self.logbook)

    def exportCsv(self, csv_file_name, logbook):
        csv_columns = logbook[0].keys()
        csv_path = os.path.join(BASE_DIR, "results", csv_file_name)
        try:
            with open(csv_path, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                writer.writeheader()
                for data in logbook:
                    writer.writerow(data)
        except IOError:
            print("I/O error: ", csv_path, csv_file_name)

    def runMain(self):
        self.generatingPopFitness()
        self.runGenerations()
        # self.getBestInd()
        # self.doExport()
