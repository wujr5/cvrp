from copy import deepcopy
import os
import io
import random
import numpy
import csv
from functools import cmp_to_key

from json import load
from deap import base, creator, tools

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class nsgaAlgo():

    def __init__(self, popSize, mutProb, numGen):
        self.json_instance = self.load_instance('./data/json/C101.json')
        self.ind_size = self.json_instance['Number_of_customers']
        self.pop_size = popSize
        self.mut_prob = mutProb
        self.num_gen = numGen
        self.toolbox = base.Toolbox()
        # self.logbook, self.stats = self.createStatsObjs()
        self.createCreators()

    # 设置 deap 库需要使用的各个函数
    def createCreators(self):
        # 适应值函数
        creator.create('FitnessMin', base.Fitness, weights=(-1.0, -1.0))
        # 创建个体容器
        creator.create('Individual', list, fitness=creator.FitnessMin)

        # 初始化种群方式：随机
        # self.toolbox.register('indexes', random.sample, range(
        #     1, self.ind_size + 1), self.ind_size)

        # 初始化种群方式：随机抽取顾客点后，根据距离长短
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

            # 2-opt操作
            for i in range(len(self.offsprings)):
                self.offsprings[i] = self.operate2opt(self.offsprings[i])
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
                f'迭代：{gen + 1}，车辆：{best_individual.fitness.values[0]}，距离：{best_individual.fitness.values[1]}，相同次数：{best_fitness_count}')

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

    # 满意度函数
    def getSatisfaction(self, individual):
        speed = 30  # 行驶速度
        left_edge = 100  # 可容忍早到时间
        right_edge = 100  # 可容忍迟到时间

        total_satisfaction = 0
        updated_route = self.routeToSubroute(individual)

        for sub_route in updated_route:
            # 记录上一个顾客点 id，默认从配送点出发
            last_customer_id = 0
            # 当前路线的总满意度
            sub_satisfaction = 0
            # 当前路线的耗时
            sub_time_cost = 0

            for customer_id in sub_route:
                # 顾客点
                customer = self.json_instance["customer_" + str(customer_id)]
                # 行驶距离
                distance = self.json_instance["distance_matrix"][last_customer_id][customer_id]
                # 耗时
                sub_time_cost = sub_time_cost + distance / speed

                # 早到 left_edge 分钟内
                if sub_time_cost >= (customer['ready_time'] - left_edge) and sub_time_cost < customer['ready_time']:
                    sub_satisfaction += 100 * \
                        (1 - (customer['ready_time'] -
                         sub_time_cost) / left_edge)
                # 早到
                elif sub_time_cost < customer['ready_time']:
                    sub_satisfaction += 0
                # 刚好
                elif sub_time_cost >= customer['ready_time'] and sub_time_cost <= customer['due_time']:
                    sub_satisfaction += 100
                # 迟到 right_edge 分钟内
                elif sub_time_cost > customer['due_time'] and sub_time_cost <= (customer['due_time'] + right_edge):
                    sub_satisfaction += 100 * \
                        (1 - (sub_time_cost -
                         customer['due_time']) / right_edge)
                # 迟到
                elif sub_time_cost > customer['due_time']:
                    sub_satisfaction += 0

                # 加上服务时间
                sub_time_cost += customer['service_time']

                # Update last_customer_id to the new one
                last_customer_id = customer_id

            # Adding this to total cost
            total_satisfaction = total_satisfaction + sub_satisfaction

        return total_satisfaction / self.json_instance['Number_of_customers']

    def getBestInd(self):
        self.best_individual = tools.selBest(self.pop, 1)[0]

        # Printing the best after all generations
        print(f"最好：{self.best_individual}")
        print(f"车辆：{self.best_individual.fitness.values[0]}")
        print(f"距离：{self.best_individual.fitness.values[1]}")

        # Printing the route from the best individual
        self.printRoute(self.routeToSubroute(self.best_individual))

    def doExport(self):
        csv_file_name = f"{self.json_instance['instance_name']}_" \
                        f"pop{self.pop_size}" \
                        f"_mutProb{self.mut_prob}_numGen{self.num_gen}.csv"
        self.exportCsv(csv_file_name, self.logbook)

    # 加载 json 文件

    def load_instance(self, json_file):
        """
        Inputs: path to json file
        Outputs: json file object if it exists, or else returns NoneType
        """
        if os.path.exists(path=json_file):
            with io.open(json_file, 'rt', newline='') as file_object:
                return load(file_object)
        return None

    # 将二维数组的子路径转化为一维数组
    def subroute2Route(self, subroute):
        ind = []

        for r in subroute:
            ind += r

        return ind

    # 返回带子路径的二维数组
    def routeToSubroute(self, individual):
        """
        Inputs: Sequence of customers that a route has
                Loaded instance problem
        Outputs: Route that is divided in to subroutes
                which is assigned to each vechicle.
        """
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

        # Returning the final route with each list inside for a vehicle
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
        """
        Inputs: Individual route
                Json file object loaded instance
        Outputs: Number of vechiles according to the given problem and the route
        """
        # Get the route with subroutes divided according to demand
        updated_route = self.routeToSubroute(individual)
        num_of_vehicles = len(updated_route)
        return num_of_vehicles

    # 计算个体的距离成本

    def getRouteCost(self, individual, unit_cost=1):
        total_cost = 0
        updated_route = self.routeToSubroute(individual)

        for sub_route in updated_route:
            # Initializing the subroute distance to 0
            sub_route_distance = 0
            # Initializing customer id for depot as 0
            last_customer_id = 0

            for customer_id in sub_route:
                # Distance from the last customer id to next one in the given subroute
                distance = self.json_instance["distance_matrix"][last_customer_id][customer_id]
                sub_route_distance += distance
                # Update last_customer_id to the new one
                last_customer_id = customer_id

            # After adding distances in subroute, adding the route cost from last customer to depot
            # that is 0
            sub_route_distance = sub_route_distance + \
                self.json_instance["distance_matrix"][last_customer_id][0]

            # Cost for this particular sub route
            sub_route_transport_cost = unit_cost * sub_route_distance

            # Adding this to total cost
            total_cost = total_cost + sub_route_transport_cost

        return total_cost

    # Get the fitness of a given route

    def evaluate(self, individual, unit_cost=1):

        # 用车成本
        vehicles = self.getVehicleNum(individual)

        # 路程成本
        route_cost = self.getRouteCost(individual, unit_cost)

        # 获取满意度
        # satisfaction = self.getSatisfaction(individual)

        # return (1 / satisfaction * 100, vehicles + route_cost)
        return (vehicles, route_cost)

    ## Statistics and Logging

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

    # Exporting CSV files

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
        self.getBestInd()
        # self.doExport()
