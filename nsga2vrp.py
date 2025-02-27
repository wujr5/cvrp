from copy import deepcopy
from math import floor
import os
import io
import random
import csv
from functools import cmp_to_key
import datetime
from json import load
from deap import base, creator, tools
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class nsgaAlgo():

    def __init__(self, popSize, mutProb, numGen, type, file, baseAl=2, time=12):
        self.file = file
        self.base = baseAl
        self.json_instance = self.load_instance(
            f'./data/json/{self.file}.json')

        self.speed = self.load_speed('./data/speed.csv')
        self.ind_size = self.json_instance['Number_of_customers']

        self.pop_size = popSize
        self.mut_prob = mutProb
        self.num_gen = numGen
        self.type = type
        self.toolbox = base.Toolbox()

        self.A = 0.5  # 顾客时间窗左端（即 ready time）=0
        self.B = 1 - self.A  # 其余则为B类，A + B = 1

        # 车辆出发原点时间，0，表示 6:00，1 单位时间是 1 分钟
        self.start_time = time
        self.original_time = (self.start_time - 6) * 60  # 设置为下午 13:00 出发

        self.logbook = tools.Logbook()
        self.logbook.header = "generation", "fitness"

        # 算子操作的最多执行次数
        self.opt_stop_num = 20

        self.createCreators()

    def reset(self):
        self.logbook = tools.Logbook()
        self.logbook.header = "generation", "fitness"

    # 加载 json 文件
    def load_instance(self, json_file):
        if os.path.exists(path=json_file):
            with io.open(json_file, 'rt', newline='') as file_object:
                return load(file_object)
        return None

    # 读取速度文件
    def load_speed(self, csv_file):
        if os.path.exists(path=csv_file):
            with io.open(csv_file, 'rt', newline='', encoding='utf8') as f:
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
                    # speed_config[timegap.total_seconds() / 60] = 30

                speed = {0.0: 0.0}
                time_keys = list(speed_config.keys())
                time_keys.remove(-1)
                time_keys.sort()

                last_time = 0
                last_distance = 0
                for t in time_keys:
                    for k in range(last_time, int(t)):
                        for j in range(100 + 1):
                            speed[k + j * 0.01] = round(last_distance +
                                                        speed_config[t] *
                                                        (k + j * 0.01 - last_time) / 60, 2)

                    last_time = int(t)
                    last_distance = speed[float(t)]
                return speed

        return None

    # 设置 deap 库需要使用的各个函数
    def createCreators(self):
        # 适应值函数
        creator.create('FitnessMulti', base.Fitness, weights=(-1.0,))
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
        self.toolbox.register('evaluate', self.evaluate)

        # 设置选择算法
        self.toolbox.register("select", tools.selBest)

        # 设置交配算法
        if self.base == 1:
            self.toolbox.register("mate", self.baseCrossOverVrp)
        else:
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
    def init_generation(self):
        self.pop = self.toolbox.population(n=self.pop_size)

        for ind in self.pop:
            ind.fitness.values = self.toolbox.evaluate(ind)

    # 运行入口
    def runGenerations(self):
        type_config = {1: '随机', 2: '定向'}

        for gen in range(self.num_gen):

            # 选择用于交配的后代
            self.parents = tools.selBest(self.pop, 40)

            self.offsprings = []

            # 交配与变异操作
            for i in range(40):
                select = random.sample(list(range(40)), 2)
                ind1 = self.parents[select[0]]
                ind2 = self.parents[select[1]]

                if random.random() < 0.9:
                    # 交配
                    new1, new2 = self.toolbox.mate(ind1, ind2)
                    self.offsprings += [new1, new2]

                    # 变异
                    new3 = self.toolbox.mutate(new1)
                    new4 = self.toolbox.mutate(new2)
                    self.offsprings += [new3, new4]

                    # 算子操作
                    new5 = self.opt_pdp(new3, gen)
                    new6 = self.opt_pdp(new4, gen)
                    self.offsprings += [new5, new6]

            for i in range(len(self.offsprings)):
                # 重新计算适应值
                self.offsprings[i].fitness.values = self.toolbox.evaluate(
                    self.offsprings[i])

            # 使用 nsga2 算法，重新选择种群
            self.pop = self.toolbox.select(
                self.pop + self.offsprings, self.pop_size)

            self.best_individual = tools.selBest(self.pop, 1)[0]

            print(
                f'迭代：{gen + 1}，时间：{self.start_time}，类型：{type_config[self.type]}，适应值：{self.best_individual.fitness.values}，车辆：{self.getVehicleNum(self.best_individual)}，距离：{self.getDistance(self.best_individual)}，满意度：{self.getSatisfaction(self.best_individual, True)[0]}')

            # 生成日志
            self.logbook.record(
                generation=gen + 1, fitness=f'{self.best_individual.fitness.values}')

        print(f'最好路径：{self.best_individual}')

        self.printRoute(self.best_individual)

    def getDistance(self, ind):
        all_sub_route = self.routeToSubroute(ind)
        all_distance = 0

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

            # 乘以单位成本，加上早到和迟到惩罚成本
            all_distance += sub_route_distance

        return round(all_distance, 1)

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

    # 基础交配算法
    def baseCrossOverVrp(self, input_ind1, input_ind2):
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

    # 交配算法
    def crossOverVrp(self, input_ind1, input_ind2):
        # 升序排列
        def cmp(a, b):
            return a[1] - b[1]

        # 根据左时间窗进行交叉
        def crossByLeftTime(item1, item2):
            length = len(item1)
            newitem = []

            # 取出客户点，插入指定位置
            def insertPos(ind, customer, pos):
                ind.remove(customer)
                ind.insert(pos, customer)

            for i in range(0, length):
                ready_time_1 = self.json_instance[f"customer_{item1[i]}"]["ready_time"]
                due_time_1 = self.json_instance[f"customer_{item1[i]}"]["due_time"]
                avg1 = (ready_time_1 + due_time_1) / 2

                ready_time_2 = self.json_instance[f"customer_{item2[i]}"]["ready_time"]
                due_time_2 = self.json_instance[f"customer_{item2[i]}"]["due_time"]
                avg2 = (ready_time_2 + due_time_2) / 2

                if avg1 < avg2:
                    newitem.append(item1[i])
                    insertPos(item2, item1[i], i)
                else:
                    newitem.append(item2[i])
                    insertPos(item1, item2[i], i)

            return newitem

        # 根据最小距离进行交叉
        def crossByDistance(item1, item2):
            length = len(item1)
            newitem = []

            # 交换两个客户点的位置
            def swapPos(ind, customer, pos):
                pos_new = ind.index(customer)
                ind[pos], ind[pos_new] = ind[pos_new], ind[pos]

            for i in range(0, length):
                if i == 0:
                    rand = random.randint(0, 1)
                    newitem = [item1[0] if rand == 0 else item2[0]]
                    swapPos(item2 if rand == 0 else item1, newitem[0], i)
                else:
                    distance_1 = self.json_instance["distance_matrix"][newitem[i - 1]][item1[i]]
                    distance_2 = self.json_instance["distance_matrix"][newitem[i - 1]][item2[i]]

                    if distance_1 < distance_2:
                        newitem.append(item1[i])
                        swapPos(item2, item1[i], i)
                    else:
                        newitem.append(item2[i])
                        swapPos(item1, item2[i], i)
            return newitem

        # 存放子代
        new1 = crossByLeftTime(deepcopy(input_ind1), deepcopy(input_ind2))
        new2 = crossByDistance(deepcopy(input_ind1), deepcopy(input_ind2))

        return creator.Individual(list(new1)), creator.Individual(list(new2))

    # 变异算法
    def mutation(self, individual):
        size = len(individual)
        ind = deepcopy(individual)

        for i in range(size):
            if random.random() < self.mut_prob:
                swap_indx1 = random.randint(0, size - 1)
                swap_indx2 = random.randint(0, size - 1)
                ind[swap_indx1], ind[swap_indx2] = ind[swap_indx2], ind[swap_indx1]

        return creator.Individual(list(ind))

    # 根据输入的速度数据获取消耗的时间
    def getTimeCostByInputSpeed(self, startTime, distance):
        # 加上出发时间
        t = startTime + self.original_time

        d = distance / 10
        find_d = 0
        time_cost = 0

        while find_d < d:
            find_d = self.speed[round(t + time_cost, 2)] - \
                self.speed[round(t, 2)]
            time_cost += 0.01

        return time_cost

    # 满意度函数
    def getSatisfaction(self, individual, debug=False):
        left_edge = 10  # 可容忍早到时间
        right_edge = 15  # 可容忍迟到时间

        # 路径时间
        time_of_route = []

        all_sub_route = self.routeToSubroute(individual)
        A_Customer = []
        B_Customer = []

        for sub_route in all_sub_route:
            last_customer_id = 0
            sub_time_cost = 0

            time_of_route_customer = []

            for customer_id in sub_route:
                customer_satisfaction = 0
                customer = self.json_instance["customer_" + str(customer_id)]
                distance = self.json_instance["distance_matrix"][last_customer_id][customer_id]

                sub_time_cost = sub_time_cost + self.getTimeCostByInputSpeed(
                    sub_time_cost, distance)

                if self.json_instance[f'customer_{customer_id}']['demand'] > 0:
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

                    if customer['ready_time'] == 0:
                        A_Customer.append(customer_satisfaction)
                    else:
                        B_Customer.append(customer_satisfaction)

                # 加上服务时间
                sub_time_cost += customer['service_time']

                last_customer_id = customer_id

                time_of_route_customer.append(round(sub_time_cost, 2))

            # 回到送货点耗时
            sub_time_cost = sub_time_cost + self.getTimeCostByInputSpeed(
                sub_time_cost, self.json_instance["distance_matrix"][last_customer_id][0])

            time_of_route_customer.append(round(sub_time_cost, 2))
            time_of_route.append(time_of_route_customer)

        # 加权计算平均满意度
        A_Satisfaction = 0
        B_Satisfaction = 0

        for s in A_Customer:
            A_Satisfaction += s
        for s in B_Customer:
            B_Satisfaction += s

        rate_a = 0
        rate_b = 0
        if len(A_Customer) != 0:
            rate_a = 1 / len(A_Customer)
        if len(B_Customer) != 0:
            rate_b = 1 / len(B_Customer)

        # if debug:
        #     print(rate_a, rate_b, A_Satisfaction, B_Satisfaction)

        return round(A_Satisfaction * self.A * rate_a + B_Satisfaction * self.B * rate_b, 2), time_of_route

    # 取送一体问题的混合算子操作
    def opt_pdp(self, ind, gen):
        # 单个算子当前的最多迭代次数
        sn = 1 + floor(20 * (gen / self.num_gen))
        # 包含所有路径的数组
        ind_new = deepcopy(ind)

        count_all = 0  # 所有次数
        count_1 = 0  # relocate 的次数
        count_2 = 0  # insert 的次数
        count_3 = 0  # swap 的次数

        while count_all < self.opt_stop_num:
            while count_1 < sn:
                temp = self.opt_relocate(ind_new)
                if self.evaluate(temp)[0] < self.evaluate(ind_new)[0]:
                    ind_new = temp
                    count_1 = 0
                    count_all = 0
                else:
                    count_1 += 1

            count_all += count_1

            while count_2 < sn:
                temp = self.opt_insert(ind_new)
                if self.evaluate(temp)[0] < self.evaluate(ind_new)[0]:
                    ind_new = temp
                    count_all = 0
                    count_2 = 0
                else:
                    count_2 += 1

            count_all += count_2

            while count_3 < sn:
                temp = self.opt_swap(ind_new)
                if self.evaluate(temp)[0] < self.evaluate(ind_new)[0]:
                    ind_new = temp
                    count_all = 0
                    count_3 = 0
                else:
                    count_3 += 1

            count_all += count_3

        return creator.Individual(ind_new)

    # 随机取两个路径，从其中一个路径中取出一对订单的客户点，随机插入到另一个路径中
    def opt_relocate(self, ind):
        routes = self.routeToSubroutePDP(ind)
        # 挑选路径 a，b
        [a, b] = random.sample(range(len(routes)), 2)

        # print(routes[a], routes[b])

        # 从 a 路径取出客户点
        customer_1 = routes[a][random.randint(0, len(routes[a]) - 1)]
        customer_1_demand = self.json_instance[f'customer_{customer_1}']['demand']
        customer_2 = self.json_instance[f'customer_{customer_1}'][
            'pickup_for' if customer_1_demand < 0 else 'delivery_from']

        routes[a].remove(customer_1)
        routes[a].remove(customer_2)

        # print(customer_1, customer_2,
        #       self.json_instance[f'customer_{customer_1}'])

        # 插入 b 路径
        [j, k] = random.sample(range(len(routes[b]) + 1), 2)
        if j > k:
            j, k = k, j

        routes[b].insert(
            j, (customer_1 if customer_1_demand < 0 else customer_2))
        routes[b].insert(
            k, (customer_2 if customer_1_demand < 0 else customer_1))

        # print(routes[a], routes[b])

        # 可能存在空数组情况
        if routes[a] == []:
            routes.remove([])

        return self.subrouteToRoutePDP(routes)

    # 在同一条路径内，取出一对订单的客户点，随机插入到另一个位置，插入时保证取货点在送货点之前
    def opt_insert(self, ind):
        routes = self.routeToSubroutePDP(ind)
        [a] = random.sample(range(len(routes)), 1)

        # print(routes[a])

        # 从 a 路径取出客户点
        customer_1 = routes[a][random.randint(0, len(routes[a]) - 1)]
        customer_1_demand = self.json_instance[f'customer_{customer_1}']['demand']
        customer_2 = self.json_instance[f'customer_{customer_1}'][
            'pickup_for' if customer_1_demand < 0 else 'delivery_from']

        # print(customer_1, customer_2)

        routes[a][routes[a].index(customer_1)] = 0
        routes[a][routes[a].index(customer_2)] = 0

        [j, k] = random.sample(range(len(routes[a]) + 2), 2)
        if j > k:
            j, k = k, j

        routes[a].insert(
            j, (customer_1 if customer_1_demand < 0 else customer_2))
        routes[a].insert(
            k, (customer_2 if customer_1_demand < 0 else customer_1))

        while 0 in routes[a]:
            routes[a].remove(0)

        # print(routes[a])

        return self.subrouteToRoutePDP(routes)

    # 随机在某条路径中，抽取两对订单的客户点，交换两个点单的对应的取送客户点的位置
    def opt_swap(self, ind):
        routes = self.routeToSubroutePDP(ind)
        [a] = random.sample(range(len(routes)), 1)

        # 选出长度大于 2 的路径
        while len(routes[a]) == 2:
            [a] = random.sample(range(len(routes)), 1)

        # print(routes[a])

        # 从 a 路径取出客户点
        customer_1 = routes[a][random.randint(0, len(routes[a]) - 1)]
        customer_1_demand = self.json_instance[f'customer_{customer_1}']['demand']
        customer_2 = self.json_instance[f'customer_{customer_1}'][
            'pickup_for' if customer_1_demand < 0 else 'delivery_from']

        # 确保 1 是取货，2 是送货
        if customer_1_demand > 0:
            customer_1, customer_2 = customer_2, customer_1

        # 找到另一对客户点
        temp_route = deepcopy(routes[a])
        temp_route.remove(customer_1)
        temp_route.remove(customer_2)
        customer_3 = random.sample(temp_route, 1)[0]
        customer_3_demand = self.json_instance[f'customer_{customer_3}']['demand']
        customer_4 = self.json_instance[f'customer_{customer_3}'][
            'pickup_for' if customer_3_demand < 0 else 'delivery_from']

        # 确保 3 是取货，4 是送货
        if customer_3_demand > 0:
            customer_3, customer_4 = customer_4, customer_3

        # print(customer_1, customer_2, customer_3, customer_4)

        # 交换位置
        e = routes[a].index(customer_1)
        f = routes[a].index(customer_2)
        g = routes[a].index(customer_3)
        h = routes[a].index(customer_4)
        routes[a][e], routes[a][g] = routes[a][g], routes[a][e]
        routes[a][f], routes[a][h] = routes[a][h], routes[a][f]

        # print(routes[a])

        return self.subrouteToRoutePDP(routes)

    # 取送一体问题的路径生成算法
    def routeToSubroutePDP(self, ind):
        # 升序排列
        def cmp(a, b):
            return a['i'] - b['i']

        rtnl_ind = deepcopy(ind)

        max_load = self.json_instance['vehicle_capacity']
        speed = 1

        sub_route = []
        sub_route_load = 0
        sub_route_time_cost = 0
        last_customer_id = 0
        route = []

        while len(rtnl_ind) > 0:
            customer = rtnl_ind[0]
            demand = self.json_instance[f'customer_{customer}']['demand']
            delivery_from = self.json_instance[f'customer_{customer}']['delivery_from']
            pickup_for = self.json_instance[f'customer_{customer}']['pickup_for']
            due_time = self.json_instance[f'customer_{customer}']['due_time']
            service_time = self.json_instance[f'customer_{customer}']['service_time']
            distance = self.json_instance['distance_matrix'][last_customer_id][customer]
            if demand < 0:
                due_time = self.json_instance[f'customer_{pickup_for}']['due_time']

            # 还没取货，送货点放到最后
            if demand > 0 and delivery_from not in sub_route:
                rtnl_ind.insert(len(rtnl_ind), customer)
                # print('1: ', demand, rtnl_ind[0])
                del rtnl_ind[0]
            # 已取货，送货点加入路径
            elif demand > 0 and delivery_from in sub_route:
                sub_route.append(customer)
                sub_route_load += demand
                sub_route_time_cost += distance / speed
                sub_route_time_cost += service_time
                last_customer_id = customer
                rtnl_ind.remove(customer)
                # print('2: ', demand, customer)
            # 满足取货条件（取货客户点、满足车载量、满足右时间窗），取货点加入路径
            elif demand < 0 and (sub_route_load + abs(demand)) < max_load and (sub_route_time_cost + distance / speed) <= due_time:
                sub_route.append(customer)
                sub_route_load += demand
                sub_route_time_cost += distance / speed
                sub_route_time_cost += service_time
                last_customer_id = customer
                rtnl_ind.remove(customer)
                # print('3: ', demand, customer)
            # 路径完成，计算新路径
            else:
                # 缺失的送货点
                missing_customer = []

                for i in sub_route:
                    c = self.json_instance[f'customer_{i}']
                    if c['demand'] < 0 and c['pickup_for'] not in sub_route:
                        missing_customer.append({
                            'c': c['pickup_for'],
                            'i': rtnl_ind.index(c['pickup_for'])
                        })

                # 基于缺失的客户点在个体中的相对顺序排序，index 小的排在前面
                missing_customer.sort(key=cmp_to_key(cmp))

                # 将缺失的客户点加入到路径中
                sub_route += map(lambda a: a['c'], missing_customer)

                # print('4: ', sub_route, missing_customer)

                # 移除客户点
                for i in missing_customer:
                    rtnl_ind.remove(i['c'])

                route.append(sub_route)
                sub_route = []
                sub_route_load = 0
                last_customer_id = 0
                sub_route_time_cost = 0

        if len(sub_route) > 0:
            route.append(sub_route)

        return route

    # pdp 路径转成个体
    def subrouteToRoutePDP(self, routes):
        result = []
        for i in routes:
            result += i
        return result

    # 返回带子路径的二维数组
    def routeToSubroute(self, individual):
        pdp = True  # 取送一体 cvrp 问题
        if pdp:
            return self.routeToSubroutePDP(individual)

        route = []
        sub_route = []
        vehicle_load = 0
        vehicle_capacity = self.json_instance['vehicle_capacity']
        speed = 1

        last_customer_id = 0
        time_cost = 0
        time_gap = 0

        for customer_id in individual:
            distance = self.json_instance["distance_matrix"][last_customer_id][customer_id]
            demand = self.json_instance[f"customer_{customer_id}"]["demand"]
            due_time = self.json_instance[f"customer_{customer_id}"]["due_time"]
            service_time = self.json_instance[f"customer_{customer_id}"]["service_time"]

            updated_vehicle_load = vehicle_load + demand
            time_cost = time_cost + distance / speed

            if updated_vehicle_load <= vehicle_capacity and time_cost <= (due_time + time_gap):
                sub_route.append(customer_id)
                vehicle_load = updated_vehicle_load
                time_cost += service_time
            else:
                route.append(sub_route)
                sub_route = [customer_id]
                vehicle_load = demand
                time_cost = self.json_instance["distance_matrix"][0][customer_id] / \
                    speed + service_time

            last_customer_id = customer_id

        if sub_route != []:
            route.append(sub_route)

        return route

    def printRoute(self, route, merge=False):
        route_str = '0'
        sub_route_count = 0

        time_of_route = self.getSatisfaction(route)[1]
        index = 0

        for sub_route in self.routeToSubroute(route):
            sub_route_count += 1
            sub_route_str = '0'
            for customer_id in sub_route:
                sub_route_str = f'{sub_route_str} - {customer_id}'
                route_str = f'{route_str} - {customer_id}'
            sub_route_str = f'{sub_route_str} - 0'
            if not merge:
                print(
                    f'Vehicle {sub_route_count}\'s route: {sub_route_str}\n\t时间：{time_of_route[index]}')
                index += 1
            route_str = f'{route_str} - 0'
        if merge:
            print(route_str)

    # 计算个体的车辆数

    def getVehicleNum(self, individual):
        all_sub_route = self.routeToSubroute(individual)
        num_of_vehicles = len(all_sub_route)
        return num_of_vehicles

    # 计算个体的距离成本

    def getRouteCost(self, individual):
        all_sub_route = self.routeToSubroute(individual)
        # 所有距离
        all_distance = 0

        for sub_route in all_sub_route:
            sub_route_distance = 0
            # 从配送点出发
            last_customer_id = 0

            for customer_id in sub_route:
                distance = self.json_instance["distance_matrix"][last_customer_id][customer_id]
                sub_route_distance += distance

                # 去往下一个客户点需要的时间
                last_customer_id = customer_id

            # 回到配送点
            sub_route_distance = sub_route_distance + \
                self.json_instance["distance_matrix"][last_customer_id][0]

            # 乘以单位成本，加上早到和迟到惩罚成本

            all_distance += sub_route_distance

        return all_distance

    # 计算适应值
    def evaluate(self, individual):

        # 车辆数
        vehicles = self.getVehicleNum(individual)

        # 总距离
        total_distance = self.getRouteCost(individual)

        # 满意度
        satisfaction = self.getSatisfaction(individual)[0]

        return round(vehicles * 200 + total_distance * 50 + (100 - satisfaction) * 50, 2),

    # 生成 csv 文件

    def doExport(self, times=1):
        csv_file_name = f"{self.json_instance['instance_name']}_result_{'base' if self.base == 1 else 'opt'}_{self.start_time}_{times}.csv"
        csv_columns = self.logbook[0].keys()
        csv_path = os.path.join(BASE_DIR, "results", csv_file_name)

        try:
            with open(csv_path, 'w', encoding='utf8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                writer.writeheader()
                for data in self.logbook:
                    writer.writerow(data)

                csvfile.writelines('\n')
                csvfile.writelines(
                    f'最好的粒子：{self.best_individual}\n')

                csvfile.writelines('\n')

                csvfile.writelines('所有解：\n')
                pops = tools.selBest(self.pop, 120)
                for ind in pops:
                    csvfile.writelines(f'{ind.fitness.values}\n')

        except IOError:
            print("I/O error: ", csv_path, csv_file_name)

    def getCoordinatesDframe(self):
        num_of_cust = self.json_instance['Number_of_customers']
        # Getting all customer coordinates
        customer_list = [i for i in range(1, num_of_cust + 1)]
        x_coord_cust = [
            self.json_instance[f'customer_{i}']['coordinates']['x'] for i in customer_list]
        y_coord_cust = [
            self.json_instance[f'customer_{i}']['coordinates']['y'] for i in customer_list]
        # Getting depot x,y coordinates
        depot_x = [self.json_instance['depart']['coordinates']['x']]
        depot_y = [self.json_instance['depart']['coordinates']['y']]
        # Adding depot details
        customer_list = [0] + customer_list
        x_coord_cust = depot_x + x_coord_cust
        y_coord_cust = depot_y + y_coord_cust
        df = pd.DataFrame({"X": x_coord_cust,
                           "Y": y_coord_cust,
                           "customer_list": customer_list
                           })
        return df

    def plotSubroute(self, subroute, dfhere, color):
        totalSubroute = [0]+subroute+[0]
        subroutelen = len(subroute)
        for i in range(subroutelen+1):
            firstcust = totalSubroute[0]
            secondcust = totalSubroute[1]
            plt.plot([dfhere.X[firstcust], dfhere.X[secondcust]],
                     [dfhere.Y[firstcust], dfhere.Y[secondcust]], c=color)
            totalSubroute.pop(0)

    def plotRoute(self, route, csv_title):
        # Loading the instance

        subroutes = self.routeToSubroute(route)
        colorslist = ["blue", "green", "red", "cyan",
                      "magenta", "yellow", "black", "#f1a03a", "#fbe5e4", "#2568f6"]
        colorindex = 0

        # getting df
        dfhere = self.getCoordinatesDframe()

        # Plotting scatter
        plt.figure(figsize=(10, 10), dpi=144)

        for i in range(dfhere.shape[0]):
            if i == 0:
                plt.scatter(dfhere.X[i], dfhere.Y[i], c='green', s=200)
                plt.text(dfhere.X[i], dfhere.Y[i], "depot", fontsize=12)
            else:
                plt.scatter(dfhere.X[i], dfhere.Y[i], c='orange', s=200)
                plt.text(dfhere.X[i], dfhere.Y[i], f'{i}', fontsize=12)

        # Plotting routes
        for route in subroutes:
            self.plotSubroute(route, dfhere, color=colorslist[colorindex])
            colorindex += 1

        # Plotting is done, adding labels, Title
        plt.xlabel("X - Coordinate")
        plt.ylabel("Y - Coordinate")
        # plt.title(csv_title)
        plt.savefig(f"./figures/Route_{csv_title}.png")

    def plotFitness(self):
        result1 = pd.read_csv('results/a_12_type1.csv')
        result2 = pd.read_csv('results/a_12_type2.csv')
        result3 = pd.read_csv('results/a_12_type2_opt.csv')

        plt.figure(figsize=(10, 10), dpi=144)
        plt.plot(result1['generation'], result1['fitness'])
        plt.plot(result2['generation'], result2['fitness'])
        plt.plot(result3['generation'], result3['fitness'])
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.legend(['GA', 'IGA', 'MA'], loc=0, ncol=2)
        # plt.title('7h')
        plt.xlim(0, 400)
        plt.ylim(6500, 14000)
        plt.savefig(f"./figures/generation_fitness_12h_pdp.png")

    def runMain(self):
        self.init_generation()
        self.runGenerations()
        # self.doExport()
