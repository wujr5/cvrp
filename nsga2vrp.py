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
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class nsgaAlgo():

    def __init__(self, popSize, mutProb, numGen, type, file, baseAl=2):
        self.file = file
        self.base = baseAl
        self.json_instance = self.load_instance(
            f'./data/json/{self.file}.json')

        self.speed = self.load_speed('./data/speed.csv')
        self.ind_size = self.json_instance['Number_of_customers']

        length = len(self.json_instance['distance_matrix'][0])
        for i in range(length):
            for j in range(length):
                self.json_instance['distance_matrix'][i][j] *= 2

        self.pop_size = popSize
        self.mut_prob = mutProb
        self.num_gen = numGen
        self.type = type
        self.toolbox = base.Toolbox()

        self.A = 0.5  # 顾客时间窗左端（即 ready time）=0
        self.B = 1 - self.A  # 其余则为B类，A + B = 1

        # 车辆出发原点时间，0，表示 6:00，1 单位时间是 1 分钟
        self.start_time = 11
        self.original_time = (self.start_time - 6) * 60  # 设置为下午 13:00 出发

        self.logbook = tools.Logbook()
        self.logbook.header = "generation", "fitness"

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

        # self.rationalize(result)

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
            self.parents = tools.selBest(self.pop, 90)

            self.offsprings = []

            # 交配与变异操作
            for i in range(90):
                select = random.sample(list(range(90)), 2)
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

            for i in range(len(self.offsprings)):
                # 2-opt操作
                # if self.type != 1:
                # self.offsprings[i] = self.operate2opt(self.offsprings[i])
                # 重新计算适应值
                self.offsprings[i].fitness.values = self.toolbox.evaluate(
                    self.offsprings[i])

            # 使用 nsga2 算法，重新选择种群
            self.pop = self.toolbox.select(
                self.pop + self.offsprings, self.pop_size)

            self.best_individual = tools.selBest(self.pop, 1)[0]

            print(
                f'迭代：{gen + 1}，类型：{type_config[self.type]}，适应值：{self.best_individual.fitness.values}，车辆：{self.getVehicleNum(self.best_individual)}，距离：{self.getDistance(self.best_individual)}，满意度：{self.getSatisfaction(self.best_individual, True)}')

            # 生成日志
            self.logbook.record(
                generation=gen + 1, fitness=f'{self.best_individual.fitness.values}')
        # print(
        #     f'变异：{self.mut_prob}，类型：{type_config[self.type]}，车辆：{self.best_individual.fitness.values[0]}，距离：{self.best_individual.fitness.values[1]}')

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

        return round(all_distance, 3)

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

    def rationalize(self, ind):
        rtnl_ind = deepcopy(ind)

        print(rtnl_ind)

        for i in rtnl_ind:
            print(i, self.json_instance[f'customer_{i}'])

        exit()

        return creator.Individual(list(rtnl_ind))

    # 根据输入的速度数据获取消耗的时间
    def getTimeCostByInputSpeed(self, startTime, distance):
        # 加上出发时间
        t = startTime + self.original_time

        d = distance / 10
        find_d = 0
        time_cost = 1

        while find_d < d:
            find_d = self.speed[t + time_cost] - \
                self.speed[float(t)]
            time_cost += 1

        return time_cost

    # 满意度函数
    def getSatisfaction(self, individual, debug=False):
        left_edge = 10  # 可容忍早到时间
        right_edge = 10  # 可容忍迟到时间

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
                sub_time_cost += customer['service_time']

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

        rate_a = 0
        rate_b = 0
        if len(A_Customer) != 0:
            rate_a = 1 / len(A_Customer)
        if len(B_Customer) != 0:
            rate_b = 1 / len(B_Customer)

        # if debug:
        #     print(rate_a, rate_b, A_Satisfaction, B_Satisfaction)

        return round(A_Satisfaction * self.A * rate_a + B_Satisfaction * self.B * rate_b, 2)

    # 返回带子路径的二维数组
    def routeToSubroute(self, individual):
        route = []
        sub_route = []
        vehicle_load = 0
        vehicle_capacity = self.json_instance['vehicle_capacity']
        speed = 1

        last_customer_id = 0
        time_cost = 0
        time_gap = 10

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
        satisfaction = self.getSatisfaction(individual)

        return round(vehicles * 20 + total_distance * 2 + (100 - satisfaction) * 10, 4),

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
        plt.title(csv_title)
        plt.savefig(f"./figures/Route_{csv_title}.png")

    def plotFitness(self):
        result1 = pd.read_csv('results/result_type_7h_01.csv')
        # result2 = pd.read_csv('results/result_type_7h_02.csv')
        result3 = pd.read_csv('results/result_type_7h_03.csv')

        plt.figure(figsize=(10, 10), dpi=144)
        plt.plot(result1['index'], result1['fitness'])
        # plt.plot(result2['index'], result2['fitness'])
        plt.plot(result3['index'], result3['fitness'])
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        # plt.title('7h')
        plt.xlim(0, 800)
        plt.ylim(4000, 18000)
        plt.savefig(f"./figures/generation_fitness_7h.png")

    def runMain(self):
        self.init_generation()
        self.runGenerations()
        # self.doExport()
