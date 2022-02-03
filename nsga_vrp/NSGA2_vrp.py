from copy import deepcopy
import os
import io
import random
import numpy
import csv
from functools import cmp_to_key

from json import load, dump
from deap import base, creator, tools

BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

# Load the given problem, which can be a json file


def load_instance(json_file):
    """
    Inputs: path to json file
    Outputs: json file object if it exists, or else returns NoneType
    """
    if os.path.exists(path=json_file):
        with io.open(json_file, 'rt', newline='') as file_object:
            return load(file_object)
    return None

# Take a route of given length, divide it into subroute where each subroute is assigned to vehicle


def routeToSubroute(individual, instance):
    """
    Inputs: Sequence of customers that a route has
            Loaded instance problem
    Outputs: Route that is divided in to subroutes
             which is assigned to each vechicle.
    """
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


def printRoute(route, merge=False):
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


# Calculate the number of vehicles required, given a route
def getNumVehiclesRequired(individual, instance):
    """
    Inputs: Individual route
            Json file object loaded instance
    Outputs: Number of vechiles according to the given problem and the route
    """
    # Get the route with subroutes divided according to demand
    updated_route = routeToSubroute(individual, instance)
    num_of_vehicles = len(updated_route)
    return num_of_vehicles


# Given a route, give its total cost
def getRouteCost(individual, instance, unit_cost=1):
    """
    Inputs : 
        - Individual route
        - Problem instance, json file that is loaded
        - Unit cost for the route (can be petrol etc)

    Outputs:
        - Total cost for the route taken by all the vehicles
    """
    total_cost = 0
    updated_route = routeToSubroute(individual, instance)

    for sub_route in updated_route:
        # Initializing the subroute distance to 0
        sub_route_distance = 0
        # Initializing customer id for depot as 0
        last_customer_id = 0

        for customer_id in sub_route:
            # Distance from the last customer id to next one in the given subroute
            distance = instance["distance_matrix"][last_customer_id][customer_id]
            sub_route_distance += distance
            # Update last_customer_id to the new one
            last_customer_id = customer_id

        # After adding distances in subroute, adding the route cost from last customer to depot
        # that is 0
        sub_route_distance = sub_route_distance + \
            instance["distance_matrix"][last_customer_id][0]

        # Cost for this particular sub route
        sub_route_transport_cost = unit_cost * sub_route_distance

        # Adding this to total cost
        total_cost = total_cost + sub_route_transport_cost

    return total_cost


def getSatisfaction(individual, instance):
    speed = 30  # 行驶速度
    left_edge = 100  # 可容忍早到时间
    right_edge = 100  # 可容忍迟到时间

    total_satisfaction = 0
    updated_route = routeToSubroute(individual, instance)

    for sub_route in updated_route:
        # 记录上一个顾客点 id，默认从配送点出发
        last_customer_id = 0
        # 当前路线的总满意度
        sub_satisfaction = 0
        # 当前路线的耗时
        sub_time_cost = 0

        for customer_id in sub_route:
            # 顾客点
            customer = instance["customer_" + str(customer_id)]
            # 行驶距离
            distance = instance["distance_matrix"][last_customer_id][customer_id]
            # 耗时
            sub_time_cost = sub_time_cost + distance / speed

            # 早到 left_edge 分钟内
            if sub_time_cost >= (customer['ready_time'] - left_edge) and sub_time_cost < customer['ready_time']:
                sub_satisfaction += 100 * \
                    (1 - (customer['ready_time'] - sub_time_cost) / left_edge)
            # 早到
            elif sub_time_cost < customer['ready_time']:
                sub_satisfaction += 0
            # 刚好
            elif sub_time_cost >= customer['ready_time'] and sub_time_cost <= customer['due_time']:
                sub_satisfaction += 100
            # 迟到 right_edge 分钟内
            elif sub_time_cost > customer['due_time'] and sub_time_cost <= (customer['due_time'] + right_edge):
                sub_satisfaction += 100 * \
                    (1 - (sub_time_cost - customer['due_time']) / right_edge)
            # 迟到
            elif sub_time_cost > customer['due_time']:
                sub_satisfaction += 0

            # 加上服务时间
            sub_time_cost += customer['service_time']

            # Update last_customer_id to the new one
            last_customer_id = customer_id

        # Adding this to total cost
        total_satisfaction = total_satisfaction + sub_satisfaction

    return total_satisfaction / instance['Number_of_customers']


# Get the fitness of a given route


def eval_indvidual_fitness(individual, instance, unit_cost):
    """
    Inputs: individual route as a sequence
            Json object that is loaded as file object
            unit_cost for the distance 
    Outputs: Returns a tuple of (Number of vechicles, Route cost from all the vechicles)
    """

    # 用车成本
    vehicles = getNumVehiclesRequired(individual, instance) * 5

    # 路程成本
    route_cost = getRouteCost(individual, instance, unit_cost)

    # 获取满意度
    satisfaction = getSatisfaction(individual, instance)

    return (1 / satisfaction * 100, vehicles + route_cost)

# Crossover method with ordering
# This method will let us escape illegal routes with multiple occurences
#   of customers that might happen. We would never get illegal individual from this
#   crossOver


def cxOrderedVrp(input_ind1, input_ind2):
    # Modifying this to suit our needs
    #  If the sequence does not contain 0, this throws error
    #  So we will modify inputs here itself and then
    #       modify the outputs too

    ind1 = [x-1 for x in input_ind1]
    ind2 = [x-1 for x in input_ind2]
    size = min(len(ind1), len(ind2))
    a, b = random.sample(range(size), 2)
    if a > b:
        a, b = b, a

    # print(f"The cutting points are {a} and {b}")
    holes1, holes2 = [True] * size, [True] * size
    for i in range(size):
        if i < a or i > b:
            holes1[ind2[i]] = False
            holes2[ind1[i]] = False

    # We must keep the original values somewhere before scrambling everything
    temp1, temp2 = ind1, ind2
    k1, k2 = b + 1, b + 1
    for i in range(size):
        if not holes1[temp1[(i + b + 1) % size]]:
            ind1[k1 % size] = temp1[(i + b + 1) % size]
            k1 += 1

        if not holes2[temp2[(i + b + 1) % size]]:
            ind2[k2 % size] = temp2[(i + b + 1) % size]
            k2 += 1

    # Swap the content between a and b (included)
    for i in range(a, b + 1):
        ind1[i], ind2[i] = ind2[i], ind1[i]

    # Finally adding 1 again to reclaim original input
    ind1 = [x+1 for x in ind1]
    ind2 = [x+1 for x in ind2]
    return ind1, ind2


def mutationShuffle(individual, indpb):
    """
    Inputs : Individual route
             Probability of mutation betwen (0,1)
    Outputs : Mutated individual according to the probability
    """
    size = len(individual)
    for i in range(size):
        if random.random() < indpb:
            swap_indx = random.randint(0, size - 2)
            if swap_indx >= i:
                swap_indx += 1
            individual[i], individual[swap_indx] = \
                individual[swap_indx], individual[i]

    return individual,


## Statistics and Logging

def createStatsObjs():
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


def recordStat(invalid_ind, logbook, pop, stats, gen):
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
        f'迭代：{gen}，满意度：{best_individual.fitness.values[0]}，成本：{best_individual.fitness.values[1]}')

# Exporting CSV files


def exportCsv(csv_file_name, logbook):
    csv_columns = logbook[0].keys()
    csv_path = os.path.join(BASE_DIR, "results", csv_file_name)
    try:
        with open(csv_path, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in logbook:
                writer.writerow(data)
    except IOError:
        print("I/O error")


class nsgaAlgo(object):

    def __init__(self):
        self.json_instance = load_instance('./data/json/C101.json')
        self.ind_size = self.json_instance['Number_of_customers']
        self.pop_size = 400
        self.cross_prob = 0.85
        self.mut_prob = 0.02
        self.num_gen = 150
        self.toolbox = base.Toolbox()
        self.logbook, self.stats = createStatsObjs()
        self.createCreators()

    def createCreators(self):
        creator.create('FitnessMin', base.Fitness, weights=(-1.0, -1.0))
        creator.create('Individual', list, fitness=creator.FitnessMin)

        # 初始化种群方式：随机
        # self.toolbox.register('indexes', random.sample, range(
        #     1, self.ind_size + 1), self.ind_size)

        # 初始化种群方式：随机抽取顾客点后，根据距离长短
        self.toolbox.register('indexes', self.initPopulation, range(
            1, self.ind_size + 1), self.ind_size)

        # Creating individual and population from that each individual
        self.toolbox.register('individual', tools.initIterate,
                              creator.Individual, self.toolbox.indexes)
        self.toolbox.register('population', tools.initRepeat,
                              list, self.toolbox.individual)

        # Creating evaluate function using our custom fitness
        #   toolbox.register is partial, *args and **kwargs can be given here
        #   and the rest of args are supplied in code
        self.toolbox.register('evaluate', eval_indvidual_fitness,
                              instance=self.json_instance, unit_cost=1)

        # Selection method
        self.toolbox.register("select", tools.selNSGA2)

        # Crossover method
        self.toolbox.register("mate", cxOrderedVrp)

        # Mutation method
        self.toolbox.register("mutate", mutationShuffle, indpb=self.mut_prob)

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

    def generatingPopFitness(self):
        self.pop = self.toolbox.population(n=self.pop_size)
        self.invalid_ind = [ind for ind in self.pop if not ind.fitness.valid]
        self.fitnesses = list(map(self.toolbox.evaluate, self.invalid_ind))

        for ind, fit in zip(self.invalid_ind, self.fitnesses):
            ind.fitness.values = fit

        self.pop = self.toolbox.select(self.pop, len(self.pop))

        recordStat(self.invalid_ind, self.logbook, self.pop, self.stats, gen=0)

    def runGenerations(self):
        # Running algorithm for given number of generations
        for gen in range(self.num_gen):

            # Selecting individuals
            # Selecting offsprings from the population, about 1/2 of them
            self.offspring = tools.selTournamentDCD(self.pop, len(self.pop))
            self.offspring = [self.toolbox.clone(
                ind) for ind in self.offspring]

            # Performing , crossover and mutation operations according to their probabilities
            for ind1, ind2 in zip(self.offspring[::2], self.offspring[1::2]):
                # Mating will happen 80% of time if cross_prob is 0.8
                if random.random() <= self.cross_prob:
                    # print("Mating happened")
                    self.toolbox.mate(ind1, ind2)

                    # If cross over happened to the individuals then we are deleting those individual
                    #   fitness values, This operations are being done on the offspring population.
                    del ind1.fitness.values, ind2.fitness.values
                self.toolbox.mutate(ind1)
                self.toolbox.mutate(ind2)

            # Calculating fitness for all the invalid individuals in offspring
            self.invalid_ind = [
                ind for ind in self.offspring if not ind.fitness.valid]
            self.fitnesses = self.toolbox.map(
                self.toolbox.evaluate, self.invalid_ind)
            for ind, fit in zip(self.invalid_ind, self.fitnesses):
                ind.fitness.values = fit

            # Recalcuate the population with newly added offsprings and parents
            # We are using NSGA2 selection method, We have to select same population size
            self.pop = self.toolbox.select(
                self.pop + self.offspring, self.pop_size)

            # Recording stats in this generation
            recordStat(self.invalid_ind, self.logbook,
                       self.pop, self.stats, gen + 1)

    def getBestInd(self):
        self.best_individual = tools.selBest(self.pop, 1)[0]

        # Printing the best after all generations
        print(f"最好的粒子：{self.best_individual}")
        print(f"满意度：{self.best_individual.fitness.values[0]}")
        print(f"总成本：{self.best_individual.fitness.values[1]}")

        # Printing the route from the best individual
        printRoute(routeToSubroute(self.best_individual, self.json_instance))

    def doExport(self):
        csv_file_name = f"{self.json_instance['instance_name']}_" \
                        f"pop{self.pop_size}_crossProb{self.cross_prob}" \
                        f"_mutProb{self.mut_prob}_numGen{self.num_gen}.csv"
        exportCsv(csv_file_name, self.logbook)

    def runMain(self):
        self.generatingPopFitness()
        self.runGenerations()
        self.getBestInd()
        self.doExport()
