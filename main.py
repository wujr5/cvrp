from nsga2vrp import *
import argparse
import io
import re
import csv

route1 = [[17, 14, 11, 47, 41, 8, 19, 49, 44, 26, 38, 56],
          [5, 27, 12, 13, 7, 20, 43, 42, 50, 35, 57, 37],
          [4, 15, 25, 9, 39, 45, 34, 30, 60, 55],
          [16, 6, 29, 36, 18, 28, 48, 46, 24, 58, 54, 59],
          [1, 22, 31, 10, 40, 23, 21, 52, 53, 51],
          [2, 3, 32, 33]]

route2 = [[25, 5, 27, 12, 13, 20, 43, 50, 35, 42, 55, 57],
          [10, 24, 22, 40, 23, 54, 21, 52, 53, 51],
          [1, 31, 17, 14, 11, 15, 47, 41, 44, 45],
          [9, 8, 39, 19, 49, 26, 38, 56],
          [4, 6, 34, 36, 18, 28, 48, 30, 60, 29, 59, 58],
          [2, 3, 32, 33, 7, 37, 16, 46]]


def main():

    # Parsing arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--popSize', type=int, default=50, required=False,
                        help="Enter the population size")
    parser.add_argument('--pb', type=float, default=0.02, required=False,
                        help="Mutation Probabilty")
    parser.add_argument('--gen', type=int, default=400, required=False,
                        help="Number of generations to run")
    parser.add_argument('--file', type=str, default='a102', required=False,
                        help="算例")
    parser.add_argument('--type', type=int, default=2, required=False,
                        help="初始化类型，1 随机，2 指定方向")
    parser.add_argument('--base', type=int, default=2, required=False,
                        help="交叉算法：1 基础，2 优化")
    parser.add_argument('--time', type=int, default=12, required=False,
                        help="开始时间")

    args = parser.parse_args()

    nsgaObj = nsgaAlgo(popSize=args.popSize, mutProb=args.pb,
                       numGen=args.gen, type=args.type, file=args.file, baseAl=args.base, time=args.time)

    typeObj = {1: '随机', 2: '定向'}
    print(
        f'种群大小：{args.popSize}，变异率：{args.pb}，类型：{typeObj[args.type]}，迭代数：{args.gen}，时间：{args.time}')

    nsgaObj.runMain()

    # 保存距离文件为 csv
    # distance = nsgaObj.json_instance['distance_matrix']
    # with open('distance-a102.csv', 'w', encoding='utf8') as csvfile:
    #     csvfile.writelines(
    #         'distance,' + ','.join(map(str, range(0, nsgaObj.ind_size + 1))) + '\n')

    #     for i in range(len(distance)):
    #         csvfile.writelines(
    #             str(i) + ',' + ','.join(map(str, distance[i])) + '\n')


def run_30_times():

    # typeObj = {1: '随机', 2: '定向'}
    # print(
    #     f'种群大小：{args.popSize}，变异率：{args.pb}，初始化类型：{typeObj[args.type]}，迭代数：{args.numGen}')
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=int, default=1, required=False,
                        help="初始化类型，1 随机，2 指定方向")
    parser.add_argument('--pb', type=float, default=0.02, required=False,
                        help="Mutation Probabilty")
    parser.add_argument('--file', type=str, default='RC104', required=False,
                        help="算例")
    parser.add_argument('--base', type=int, default=2, required=False,
                        help="交叉算法：1 基础，2 优化")
    parser.add_argument('--time', type=int, default=12, required=False,
                        help="开始时间")
    parser.add_argument('--gen', type=int, default=800, required=False,
                        help="Number of generations to run")

    args = parser.parse_args()

    nsgaObj = nsgaAlgo(popSize=120, mutProb=args.pb,
                       numGen=args.gen, type=args.type, file=args.file, baseAl=args.base, time=args.time)

    for i in range(30):
        print(f'第 {i + 1} 轮')
        nsgaObj.reset()
        nsgaObj.runMain()
        nsgaObj.doExport(i + 1)


def print_route():
    nsgaObj = nsgaAlgo(popSize=120, mutProb=0.02, numGen=400,
                       type=2, file='a102', baseAl=2, time=18)
    nsgaObj.printRoute(route1)
    print(nsgaObj.getSatisfaction(route1))


def plot_route():
    nsgaObj = nsgaAlgo(popSize=120, mutProb=0.02, numGen=400,
                       type=2, file='a102', baseAl=2, time=12)
    nsgaObj.plotRoute(route2, 'a102_0303_02')


def plot_fitness():
    nsgaObj = nsgaAlgo(popSize=120, mutProb=0.02, numGen=500,
                       type=2, file='a102', baseAl=2, time=12)
    nsgaObj.plotFitness()


def parse():
    with io.open('results/result_15h_03.csv', 'rt', newline='', encoding='utf8') as f:
        lines = f.readlines()
        all_newlines = []
        for l in lines:
            l = re.sub(r'\n', '', l)
            row = l.split('，')

            newline = []
            for r in row:
                newline.append(re.sub(r'\(|\)|,', '', r.split('：')[1]))
            all_newlines.append(newline)

        with open('results/result_type_15h_03.csv', 'w', encoding='utf8') as csvfile:
            csvfile.writelines(
                'index,type,fitness,vehicle,distance,satisfaction\n')
            for data in all_newlines:
                csvfile.writelines(','.join(data) + '\n')


if __name__ == '__main__':
    # main()
    # run_30_times()
    # print_route()
    plot_route()
    # plot_fitness()
    # parse()
