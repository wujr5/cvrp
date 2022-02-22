from nsga2vrp import *
import argparse
import io
import re
import csv

route = [2, 4, 45, 5, 7, 78, 69, 65, 54, 68, 70, 10, 47, 17, 16, 15, 11, 90, 14, 86, 64, 82, 39, 53, 9, 87, 98, 88, 13, 12, 60, 100, 96, 95, 91, 92, 80, 72, 81, 76, 66, 99, 52, 84, 55, 34, 31, 29, 27,
         26, 28, 30, 32, 50, 62, 67, 71, 33, 59, 24, 75, 18, 63, 57, 97, 74, 58, 77, 25, 51, 89, 19, 49, 20, 22, 48, 21, 23, 56, 85, 94, 93, 41, 40, 43, 44, 42, 35, 37, 38, 83, 61, 3, 1, 36, 73, 79, 8, 46, 6]


def main():

    # Parsing arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--popSize', type=int, default=120, required=False,
                        help="Enter the population size")
    parser.add_argument('--pb', type=float, default=0.02, required=False,
                        help="Mutation Probabilty")
    parser.add_argument('--numGen', type=int, default=1000, required=False,
                        help="Number of generations to run")
    parser.add_argument('--type', type=int, default=1, required=False,
                        help="初始化类型，1 随机，2 指定方向")

    args = parser.parse_args()

    nsgaObj = nsgaAlgo(args.popSize, args.pb, args.numGen, args.type)

    typeObj = {1: '随机', 2: '定向'}
    print(
        f'种群大小：{args.popSize}，变异率：{args.pb}，初始化类型：{typeObj[args.type]}，迭代数：{args.numGen}')

    nsgaObj.runMain()


def run_30_times():

    # typeObj = {1: '随机', 2: '定向'}
    # print(
    #     f'种群大小：{args.popSize}，变异率：{args.pb}，初始化类型：{typeObj[args.type]}，迭代数：{args.numGen}')
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=int, default=1, required=False,
                        help="初始化类型，1 随机，2 指定方向")
    parser.add_argument('--pb', type=float, default=0.02, required=False,
                        help="Mutation Probabilty")
    parser.add_argument('--file', type=str, default='C101', required=False,
                        help="算例")
    parser.add_argument('--base', type=int, default=2, required=False,
                        help="交叉算法：1 基础，2 优化")

    args = parser.parse_args()

    nsgaObj = nsgaAlgo(popSize=120, mutProb=args.pb,
                       numGen=1000, type=args.type, file=args.file, baseAl=args.base)

    for i in range(30):
        print(f'第 {i + 1} 轮')
        nsgaObj.reset()
        nsgaObj.runMain()
        nsgaObj.doExport(i + 1)


def print_route():
    nsgaObj = nsgaAlgo(popSize=120, mutProb=0.02, numGen=1000, type=2)
    nsgaObj.printRoute(nsgaObj.routeToSubroute(route))


def plot_route():
    nsgaObj = nsgaAlgo(popSize=120, mutProb=0.02, numGen=1000, type=2)
    nsgaObj.plotRoute(route, 'RC104')


def plot_fitness():
    nsgaObj = nsgaAlgo(popSize=120, mutProb=0.02, numGen=1000,
                       type=2, file="RC104", baseAl=2)
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
    run_30_times()
    # print_route()
    # plot_route()
    # plot_fitness()
    # parse()
