from nsga2vrp import *
import argparse
import io
import re
import csv

route1 = [13, 5, 9, 39, 20, 43, 50, 35, 7, 8, 19, 49, 18, 28, 37, 29, 26, 58, 59, 56, 38, 48, 6, 15, 4, 25, 12, 27, 36,
          45, 34, 57, 55, 42, 1, 11, 22, 24, 23, 14, 17, 53, 31, 52, 54, 47, 41, 44, 16, 10, 21, 51, 40, 46, 30, 60, 2, 3, 32, 33]

route2 = [7, 8, 37, 38, 9, 5, 39, 19, 49, 20, 50, 35, 16, 1, 23, 21, 31, 53, 51, 46, 10, 40, 25, 12, 13, 43, 42, 55, 4,
          15, 11, 17, 14, 47, 41, 27, 44, 45, 34, 57, 18, 6, 24, 22, 36, 29, 28, 48, 30, 26, 52, 54, 58, 59, 56, 60, 2, 3, 32, 33]


def main():

    # Parsing arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--popSize', type=int, default=50, required=False,
                        help="Enter the population size")
    parser.add_argument('--pb', type=float, default=0.02, required=False,
                        help="Mutation Probabilty")
    parser.add_argument('--gen', type=int, default=400, required=False,
                        help="Number of generations to run")
    parser.add_argument('--file', type=str, default='lrc104', required=False,
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
                       type=2, file='lrc104', baseAl=2, time=12)
    nsgaObj.printRoute(route1)


def plot_route():
    nsgaObj = nsgaAlgo(popSize=120, mutProb=0.02, numGen=400,
                       type=2, file='lrc104', baseAl=2, time=12)
    nsgaObj.plotRoute(route2, 'a102_18h')


def plot_fitness():
    nsgaObj = nsgaAlgo(popSize=120, mutProb=0.02, numGen=500,
                       type=2, file='a101', baseAl=2, time=12)
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
