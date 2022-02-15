from nsga2vrp import *
import argparse


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

    nsgaObj = nsgaAlgo(popSize=120, mutProb=0.02, numGen=1000, type=2)

    for i in range(30):
        print(f'第 {i + 1} 轮')
        nsgaObj.reset()
        nsgaObj.runMain()
        nsgaObj.doExport(i + 1)


def print_route():

    route = [71, 92, 95, 64, 91, 54, 80, 94, 62, 67, 87, 9, 55, 60, 100, 82, 12, 78, 70, 96, 31, 40, 44, 42, 61, 81, 39, 37, 35, 5, 73, 83, 38, 79, 6, 4, 66, 72, 41, 43, 14, 47, 17, 16, 15, 13, 11, 53, 68,
             21, 20, 25, 10, 57, 59, 97, 7, 8, 36, 1, 3, 45, 46, 2, 69, 23, 99, 56, 86, 52, 22, 48, 89, 75, 93, 74, 58, 77, 65, 88, 98, 90, 85, 63, 26, 27, 28, 30, 32, 33, 76, 84, 50, 34, 29, 24, 49, 19, 18, 51]

    nsgaObj = nsgaAlgo(popSize=120, mutProb=0.02, numGen=1000, type=2)
    nsgaObj.printRoute(nsgaObj.routeToSubroute(route))


def plot_route():
    route = [71, 92, 95, 64, 91, 54, 80, 94, 62, 67, 87, 9, 55, 60, 100, 82, 12, 78, 70, 96, 31, 40, 44, 42, 61, 81, 39, 37, 35, 5, 73, 83, 38, 79, 6, 4, 66, 72, 41, 43, 14, 47, 17, 16, 15, 13, 11, 53, 68,
             21, 20, 25, 10, 57, 59, 97, 7, 8, 36, 1, 3, 45, 46, 2, 69, 23, 99, 56, 86, 52, 22, 48, 89, 75, 93, 74, 58, 77, 65, 88, 98, 90, 85, 63, 26, 27, 28, 30, 32, 33, 76, 84, 50, 34, 29, 24, 49, 19, 18, 51]

    nsgaObj = nsgaAlgo(popSize=120, mutProb=0.02, numGen=1000, type=2)
    nsgaObj.plotRoute(route, 'RC104')


if __name__ == '__main__':
    # main()
    # run_30_times()
    # print_route()
    plot_route()
