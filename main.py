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


if __name__ == '__main__':
    # main()
    run_30_times()
