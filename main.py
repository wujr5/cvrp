from nsga2vrp import *
import argparse


def main():

    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_name', type=str, default="./data/json/C101.json", required=False,
                        help="Enter the input Json file name")
    parser.add_argument('--popSize', type=int, default=200, required=False,
                        help="Enter the population size")
    parser.add_argument('--mutProb', type=float, default=0.02, required=False,
                        help="Mutation Probabilty")
    parser.add_argument('--numGen', type=int, default=1000, required=False,
                        help="Number of generations to run")

    args = parser.parse_args()

    nsgaObj = nsgaAlgo(args.popSize, args.mutProb, args.numGen)

    print(
        f'种群大小：{args.popSize}，变异率：{args.mutProb}，迭代数：{args.numGen}')

    nsgaObj.runMain()


if __name__ == '__main__':
    main()
