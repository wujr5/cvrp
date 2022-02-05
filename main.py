from nsga2vrp import *
import argparse


def main():

    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_name', type=str, default="./data/json/C101.json", required=False,
                        help="Enter the input Json file name")
    parser.add_argument('--popSize', type=int, default=180, required=False,
                        help="Enter the population size")
    parser.add_argument('--crossProb', type=float, default=0.9, required=False,
                        help="Crossover Probability")
    parser.add_argument('--mutProb', type=float, default=0.07, required=False,
                        help="Mutation Probabilty")
    parser.add_argument('--numGen', type=int, default=1000, required=False,
                        help="Number of generations to run")

    args = parser.parse_args()

    nsgaObj = nsgaAlgo()

    nsgaObj.pop_size = args.popSize
    nsgaObj.cross_prob = args.crossProb
    nsgaObj.mut_prob = args.mutProb
    nsgaObj.num_gen = args.numGen

    print(
        f'种群大小：{args.popSize}，交叉率：{args.crossProb}，变异率：{args.mutProb}，迭代数：{args.numGen}')

    nsgaObj.runMain()


if __name__ == '__main__':
    main()
