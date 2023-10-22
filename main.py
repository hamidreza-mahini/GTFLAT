#!/usr/bin/env python
import argparse
from FLAlgorithms.servers.serveravg import FedAvg
from FLAlgorithms.servers.serverFedProx import FedProx
from FLAlgorithms.servers.serverFedDistill import FedDistill
from FLAlgorithms.servers.serverpFedGen import FedGen
from FLAlgorithms.servers.serverpFedEnsemble import FedEnsemble
from utils.model_utils import create_model
from utils.plot_utils import *
import torch
from multiprocessing import Pool

def create_server_n_user(args, i):
    model = create_model(args.model, args.dataset, args.algorithm)
    if ('FedAvg' in args.algorithm):
        server=FedAvg(args, model, i)
    elif 'FedGen' in args.algorithm:
        server=FedGen(args, model, i)
    elif ('FedProx' in args.algorithm):
        server = FedProx(args, model, i)
    elif ('FedDistill' in args.algorithm):
        server = FedDistill(args, model, i)
    elif ('FedEnsemble' in args.algorithm):
        server = FedEnsemble(args, model, i)
    else:
        print("Algorithm {} has not been implemented.".format(args.algorithm))
        exit()
    return server


def run_job(args, i):
    torch.manual_seed(i)
    print("\n\n         [ Start training iteration {} ]           \n\n".format(i))
    # Generate model
    server = create_server_n_user(args, i)
    if args.train:
        server.train(args)
        server.test()

def main(args):
    for i in range(args.times):
        run_job(args, i)
    print("Finished training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #GTFLAT>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    parser.add_argument('-gt', '--gtflat', type=int, choices=[0, 1], default=1,
                        help="Enabling GTFLAT (1)")
    parser.add_argument('-gti','--gtflat_num_iterations', type=int, default=1,
                        help="Evolutionary Game Repetition")
    parser.add_argument('-gtg', '--gtflat_num_generations', type=int, default=50,
                        help="Evolutionary Game Generations")
    parser.add_argument('-gtcli', '--gtflat_compare_layer_index', type=int, default=0,
                        help="Model Compare Index")  
    parser.add_argument('-gtclith','--gtflat_cli_threshold', type=float, default=0.005, 
                        help="The cli threshold")                                                                              
    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    parser.add_argument("--dataset", type=str, default="Mnist-alpha1.0-ratio1")
    parser.add_argument("--model", type=str, default="cnn")
    parser.add_argument("--train", type=int, default=1, choices=[0,1])
    parser.add_argument("--algorithm", type=str, default="GTFedAvg")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gen_batch_size", type=int, default=32, help='number of samples from generator')
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Local learning rate")
    parser.add_argument("--personal_learning_rate", type=float, default=0.01, help="Personalized learning rate to caculate theta aproximately using K steps")
    parser.add_argument("--ensemble_lr", type=float, default=1e-4, help="Ensemble learning rate.")
    parser.add_argument("--beta", type=float, default=1.0, help="Average moving parameter for pFedMe, or Second learning rate of Per-FedAvg")
    parser.add_argument("--lamda", type=int, default=1, help="Regularization term")
    parser.add_argument("--mix_lambda", type=float, default=0.1, help="Mix lambda for FedMXI baseline")
    parser.add_argument("--embedding", type=int, default=0, help="Use embedding layer in generator network")
    parser.add_argument("--num_glob_iters", type=int, default=5)
    parser.add_argument("--local_epochs", type=int, default=20)
    parser.add_argument("--num_users", type=int, default=5, help="Number of Users per round")
    parser.add_argument("--K", type=int, default=1, help="Computation steps")
    parser.add_argument("--times", type=int, default=3, help="running time")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu","cuda"], help="run device (cpu | cuda)")
    parser.add_argument("--result_path", type=str, default="results", help="directory path to save results")

    args = parser.parse_args()

    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm: {}".format(args.algorithm))
    print("Batch size: {}".format(args.batch_size))
    print("Learing rate       : {}".format(args.learning_rate))
    print("Ensemble learing rate       : {}".format(args.ensemble_lr))
    print("Average Moving       : {}".format(args.beta))
    print("Subset of users      : {}".format(args.num_users))
    print("Number of global rounds       : {}".format(args.num_glob_iters))
    print("Number of local rounds       : {}".format(args.local_epochs))
    print("Dataset       : {}".format(args.dataset))
    print("Local Model       : {}".format(args.model))
    print("Device            : {}".format(args.device))
    print("=" * 80)
    main(args)
