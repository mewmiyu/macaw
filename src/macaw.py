import utils
import sys
import methods.train as train

if __name__ == '__main__':
    if (len(sys.argv)) != 2:
        print("Failed to load config file.")
        exit(-1)
    cnfg = utils.read_yaml(sys.argv[1])
    match cnfg['METHOD']:
        case 'train':   
            train.train(cnfg)
        case _:
            print(f"Unknown method: {cnfg['METHOD']}. Please use one of the following: train")
            exit(-1)