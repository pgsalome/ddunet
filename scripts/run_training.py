
#import sys
#import os
#sys.path.append(os.getcwd())

from ddunet.utils import load_config
from ddunet.train import train
import argparse

if __name__ == "__main__":
   
    print("\n########################")
    print("DDunet")
    print("########################\n")
    parser = argparse.ArgumentParser(description='DDUnet')
    parser.add_argument('--config', type = str, help = 'Path to the configuration file', required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    train(config)

    
    print('Done')

