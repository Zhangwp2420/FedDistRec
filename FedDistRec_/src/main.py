from engine import Engine
from dataset import FederatedRecDataset
from logger import get_args
from utils import get_model_class
import json

def main():

    args, _ = get_args()
    dataset = FederatedRecDataset(args.dataset)
    print(json.dumps(vars(args), indent=4, ensure_ascii=False))
    print("-" * 50)
    enginer = Engine(args, dataset, get_model_class(args.model))
    results = enginer.run(args.num_round)
           
    return results


if __name__ == '__main__':
    main()