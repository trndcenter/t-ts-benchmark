import nip
import os

from src.benchmarks.iofrol_gsdtsr_model import IofrolGsdtsrModel
from src.benchmarks.redh_model import RedhModel


if __name__ == "__main__":
    for dataset in ['iofrol', 'gsdtsr', 'redh']:
        config  = nip.parse(f'configs/benchmarks/{dataset}.yaml')
        if not os.path.isfile(config['model']['path_dataset'].to_python()):
            print(f"Dataset {dataset} not found")
            continue
        creator = nip.construct(config['model'])
        creator.train_test()
        print(f"Dataset: {dataset}")
        if dataset == 'redh':
            metrics = creator.find_metrics(threshold=0.159)
            for key in metrics:
                print(f"{key}: {metrics[key]}")
        else:
            print(f"APFD: {creator.find_apfd()}")
            print(f"NAPFD (50%): {creator.find_apfd(ratio=0.5)}")
            print()
            print("Time metrics")
            time_metrics = creator.find_time()
            for key in time_metrics:
                print(f"{key}: {time_metrics[key]}")
        print("--------------------------")

        

