from benchmarl.benchmark import Benchmark
from benchmarl.algorithms import MappoConfig
from benchmarl.experiment import ExperimentConfig
from benchmarl.models.mlp import MlpConfig
from src.environments.common import GEMATask

if __name__ == "__main__":
    experiment_config = ExperimentConfig.get_from_yaml("src/conf/experiment/base_experiment.yaml")
    task = GEMATask.SPREAD.get_from_yaml("src/conf/task/gema/spread.yaml")
    algorithm_config = MappoConfig.get_from_yaml()
    model_config = MlpConfig.get_from_yaml()
    critic_model_config = MlpConfig.get_from_yaml()

    benchmark = Benchmark(
        algorithm_configs=[
            MappoConfig.get_from_yaml(),
        ],
        tasks=[
            task
        ],
        seeds={0},
        experiment_config=experiment_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
    )

    benchmark.run_sequential()
