from benchmarl.benchmark import Benchmark
from benchmarl.experiment import ExperimentConfig

from src.algorithms.gema import GemaConfig
from src.environments.common import GEMATask
from src.models.policies.gema_spread_gnn_actor import GemaSpreadGnnActorConfig
from src.models.policies.gema_spread_mlp_critic import GemaMlpCriticConfig

if __name__ == "__main__":
    experiment_config = ExperimentConfig.get_from_yaml("src/conf/experiment/spread_experiment.yaml")
    task = GEMATask.SPREAD.get_from_yaml("src/conf/task/gema/spread.yaml")
    algorithm_config = GemaConfig.get_from_yaml("src/conf/algorithms/gema_spread.yaml")
    model_config = GemaSpreadGnnActorConfig.get_from_yaml("src/conf/models/gema_gnn_actor_spread.yaml")
    critic_model_config = GemaMlpCriticConfig.get_from_yaml("src/conf/models/gema_mlp_critic_spread.yaml")

    benchmark = Benchmark(
        algorithm_configs=[
            algorithm_config,
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
