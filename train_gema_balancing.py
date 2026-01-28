from benchmarl.benchmark import Benchmark
from benchmarl.algorithms import MappoConfig
from benchmarl.experiment import ExperimentConfig
from benchmarl.models.mlp import MlpConfig

from src.algorithms.gema import GemaConfig
from src.environments.common import GEMATask
from src.models.policies.gema_balancing_gnn_actor import GemaBalancingGnnActorConfig
from src.models.policies.gema_balancing_mlp_critic import GemaBalancingMlpCriticConfig

if __name__ == "__main__":
    experiment_config = ExperimentConfig.get_from_yaml("src/conf/experiment/balancing_experiment.yaml")
    task = GEMATask.LOAD_BALANCING.get_from_yaml("src/conf/task/gema/load_balancing.yaml")
    algorithm_config = GemaConfig.get_from_yaml("src/conf/algorithms/gema_balancing.yaml")
    model_config = GemaBalancingGnnActorConfig.get_from_yaml("src/conf/models/gema_gnn_actor_balancing.yaml")
    critic_model_config = GemaBalancingMlpCriticConfig.get_from_yaml("src/conf/models/gema_mlp_critic_balancing.yaml")

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
