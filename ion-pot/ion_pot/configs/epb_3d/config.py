import tyro

from .experiment_config import ExperimentConfig, EvalExperimentConfig
from .experiment import experiments, eval_experiments
from .experiment_unet import experiments as unet_experiments
from .experiment_atomic import experiments as atomic_experiments

all_experiments = {
    **atomic_experiments,
    **unet_experiments,
}
all_eval_experiments = {**eval_experiments, **atomic_experiments}
ConfiguredExperimentConfig = tyro.extras.subcommand_type_from_defaults(all_experiments)
ConfiguredEvalExperimentConfig = tyro.extras.subcommand_type_from_defaults(
    all_eval_experiments
)
