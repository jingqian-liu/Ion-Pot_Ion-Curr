import os
from datetime import datetime

from nntool.slurm import SlurmConfig
from nntool.wandb import WandbConfig
from nntool.experiment import read_toml_file, get_output_path

env_file = "./env.toml"
env = read_toml_file(env_file)
project_path = env["project"]["path"]
dataset_path = f"{project_path}/datasets"

# nntool expects both env vars when NNTOOL_OUTPUT_PATH is pre-set.
if "NNTOOL_OUTPUT_PATH" in os.environ and "NNTOOL_OUTPUT_PATH_DATE" not in os.environ:
    os.environ["NNTOOL_OUTPUT_PATH_DATE"] = datetime.now().strftime("%m%d%Y/%H%M%S")

output_path, current_time = get_output_path(f"{project_path}/outputs")


slurm = SlurmConfig(
    mode=env["slurm"]["mode"],
    partition=env["slurm"]["partition"],
    job_name=env["slurm"]["job_name"],
    output_folder=f"{output_path}/slurm",
    tasks_per_node=1,
    cpus_per_task=16,
    gpus_per_node=1,
    mem="256G",
    node_list=env["slurm"]["node_list"],
    use_distributed_env=False,
)

atomic_slurm = SlurmConfig(
    mode=env["slurm"]["mode"],
    partition=env["slurm"]["partition"],
    job_name=env["slurm"]["job_name"],
    output_folder=f"{output_path}/slurm",
    tasks_per_node=1,
    cpus_per_task=16,
    gpus_per_node=1,
    mem="128G",
    node_list=env["slurm"]["node_list"],
    use_distributed_env=False,
)

distributed_slurm = SlurmConfig(
    mode=env["slurm"]["mode"],
    partition=env["slurm"]["partition"],
    job_name=env["slurm"]["job_name"],
    output_folder=f"{output_path}/slurm",
    tasks_per_node=1,
    cpus_per_task=16,
    gpus_per_node=4,
    mem="384G",
    node_list=env["slurm"]["node_list"],
    use_distributed_env=True,
    processes_per_task=4,
    distributed_launch_command="accelerate launch --config_file configs/accelerate/accelerate.yaml --num_processes {num_processes} --num_machines {num_machines} --machine_rank {machine_rank} --main_process_ip {main_process_ip} --main_process_port {main_process_port} -m scripts.3d.train_3d_energy_distributed",
)

distributed_eval_slurm = SlurmConfig(
    mode=env["slurm"]["mode"],
    partition=env["slurm"]["partition"],
    job_name=env["slurm"]["job_name"],
    output_folder=f"{output_path}/slurm",
    tasks_per_node=1,
    cpus_per_task=24,
    gpus_per_node=4,
    mem="60G",
    node_list=env["slurm"]["node_list"],
    use_distributed_env=True,
    processes_per_task=4,
    distributed_launch_command="accelerate launch --config_file configs/accelerate/accelerate.yaml --num_processes {num_processes} --num_machines {num_machines} --machine_rank {machine_rank} --main_process_ip {main_process_ip} --main_process_port {main_process_port} -m scripts.3d.test_3d_energy_distributed",
)

debug_distributed_slurm = SlurmConfig(
    mode=env["slurm"]["mode"],
    partition=env["slurm"]["partition"],
    job_name=env["slurm"]["job_name"],
    output_folder=f"{output_path}/slurm",
    tasks_per_node=1,
    cpus_per_task=8,
    gpus_per_node=4,
    mem="60G",
    node_list=env["slurm"]["node_list"],
    use_distributed_env=True,
    processes_per_task=4,
    distributed_launch_command="accelerate launch --config_file configs/accelerate/accelerate.yaml --num_processes {num_processes} --num_machines {num_machines} --machine_rank {machine_rank} --main_process_ip {main_process_ip} --main_process_port {main_process_port} -m scripts.3d.train_3d_energy_distributed",
)

debug_slurm = SlurmConfig(
    mode=env["slurm"]["mode"],
    partition=env["slurm"]["partition"],
    job_name=env["slurm"]["job_name"],
    output_folder=f"{output_path}/slurm",
    tasks_per_node=1,
    cpus_per_task=8,
    gpus_per_node=1,
    mem="60G",
    node_list=env["slurm"]["node_list"],
    use_distributed_env=False,
)

wandb = WandbConfig(
    project=env["wandb"]["project"],
    entity=env["wandb"]["entity"],
    name=f"{current_time}",
    api_key_config_file=env_file,
)
