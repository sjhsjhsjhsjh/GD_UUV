import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from rich.console import Console
from env.env import Env

@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="main_config",
)
def main(cfg: DictConfig) -> None:
    """Run the full GD-UUV training pipeline.

        Args:
                cfg: Hydra configuration object loaded from ``configs/dqn_train_config.yaml``.

        Returns:
                None.

        Example:
                Run directly with default config:
                python run.py
        """
        # console = Console()
    env = Env(cfg)

if __name__ == "__main__":
        main()
