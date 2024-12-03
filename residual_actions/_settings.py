from dataclasses import dataclass


@dataclass
class ResidualActionsSettings:
    history_size: int
    hidden_channels_memory: int
    hidden_channels_behavior: int
    batch_size: int
    optim_learning_rate: float
    target_loss: float
    force_stop_at_plateau_epochs: int
    train_log_frequency: int
    running_loss_window_size: int

    min_epochs: int
    grace_epochs_after_min_epochs: int
    max_epochs: int | None = None
