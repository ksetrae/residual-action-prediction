import dataclasses

import torch


@dataclasses.dataclass(slots=True)
class ExpertEpisode:
    # TODO: describe dimensions
    state_mus: torch.Tensor
    state_logsigmas: torch.Tensor
    action_residuals: torch.Tensor
    actions_indices: torch.Tensor

    def __post_init__(self) -> None:
        assert self.state_mus.shape[0] == self.state_logsigmas.shape[0] == self.action_residuals.shape[0] == self.actions_indices.shape[0],\
            (self.state_mus.shape, self.state_logsigmas, self.action_residuals.shape, self.actions_indices.shape)

    def __len__(self) -> int:
        return self.state_mus.shape[0]

    def __getitem__(self, item) -> 'ExpertEpisode':
        return ExpertEpisode(
            state_mus=self.state_mus[item],
            state_logsigmas=self.state_logsigmas[item],
            action_residuals=self.action_residuals[item],
            actions_indices=self.actions_indices[item],
        )

    def concat(self, new_episode: 'ExpertEpisode'):
        total_episode = ExpertEpisode(
            state_mus=torch.cat((self.state_mus, new_episode.state_mus), dim=0),
            state_logsigmas=torch.cat((self.state_logsigmas, new_episode.state_logsigmas), dim=0),
            action_residuals=torch.cat((self.action_residuals, new_episode.action_residuals), dim=0),
            actions_indices=torch.cat((self.actions_indices, new_episode.actions_indices), dim=0),
        )
        return total_episode