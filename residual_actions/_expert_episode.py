import dataclasses

import torch


@dataclasses.dataclass(slots=True)
class ExpertEpisode:
    states: torch.Tensor
    action_residuals: torch.Tensor
    actions_indices: torch.Tensor

    def __post_init__(self) -> None:
        assert self.states.shape[0] == self.action_residuals.shape[0] == self.actions_indices.shape[0], \
            (self.states.shape, self.action_residuals.shape, self.actions_indices.shape)

    def __len__(self) -> int:
        return self.states.shape[0]

    def __getitem__(self, item) -> 'ExpertEpisode':
        return ExpertEpisode(
            states=self.states[item],
            action_residuals=self.action_residuals[item],
            actions_indices=self.actions_indices[item],
        )

    def concat(self, new_episode: 'ExpertEpisode'):
        total_episode = ExpertEpisode(
            states=torch.cat((self.states, new_episode.states), dim=0),
            action_residuals=torch.cat((self.action_residuals, new_episode.action_residuals), dim=0),
            actions_indices=torch.cat((self.actions_indices, new_episode.actions_indices), dim=0),
        )
        return total_episode
