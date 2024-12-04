import copy
import logging
from collections import deque

import torch

from ._settings import ResidualActionsSettings
from ._models import MemoryConditionedBehaviorCloning, MemoryModel
from ._expert_episode import ExpertEpisode

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.DEBUG)


class ResidualActionsLearner:
    """ Residual Actions: https://arxiv.org/pdf/2207.09705.pdf
    """

    def __init__(self,
                 state_space_size: int,
                 action_space_size: int,
                 settings: ResidualActionsSettings,
                 device: str):
        self._main_device = device
        self.settings = settings

        self.memory = MemoryModel(
            state_space_size=state_space_size,
            action_space_size=action_space_size,
            hidden_size=settings.hidden_channels_memory,
        ).to(self._main_device)

        self.behavior = MemoryConditionedBehaviorCloning(
            state_space_size=state_space_size,
            mem_hidden_size=settings.hidden_channels_memory,
            action_space_size=action_space_size,
            curr_hidden_size=settings.hidden_channels_behavior
        ).to(self._main_device)

        self.optimizer = torch.optim.Adam(
            params=list(self.memory.parameters()) + list(self.behavior.parameters()),
            lr=settings.optim_learning_rate)

        self.expert_episode: ExpertEpisode | None = None

        self.history_states = torch.zeros(1,
                                          self.settings.history_size,
                                          state_space_size)

    def add_expert_episode(self,
                           states: torch.Tensor,
                           actions: torch.Tensor) -> None:
        """
        states
            Expected dimensions: (batch, instances, features)
        actions
            Expected dimensions: (batch, binary_multilabel_actions)

        Samples are expected to be ordered by time! (first are earlier)
        """
        action_indices_list = []
        for action in actions:
            action_indices_list.append(
                action.argmax().item()
            )
        actions_indices = torch.Tensor(action_indices_list).to(actions.device).to(torch.float32)

        states_seq = self.make_history_tensor_with_instances(states)
        actions_onehot_seq = self.make_history_tensor(actions).to(torch.float32)
        action_indices_seq = self.make_history_tensor(actions_indices.unsqueeze(-1)).squeeze(-1)

        action_residuals_last = actions_onehot_seq[:, :, -1, ...] - actions_onehot_seq[:, :, -2, ...]
        action_indices_last = action_indices_seq[..., -1]

        new_episode = ExpertEpisode(states=states_seq,
                                    action_residuals=action_residuals_last,
                                    actions_indices=action_indices_last)
        if self.expert_episode is None:
            self.expert_episode = new_episode
        else:
            self.expert_episode = self.expert_episode.concat(new_episode=new_episode)

        return

    def make_history_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        size = self.settings.history_size

        tensor_seq = tensor.clone().unfold(dimension=0, size=size, step=1)  # batch, channels, sequence
        tensor_seq = tensor_seq.permute(0, 2, 1)  # batch, sequence, channels

        return tensor_seq

    def make_history_tensor_with_instances(self, tensor: torch.Tensor) -> torch.Tensor:
        size = self.settings.history_size

        tensor_seq = tensor.clone().unfold(dimension=0, size=size, step=1)  # batch, instances, channels, sequence
        tensor_seq = tensor_seq.permute(0, 1, 3, 2)  # batch, instance, sequence, channels

        return tensor_seq

    def get_state_dicts(self) -> dict[str, dict]:
        return {'memory': self.memory.state_dict(),
                'behavior': self.behavior.state_dict()}

    def set_state_dicts(self, state_dicts: dict[str, dict]) -> None:
        self.memory.load_state_dict(state_dicts['memory'])
        self.behavior.load_state_dict(state_dicts['behavior'])

    def train_epoch(self) -> tuple[float, dict]:
        ep = self.expert_episode

        sample_count = ep.states.shape[0]
        permutation = torch.randperm(sample_count)

        epoch_losses: list[float] = []
        epoch_losses_memory: list[float] = []
        epoch_losses_behavior: list[float] = []
        for i in range(0, sample_count, self.settings.batch_size):
            batch_indices = permutation[i:i + self.settings.batch_size]

            total_input = ep.states[batch_indices]
            actions_residuals = ep.action_residuals[batch_indices]
            actions_indices = ep.actions_indices[batch_indices]

            predicted_residuals, memory_latent = self.memory.forward(total_input)
            # TODO: other representation of negated actions
            batch_memory_loss = torch.nn.functional.mse_loss(predicted_residuals, actions_residuals)

            predicted_actions = self.behavior.forward(observations_current=total_input[:, -1, :],
                                                      history=memory_latent)
            batch_behavior_loss = torch.nn.functional.cross_entropy(predicted_actions, actions_indices.long())

            batch_loss = batch_memory_loss + batch_behavior_loss

            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

            epoch_losses.append(batch_loss.detach().item())
            epoch_losses_memory.append(batch_memory_loss.detach().item())
            epoch_losses_behavior.append(batch_behavior_loss.detach().item())

        loss_val = sum(epoch_losses) / len(epoch_losses)

        # TODO: division by 0 possible
        return loss_val, {'memory': sum(epoch_losses_memory) / len(epoch_losses_memory),
                          'behavior': sum(epoch_losses_behavior) / len(epoch_losses_behavior)}

    def act_and_step(self, points: torch.Tensor) -> int:
        new_states = self.history_states.clone()
        new_states = torch.roll(input=new_states, dims=1, shifts=-1)
        new_states[0, -1, ...] = points
        self.history_states = new_states.clone()

        with torch.no_grad():
            # mu, logsigma = self.memory.encoder(self.history_states.clone())
            # memory_latent = self.memory.reparametrize(mu=mu, logsigma=logsigma)
            memory_latent = self.memory.encoder(self.history_states.clone())

            action_index = self.behavior.act(observations=points.squeeze(0),
                                             history=memory_latent.squeeze(0))
        return action_index

    def train_full(self,
                   running_loss_window_size: int,
                   log_frequency: int,
                   target_loss: float,
                   force_stop_at_plateau_epochs: int,
                   min_epochs: int,
                   grace_epochs_after_min_epochs: int,
                   max_epochs: int | None = None,
                   ) -> float:
        loss = torch.tensor(torch.inf)
        best_running_loss = torch.tensor(torch.inf)
        best_states: dict[str, dict] = self.get_state_dicts()
        stop_training = False
        force_continue_training = False
        epoch_counter = 0
        plateau_counter = 0

        learner_name = self.__class__.__name__

        prev_losses = deque(maxlen=running_loss_window_size)
        while force_continue_training or not stop_training:
            loss_val, loss_info = self.train_epoch()
            epoch_counter += 1

            prev_losses.append(loss_val)
            running_loss = sum(prev_losses) / len(prev_losses)

            if epoch_counter % log_frequency == 0:
                LOGGER.info(f'{learner_name} training: epoch {epoch_counter}; loss: {round(loss_val, 8)}; '
                            f'running mean loss of {running_loss_window_size} size: {round(running_loss, 8)}; '
                            f'\nadditional loss info: {loss_info}.')

            # grace_epochs_after_min_epochs is used so that even if target loss or other condition is reached,
            #  the training process can still try several epochs for picking best state dict
            continue_because_min_epochs = epoch_counter < (min_epochs + grace_epochs_after_min_epochs)
            force_continue_training = continue_because_min_epochs
            if force_continue_training:
                # No need to check stop conditions
                continue

            if epoch_counter > min_epochs:
                # IMPORTANT: This code only runs when we trained for enough minimum epochs because
                # for cases when we are finetuning on new data we might accidentally
                # select a batch with dominantly old data, get low loss on it,
                # and save it as the best state dict
                if running_loss < best_running_loss:
                    plateau_counter = 0
                    best_running_loss = running_loss
                    best_states = copy.deepcopy(self.get_state_dicts())
                else:
                    plateau_counter += 1

            stop_because_plateau = plateau_counter > force_stop_at_plateau_epochs
            if stop_because_plateau:
                LOGGER.info(f"{learner_name} training: plateau reached, running loss: {best_running_loss}")

            stop_because_loss_reached = best_running_loss < target_loss
            if stop_because_loss_reached:
                LOGGER.info(
                    f"{learner_name} training: target loss ({target_loss}) reached, " +
                    f"running loss: {best_running_loss}")

            if max_epochs is None:
                stop_because_max_epochs = False
            else:
                stop_because_max_epochs = epoch_counter >= max_epochs
            if stop_because_max_epochs:
                LOGGER.info(f"{learner_name} training: max epochs ({max_epochs}) reached, " +
                            f"running loss: {best_running_loss}")

            stop_training = stop_because_plateau or stop_because_loss_reached or stop_because_max_epochs

        self.set_state_dicts(state_dicts=best_states)

        return loss.detach().cpu().item()

    def process_and_train_full(self,
                               states_train: torch.Tensor,
                               actions_train: torch.Tensor):
        self.add_expert_episode(states=states_train,
                                actions=actions_train)

        self.train_full(
            running_loss_window_size=self.settings.running_loss_window_size,
            log_frequency=self.settings.train_log_frequency,
            target_loss=self.settings.target_loss,
            force_stop_at_plateau_epochs=self.settings.force_stop_at_plateau_epochs,
            min_epochs=self.settings.min_epochs,
            grace_epochs_after_min_epochs=self.settings.grace_epochs_after_min_epochs,
            max_epochs=self.settings.max_epochs,
        )
