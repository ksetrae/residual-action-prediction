import typing
import copy

import torch
import numpy as np

from feedback_core.tabula_rasa.train_handling.learners import BaseLearner
from feedback_core.tabula_rasa.external.pisa.sae import PISA
from feedback_core.tabula_rasa.train_handling import TrainingProcessor
from feedback_core.configs.misc import MP_LOGGER
from feedback_core.tabula_rasa.train_handling import InterProcessQueues, InterProcessSharedDicts

from feedback_core.tabula_rasa.config_def import VAEResActionsSettingsDefinition
from ._model import MemoryConditionedBehaviorCloning, MemoryModel
from ._expert_episode import ExpertEpisode


class VAEResActionLearner(BaseLearner):
    """ Residual Actions (https://arxiv.org/pdf/2207.09705.pdf), while states are encoded with VAE
        (https://arxiv.org/abs/1312.6114), VAE pretrained on random data,
        like in DITTO (https://arxiv.org/abs/2302.03086) and World Models (https://arxiv.org/abs/1803.10122)
    """

    def __init__(self,
                 train_processor: TrainingProcessor,
                 autoencoder: PISA,
                 settings: VAEResActionsSettingsDefinition,
                 device: str):
        raise NotImplementedError("Discontinuity will be introduced. Currently this learner can't add data with"
                                  "regard to that. ")

        self._train_processor = train_processor
        self._main_device = device
        self.settings = settings

        self.history_elem_count = (self._train_processor.processing_settings.true_history_frame_len //
                                   self._train_processor.processing_settings.frame_seq_len_screening_rate)

        self.memory = MemoryModel(
            state_space_size=self._train_processor.state_autoencoder_settings.latent_space_size,
            action_space_size=self._train_processor.inputs_permutations_count,
            hidden_size=settings.hidden_channels_memory,
            proc_settings=self._train_processor.processing_settings
        ).to(self._main_device)

        self.behavior = MemoryConditionedBehaviorCloning(
            state_space_size=self._train_processor.state_autoencoder_settings.latent_space_size,
            mem_hidden_size=settings.hidden_channels_memory,
            action_space_size=self._train_processor.inputs_permutations_count,
            curr_hidden_size=settings.hidden_channels_behavior
        ).to(self._main_device)

        self.optimizer = torch.optim.Adam(
            params=list(self.memory.parameters()) + list(self.behavior.parameters()),
            lr=settings.optim_learning_rate)

        self.autoencoder = autoencoder
        self.expert_episode: ExpertEpisode | None = None

        self.history_states = torch.zeros(1,
                                          self.history_elem_count,
                                          self._train_processor.state_autoencoder_settings.latent_space_size)

    def put_current_behavior(self, queues: InterProcessQueues) -> None:
        state_dict_task = copy.deepcopy(self.memory.state_dict())
        for key in state_dict_task:
            state_dict_task[key] = state_dict_task[key].to('cpu')
        queues.controller_memory.put(state_dict_task)

        state_dict_behavior = copy.deepcopy(self.behavior.state_dict())
        for key in state_dict_behavior:
            state_dict_behavior[key] = state_dict_behavior[key].to('cpu')
        queues.controller_model_bc.put(state_dict_behavior)

    def load_outside_behavior(self, queues: InterProcessQueues) -> bool:
        if not (queues.controller_model_bc.empty() or queues.controller_memory.empty()):
            self.behavior.load_state_dict(queues.controller_model_bc.get())
            self.memory.load_state_dict(queues.controller_memory.get())
            MP_LOGGER.info("New state dict for Behavior and Memory models are loaded")
            return True
        else:
            return False

    def add_expert_episode(self,
                           state_mus: torch.Tensor,
                           state_logsigmas: torch.Tensor,
                           actions: torch.Tensor,
                           hyposcene_id: int | None = None,
                           shared_dicts: InterProcessSharedDicts | None = None) -> None:
        """
        states_encoded
            Expected dimensions: (batch, autoencoder_latent_features)
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

        mus_seq = self.make_history_tensor(state_mus)
        logsigmas_seq = self.make_history_tensor(state_logsigmas)
        actions_onehot_seq = self.make_history_tensor(actions).to(torch.float32)
        action_indices_seq = self.make_history_tensor(actions_indices.unsqueeze(-1)).squeeze(-1)

        assert (self._train_processor.processing_settings.true_history_frame_len //
                self._train_processor.processing_settings.frame_seq_len_screening_rate) >= 2

        action_residuals_last = actions_onehot_seq[:, -1, ...] - actions_onehot_seq[:, -2, ...]
        action_indices_last = action_indices_seq[..., -1]

        new_episode = ExpertEpisode(state_mus=mus_seq,
                                    state_logsigmas=logsigmas_seq,
                                    action_residuals=action_residuals_last,
                                    actions_indices=action_indices_last
                                    )
        if self.expert_episode is None:
            self.expert_episode = new_episode
        else:
            self.expert_episode = self.expert_episode.concat(new_episode=new_episode)

        if hyposcene_id is not None and shared_dicts is not None:
            shared_dicts.training_data_sizes_from_training_process[hyposcene_id] = len(self.expert_episode)
        return

    def make_history_tensor(self,
                            tensor: torch.Tensor) -> torch.Tensor:
        size = self._train_processor.processing_settings.true_history_frame_len

        tensor_seq = tensor.clone().unfold(dimension=0, size=size, step=1)
        tensor_seq = tensor_seq.permute(0, 2, 1)  # batch, sequence, channels

        if self._train_processor.processing_settings.frame_seq_len_screening_rate != 1:
            tensor_seq = tensor_seq[:, ::self._train_processor.processing_settings.frame_seq_len_screening_rate, :]

        return tensor_seq

    def get_state_dicts(self) -> dict[str, dict]:
        return {'memory': self.memory.state_dict(),
                'behavior': self.behavior.state_dict()}

    def set_state_dicts(self, state_dicts: dict[str, dict]) -> None:
        self.memory.load_state_dict(state_dicts['memory'])
        self.behavior.load_state_dict(state_dicts['behavior'])

    def train_epoch(self) -> tuple[float, dict]:
        ep = self.expert_episode

        sample_count = ep.state_mus.shape[0]
        permutation = torch.randperm(sample_count)

        epoch_losses: list[float] = []
        epoch_losses_memory: list[float] = []
        epoch_losses_behavior: list[float] = []
        for i in range(0, sample_count, self.settings.batch_size):
            batch_indices = permutation[i:i + self.settings.batch_size]

            total_input = self.autoencoder.reparametrize(mu=ep.state_mus[batch_indices],
                                                         logsigma=ep.state_logsigmas[batch_indices])
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

    def process_and_train_full(self,
                               state_mus: torch.Tensor,
                               state_logsigmas: torch.Tensor,
                               actions_train: torch.Tensor,
                               hyposcene_id: int | None = None,
                               interprocess_shared_dicts: InterProcessSharedDicts | None = None,
                               queues: InterProcessQueues | None = None):
        self.add_expert_episode(state_mus=state_mus,
                                state_logsigmas=state_logsigmas,
                                hyposcene_id=hyposcene_id,
                                shared_dicts=interprocess_shared_dicts,
                                actions=actions_train)
        self.train_full(
            running_loss_window_size=self.settings.running_loss_window_size,
            log_frequency=self.settings.train_log_frequency,
            target_loss=self.settings.target_loss,
            force_stop_at_plateau_epochs=self.settings.force_stop_at_plateau_epochs,
            min_epochs=self.settings.min_epochs,
            grace_epochs_after_min_epochs=self.settings.grace_epochs_after_min_epochs,
            max_epochs=self.settings.max_epochs,
            queues=queues
        )