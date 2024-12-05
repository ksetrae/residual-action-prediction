import torch

from set_transformer.modules import SAB, PMA


class MemoryConditionedBehaviorCloning(torch.nn.Module):
    def __init__(self,
                 state_space_size: int,
                 mem_hidden_size: int,
                 curr_hidden_size: int,
                 action_space_size: int
                 ):
        super(MemoryConditionedBehaviorCloning, self).__init__()
        # self.current_observations_processor = torch.nn.Sequential(
        #     torch.nn.Linear(state_space_size, curr_hidden_size),
        #     torch.nn.ReLU(),
        #     torch.nn.LayerNorm(curr_hidden_size),
        #     torch.nn.Linear(curr_hidden_size, curr_hidden_size),
        #     torch.nn.ReLU(),
        #     torch.nn.LayerNorm(curr_hidden_size),
        # )
        # self.current_observations_processor_instances = torch.nn.Sequential(
        #     SAB(state_space_size, curr_hidden_size, num_heads=1, ln=True),
        #     SAB(curr_hidden_size, curr_hidden_size, num_heads=1, ln=True),
        #     PMA(curr_hidden_size, num_heads=1, num_seeds=1, ln=True),
        # )

        self.current_observations_processor_flat = torch.nn.Sequential(
            torch.nn.Linear(state_space_size, curr_hidden_size),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(curr_hidden_size),
            torch.nn.Linear(curr_hidden_size, curr_hidden_size),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(curr_hidden_size),
        )

        self.history_processor = torch.nn.Sequential(
            torch.nn.Linear(mem_hidden_size, curr_hidden_size),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(curr_hidden_size),
        )

        self.joined_net = torch.nn.Linear(curr_hidden_size*2, action_space_size)

    def forward(self, observations_current: torch.Tensor, history_encoded: torch.Tensor) -> torch.Tensor:
        """

        Parameters
        ----------
        observations_current
            dimensions: (batch, instance, features)
        history_encoded
            dimensions: (batch, features)
        Returns
        -------

        """

        history_encoded = self.history_processor(history_encoded)
        history_encoded = history_encoded.detach().clone()

        # observations_current = self.current_observations_processor_instances(observations_current)
        # observations_current = observations_current.squeeze(1)
        observations_current = self.current_observations_processor_flat(observations_current)

        joined = torch.cat((observations_current, history_encoded), dim=-1)

        res = self.joined_net(joined)
        return res

    def act(self, observations: torch.Tensor, history: torch.Tensor) -> int:
        with torch.no_grad():
            action_probability = self.forward(observations_current=observations, history_encoded=history)
        probs = torch.softmax(action_probability, dim=-1)
        action_distribution = torch.distributions.Categorical(probs)
        action_index = action_distribution.sample().item()
        return action_index


class MemoryEncoder(torch.nn.Module):
    def __init__(self,
                 state_space_size: int,
                 hidden_size: int,
                 ):
        super(MemoryEncoder, self).__init__()

        # self.instance_encoder = torch.nn.Sequential(
        #     SAB(state_space_size, hidden_size, num_heads=1, ln=True),
        #     SAB(hidden_size, hidden_size, num_heads=1, ln=True),
        #     PMA(hidden_size, num_heads=1, num_seeds=1, ln=True)
        # )

        self.encoder_top = torch.nn.Sequential(
            torch.nn.Linear(state_space_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(hidden_size),
        )

        self.encoder_final = torch.nn.Linear(hidden_size * 1, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Parameters
        ----------
        x
            dimensions: (batch, sequence, instance, features)

        Returns
        -------
        x
            dimensions: (batch, features)
        """
        # assert len(x.shape) == 4
        # # Treat sequence dimension as batch for the purpose of encoding instances to a flat vector
        # # orig_shape = x.shape
        # x = x.flatten(start_dim=1, end_dim=2)
        # x = self.instance_encoder(x)
        # x = x.squeeze(-1)  # batch, sequence, features
        assert len(x.shape) == 3

        res = self.encoder_top(x)

        # For now simply take the mean over sequence. Could use RNN or attention again.
        res = res.mean(1)

        mu = self.encoder_final(res)
        return mu


class MemoryModel(torch.nn.Module):
    def __init__(self,
                 state_space_size: int,
                 hidden_size: int,
                 action_space_size: int,
                 ):
        super(MemoryModel, self).__init__()

        self.encoder = MemoryEncoder(state_space_size=state_space_size, hidden_size=hidden_size)
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(hidden_size),
            torch.nn.Linear(hidden_size, action_space_size),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        res = self.decoder(latent)
        return res, latent
