import torch


class MemoryConditionedBehaviorCloning(torch.nn.Module):
    def __init__(self,
                 state_space_size: int,
                 mem_hidden_size: int,
                 curr_hidden_size: int,
                 action_space_size: int
                 ):
        super(MemoryConditionedBehaviorCloning, self).__init__()
        self.obs_net = torch.nn.Sequential(
            torch.nn.Linear(state_space_size, curr_hidden_size),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(curr_hidden_size),
            torch.nn.Linear(curr_hidden_size, curr_hidden_size),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(curr_hidden_size),
        )

        self.hist_proc = torch.nn.Sequential(
            torch.nn.Linear(mem_hidden_size, curr_hidden_size),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(curr_hidden_size),
        )

        self.joined_net = torch.nn.Linear(curr_hidden_size*2, action_space_size)

    def forward(self, observations_current: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        history = self.hist_proc(history)
        history = history.detach().clone()

        observations_current = self.obs_net(observations_current)

        joined = torch.cat((observations_current, history), dim=-1)

        res = self.joined_net(joined)
        return res

    def act(self, observations: torch.Tensor, history: torch.Tensor) -> int:
        with torch.no_grad():
            action_probability = self.forward(observations_current=observations, history=history)
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
            dimensions: (batch, sequence, features)

        Returns
        -------

        """
        assert len(x.shape) == 3

        # res, _ = self.rnn(x)
        # res = res[:, -1, :]
        res = self.encoder_top(x)
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
