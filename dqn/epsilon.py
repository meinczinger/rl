class Epsilon:
    def __init__(
        self,
        eps_start=1.0,
        eps_end=0.15,
        eps_last_episode=100,
    ) -> None:
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_last_episode = eps_last_episode

    def calculate_epsilon(self, current_epoch: int):
        return max(
            self.eps_end,
            self.eps_start - current_epoch / self.eps_last_episode,
        )
