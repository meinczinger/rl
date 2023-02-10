class Tempreture:
    def __init__(
        self,
        start=1.0,
        end=0.15,
        last_episode=100,
    ) -> None:
        self.start = start
        self.end = end
        self.last_episode = last_episode

    def calculate_tempreture(self, current_epoch: int, decrease:bool = True):
        if decrease:
            return max(
                self.end,
                self.start - current_epoch / self.last_episode,
            )
        else:
            return max(
                self.end,
                self.start + current_epoch / self.last_episode,
            )
