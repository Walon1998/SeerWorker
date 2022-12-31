import random

from rlgym.utils import TerminalCondition
from rlgym.utils.gamestates import GameState


class SeerGameCondition(TerminalCondition):  # Mimics a Rocket League game
    def __init__(self, tick_skip=8, overtime_prob=0.1):
        super(SeerGameCondition).__init__()
        self.tick_skip = tick_skip
        self.timer = 0
        self.overtime = False
        self.done = True
        self.initial_state = None
        self.score = 0
        self.overtime_prob = overtime_prob

    def reset(self, initial_state: GameState):
        self.initial_state = initial_state
        self.done = False

        if random.random() < self.overtime_prob:
            self.overtime = True
            self.timer = 0
            self.score = 0
        else:
            self.overtime = False
            self.timer = random.randrange(10, 300)
            self.score = random.choice([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])

    def is_terminal(self, current_state: GameState) -> bool:
        reset = False

        scored = ((current_state.blue_score - self.initial_state.blue_score)
                  - (current_state.orange_score - self.initial_state.orange_score))

        self.score += scored

        if scored != 0:  # Goal scored
            reset = True

        if self.overtime and scored != 0:  # Overtime
            self.done = True

        elif self.timer <= 0 and current_state.ball.position[2] <= 110:

            if self.score != 0:
                self.done = True
            else:
                self.done = False

        else:
            self.timer -= self.tick_skip / 120

        self.timer = max(0, self.timer)

        return reset or self.done
