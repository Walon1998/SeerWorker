import random

from rlgym.utils import TerminalCondition
from rlgym.utils.gamestates import GameState


class SeerGameCondition(TerminalCondition):  # Mimics a Rocket League game
    def __init__(self, tick_skip=8, overtime_prob=0.9):
        super(SeerGameCondition).__init__()
        self.tick_skip = tick_skip
        self.timer = 0
        self.overtime = False
        self.done = True
        self.initial_state = None
        self.differential = 0
        self.score = 0
        self.overtime_prob = overtime_prob

    def reset(self, initial_state: GameState):
        self.initial_state = initial_state
        self.done = False
        self.differential = 0

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

        differential = ((current_state.blue_score - self.initial_state.blue_score)
                        - (current_state.orange_score - self.initial_state.orange_score))

        if differential != self.differential:  # Goal scored
            reset = True

        if self.overtime:
            self.timer = 0

            if differential != 0:
                self.done = True
            else:
                self.done = False
        else:
            if self.timer <= 0 and current_state.ball.position[2] <= 110:
                # Can't detect ball on ground directly, should be an alright approximation.
                # Anything below z vel of ~690uu/s should be detected. 50% for 1380 etc.
                if differential != 0:
                    self.done = True
                else:
                    self.overtime = True
                    self.done = False
                    reset = True

                self.timer = 0
            else:
                self.timer -= self.tick_skip / 120

        return reset or self.done
