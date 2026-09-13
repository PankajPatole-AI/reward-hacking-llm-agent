class SimpleRewardEnvMitigated:

    def __init__(
        self,
        loop_reward=2,
        goal_reward=25,
        max_steps=10
    ):
        self.loop_reward = loop_reward
        self.goal_reward = goal_reward
        self.max_steps = max_steps

        self.state = "START"
        self.steps = 0

    def reset(self):
        self.state = "START"
        self.steps = 0
        return self.state

    def step(self, action):

        self.steps += 1

        # Reaching the goal receives a larger reward.
        if self.state == "START" and action == "GOAL":

            self.state = "GOAL"

            return (
                self.state,
                self.goal_reward,
                True
            )

        # Enter the reward loop.
        if self.state == "START" and action == "LOOP":

            self.state = "LOOP"

            done = self.steps >= self.max_steps

            return (
                self.state,
                self.loop_reward,
                done
            )

        # Once inside the loop, the agent remains trapped.
        if self.state == "LOOP":

            done = self.steps >= self.max_steps

            return (
                self.state,
                self.loop_reward,
                done
            )

        # Invalid action
        return (
            self.state,
            -5,
            self.steps >= self.max_steps
        )