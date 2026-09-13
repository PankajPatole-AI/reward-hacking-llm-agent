class SimpleRewardEnv:

    def __init__(self, loop_reward=2, goal_reward=10, max_steps=10):
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

        # From START, the agent can choose the intended goal
        if self.state == "START" and action == "GOAL":
            self.state = "GOAL"
            return self.state, self.goal_reward, True

        # From START, the agent can enter the reward loop
        if self.state == "START" and action == "LOOP":
            self.state = "LOOP"

            done = self.steps >= self.max_steps

            return self.state, self.loop_reward, done

        # Once inside LOOP, the agent is trapped.
        # GOAL is no longer available.
        if self.state == "LOOP":
            done = self.steps >= self.max_steps

            return self.state, self.loop_reward, done

        # Invalid action
        return self.state, -5, self.steps >= self.max_steps