class GridWorld:
    def __init__(self,step_reward=-1):
        self.width = 5
        self.height = 3

        self.start = (0, 2)
        self.goal = (4, 0)

        self.obstacles = {
            (1, 1),
            (2, 1)
        }

    
        self.step_reward = step_reward

        self.agent_position = self.start

    def reset(self):
        self.agent_position = self.start
        return self.agent_position

    def step(self, action):
        x, y = self.agent_position

        if action == "UP":
            new_position = (x, y - 1)

        elif action == "DOWN":
            new_position = (x, y + 1)

        elif action == "LEFT":
            new_position = (x - 1, y)

        elif action == "RIGHT":
            new_position = (x + 1, y)

        else:
            raise ValueError("Invalid action")

        if (
            new_position[0] < 0
            or new_position[0] >= self.width
            or new_position[1] < 0
            or new_position[1] >= self.height
        ):
            return self.agent_position, -5, False

        if new_position in self.obstacles:
            return self.agent_position, -5, False

        self.agent_position = new_position

        if self.agent_position == self.goal:
            return self.agent_position, 10, True

        return self.agent_position,self.step_reward, False