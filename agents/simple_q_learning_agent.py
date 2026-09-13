import random


class SimpleQLearningAgent:

    def __init__(
        self,
        actions,
        learning_rate=0.1,
        discount_factor=0.9,
        exploration_rate=1.0,
        exploration_decay=0.995
    ):
        self.actions = actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay

        self.q_table = {}

    def get_q_values(self, state):

        if state not in self.q_table:
            self.q_table[state] = {
                action: 0.0
                for action in self.actions
            }

        return self.q_table[state]

    def choose_action(self, state):

        q_values = self.get_q_values(state)

        if random.random() < self.exploration_rate:
            return random.choice(self.actions)

        return max(q_values, key=q_values.get)

    def update(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        q_values = self.get_q_values(state)

        current_q = q_values[action]

        if done:
            target = reward

        else:
            next_q_values = self.get_q_values(next_state)

            best_next_q = max(next_q_values.values())

            target = (
                reward
                + self.discount_factor * best_next_q
            )

        q_values[action] = (
            current_q
            + self.learning_rate
            * (target - current_q)
        )

    def decay_exploration(self):

        self.exploration_rate *= self.exploration_decay