from environment.simple_reward_env import SimpleRewardEnv
from agents.simple_q_learning_agent import SimpleQLearningAgent


env = SimpleRewardEnv(
    loop_reward=2,
    goal_reward=10,
    max_steps=10
)

actions = ["GOAL", "LOOP"]

agent = SimpleQLearningAgent(actions)

episodes = 1000


for episode in range(episodes):

    state = env.reset()

    done = False

    while not done:

        action = agent.choose_action(state)

        next_state, reward, done = env.step(action)

        agent.update(
            state,
            action,
            reward,
            next_state,
            done
        )

        state = next_state

    agent.decay_exploration()


print("Training complete.")

print("\nQ-table:")

for state, values in agent.q_table.items():

    print(state, values)


print("\nExploration rate:", agent.exploration_rate)