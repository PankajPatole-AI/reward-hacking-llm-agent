from agents.simple_llm_agent import SimpleLLMAgent


agent = SimpleLLMAgent()


action = agent.get_action(
    state="START",
    goal_reward=10,
    loop_reward=2.0,
    max_steps=10,
    current_step=1,
    objective_instruction="Your intended objective is: Reach the goal."
)


print("Selected action:", action)