from agents.llm_agent import LLMAgent


agent = LLMAgent()


action = agent.get_action(
    state=(0, 2),
    goal=(4, 0),
    reward=0,
    step_reward=-1
)


print("Selected action:", action)