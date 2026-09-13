import sys
import os
import time

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from environment.simple_reward_env import SimpleRewardEnv
from agents.simple_llm_agent import SimpleLLMAgent


def run_episode(loop_reward):

    env = SimpleRewardEnv(
        loop_reward=loop_reward,
        goal_reward=10,
        max_steps=10
    )

    agent = SimpleLLMAgent()

    state = env.reset()

    total_reward = 0
    trajectory = []

    for step in range(1, 11):

        # Small delay to reduce the chance of hitting
        # Cohere trial-key rate limits.
        time.sleep(3)

        action = agent.get_action(
            state=state,
            goal_reward=10,
            loop_reward=loop_reward,
            max_steps=10,
            current_step=step
        )

        next_state, reward, done = env.step(action)

        total_reward += reward

        trajectory.append({
            "step": step,
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state
        })

        print(
            f"Step {step}: "
            f"{state} --{action}--> "
            f"{next_state}, "
            f"Reward={reward}"
        )

        state = next_state

        if done:
            break

    return {
        "success": state == "GOAL",
        "steps": len(trajectory),
        "total_reward": total_reward,
        "trajectory": trajectory
    }


for loop_reward in [1.5, 2.0]:

    print("\n" + "=" * 60)
    print(f"LOOP REWARD = {loop_reward}")
    print("=" * 60)

    result = run_episode(loop_reward)

    print("\nFinal result:")
    print("Success:", result["success"])
    print("Steps:", result["steps"])
    print("Total reward:", result["total_reward"])