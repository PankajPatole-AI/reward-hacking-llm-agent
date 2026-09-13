import matplotlib.pyplot as plt


agents = ["Q-learning", "LLM"]

success_rates = [0.0, 1.0]

plt.figure(figsize=(7, 5))

plt.bar(agents, success_rates)

plt.ylabel("Success Rate")
plt.title("Goal Success Under Misspecified Reward")

plt.ylim(0, 1.1)

plt.tight_layout()

plt.savefig(
    "results/success_rate_comparison.png",
    dpi=300
)

plt.close()


average_rewards = [20.0, 10.0]

plt.figure(figsize=(7, 5))

plt.bar(agents, average_rewards)

plt.ylabel("Average Reward")
plt.title("Average Reward Under Misspecified Reward")

plt.tight_layout()

plt.savefig(
    "results/average_reward_comparison.png",
    dpi=300
)

plt.close()


print("Plots created successfully.")