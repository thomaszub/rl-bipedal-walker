import gym
import hydra
from omegaconf import DictConfig

from agent import DDPGAgent, DDPGAgentConfig


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    env = gym.make("BipedalWalker-v3")
    print(f"State shape: {env.observation_space.shape}")
    print(f"Action shape: {env.action_space.shape}")
    agent_config = DDPGAgentConfig.fromDictConfig(cfg.agent)
    agent = DDPGAgent(agent_config)

    rewards = agent.train(env)

    with open("rewards.txt", "w") as f:
        f.write("\n".join(map(lambda r: str(r), rewards)))

    agent.save()


if __name__ == "__main__":
    main()
