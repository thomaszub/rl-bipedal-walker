import gym
import hydra
from omegaconf import DictConfig

from agent import DDPGAgent
from replay_buffer import ReplayBuffer, ReplayBufferConfig


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    _ = ReplayBuffer(ReplayBufferConfig.fromDictConfig(cfg.replay_buffer))
    env = gym.make("BipedalWalker-v3")
    agent = DDPGAgent(env.action_space)
    _ = env.reset()
    done = False
    sum_reward = 0
    while not done:
        env.render()
        new_state, reward, done, _ = env.step(agent.action())
        sum_reward += reward

    print(sum_reward)


if __name__ == "__main__":
    main()
