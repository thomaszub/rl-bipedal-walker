import gym
import hydra
from omegaconf import DictConfig
from tqdm import trange

from agent import DDPGAgent, DDPGAgentConfig
from replay_buffer import ReplayBuffer, ReplayBufferConfig


def run(env: gym.Env, agent: DDPGAgent, render: bool) -> float:
    state = env.reset()
    done = False
    sum_reward = 0
    while not done:
        if render:
            env.render()
        new_state, reward, done, _ = env.step(agent.action(state))
        agent.update(new_state, reward, done)
        sum_reward += reward
        state = new_state

    return sum_reward


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    env = gym.make("BipedalWalker-v3")
    print(f"State shape: {env.observation_space.shape}")
    print(f"Action shape: {env.action_space.shape}")
    buffer_config = ReplayBufferConfig.fromDictConfig(cfg.replay_buffer)
    buffer_state_action = ReplayBuffer(
        buffer_config, env.observation_space.shape, env.action_space.shape
    )
    agent_config = DDPGAgentConfig.fromDictConfig(cfg.agent)
    agent = DDPGAgent(agent_config, env.action_space, buffer_state_action)

    with trange(0, 1000) as tr:
        for _ in tr:
            reward = run(env, agent, False)
            tr.set_postfix(reward=reward)


if __name__ == "__main__":
    main()
