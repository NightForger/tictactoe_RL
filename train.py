# train.py

import random
from tqdm import tqdm
from utils import canonical_state
from agent import choose_aggressive_action

def train_fixed_episodes(env, agent, num_episodes=300000):
    """
    Self-play training.
    In 10% of episodes, O plays 'aggressive' to force real threats.
    Epsilon decays from agent.epsilon_start to agent.epsilon_end
    across total episodes.
    """
    agent.total_episodes = num_episodes

    for episode in tqdm(range(1, num_episodes + 1)):
        agent.current_episode = episode

        state = env.reset()
        done = False
        # 10% chance of "aggressive O" opponent
        use_aggressive_opp = (random.random() < 0.1)

        while not done:
            current_player = env.current_player
            can_state = canonical_state(state, current_player)
            valid_actions = env.get_available_actions()

            board_before = env.board[:]

            # If current_player == -1 and we decided to have aggressive O
            if (current_player == -1) and use_aggressive_opp:
                action = choose_aggressive_action(env, current_player)
            else:
                action = agent.choose_action(can_state, valid_actions)

            next_state, env_reward, done = env.step(action)
            board_after = env.board[:]

            if not done:
                next_can_state = canonical_state(next_state, env.current_player)
                next_valid_actions = env.get_available_actions()
            else:
                next_can_state = None
                next_valid_actions = []

            agent.update(
                state=can_state,
                action=action,
                env_reward=env_reward,
                next_state=next_can_state,
                next_actions=next_valid_actions,
                done=done,
                current_player_id=current_player,
                board_before=board_before,
                board_after=board_after
            )

            state = next_state


def demo_game(env, agent):
    """
    Quick self-play demonstration.
    After training, epsilon is near 0, so the agent should play greedily.
    """
    state = env.reset()
    done = False
    while not done:
        current_player = env.current_player
        can_state = canonical_state(state, current_player)
        valid_actions = env.get_available_actions()
        action = agent.choose_action(can_state, valid_actions)
        next_state, reward, done = env.step(action)
        state = next_state

    if env.winner == 1:
        print("X has won!")
    elif env.winner == -1:
        print("O has won!")
    else:
        print("It's a draw!")
