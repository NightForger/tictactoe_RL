# main.py

import matplotlib.pyplot as plt
from tictactoe import TicTacToe
from agent import TwoStepQLearningShapingAgent
from train import train_fixed_episodes, demo_game

def main():
    # Example usage
    N = 4
    K = 4
    env = TicTacToe(N=N, K=K)

    agent = TwoStepQLearningShapingAgent(
        alpha=0.5,
        gamma=0.9,
        epsilon_start=0.3,
        epsilon_end=0.0,
        N=N,
        K=K,
        winning_lines=env.winning_lines,
        total_episodes=100000
    )

    print("Training for 100,000 episodes (10% of them with 'aggressive O')...")
    train_fixed_episodes(env, agent, num_episodes=100000)

    print("\nDemo game (self-play) after training:")
    demo_game(env, agent)
    ani = env.render()
    plt.show()  # or in Jupyter you can just display(ani)

if __name__ == "__main__":
    main()
