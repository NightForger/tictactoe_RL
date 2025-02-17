# TicTacToe RL: Two-Step Q-Learning with Reward Shaping

[**Open the Colab Notebook here!**](https://colab.research.google.com/drive/1UCl0LNdFOJXsnsSumb6awv7aHiSWzAC6?usp=sharing)

Welcome to our TicTacToe Reinforcement Learning project! Here we tackle the problem of training an RL agent to play an **\(N \times N\)** TicTacToe variant, where a player needs **K** in a row to win. This is significantly more challenging than the classic 3x3 TicTacToe for several reasons, especially when \(N > K\), because a simple 1-step Q-Learning agent often cannot "see" deep enough to block future threats in time.

Below is a brief overview of **why** this problem is difficult and **how** we solve it via **two-step Q-Learning** with various reward-shaping techniques.

---

## Why 1-step Q-Learning Fails for \(N > K\)

- In standard 1-step Q-Learning, the update only looks at the immediate reward and then the best Q-value for the next state. 
- This works **okay** for small boards (like 3x3 with K=3) because most threats can be identified and blocked in a single move. 
- However, for boards where **N > K**, threats may develop over multiple moves. The agent needs to block or build sequences two or three moves in advance. Pure 1-step Q-Learning usually doesn't credit the correct actions with the correct outcomes if those outcomes occur multiple moves later.

Hence, an agent trained only on immediate 1-step updates might consistently fail to block unstoppable threats unless it accidentally stumbles upon the right move. 

---

## Our Two-Step Q-Learning Approach

To address the above, we implemented:
- **Two-step Q-Learning**: We store transitions in a short buffer, and once we have two consecutive transitions for the same player, we perform an update with a 2-step return. This allows the agent to incorporate the reward from two steps ahead, making it more likely to learn critical "blocking" or "building" moves that pay off two turns later.
- **Reward Shaping**: We add intermediate rewards and penalties for:
  - Failing to block the opponent's nearly completed line (K-1, K-2, etc.).
  - Extending our own line (gaining partial progress to K in a row).
  - Penalizing the last move of the loser when the other side wins.
- **Decaying Epsilon**: We begin training with some exploration (`epsilon_start`) and gradually reduce it to zero so that eventually, the agent exploits what it has learned.

As a result, the agent:
1. Learns to prioritize blocking the opponent's potential lines.
2. Learns to build its own lines effectively.
3. Avoids purely local (1-step) traps by looking at consequences a couple of moves ahead.

---

## Repository Contents

Below is an outline of the files you will find in this repository:

- **`tictactoe.py`**  
  Contains the `TicTacToe` environment class, which implements an \(N \times N\) board with the logic for `step()`, checking winners, possible lines to form K in a row, and early detection of unwinnable states.

- **`utils.py`**  
  Contains helper functions such as `canonical_state(...)`.

- **`agent.py`**  
  Includes:
  - `TwoStepQLearningShapingAgent`: The main Q-Learning agent with 2-step updates and shaping.
  - `choose_aggressive_action(...)`: An optional function that simulates an "aggressive opponent" for O.

- **`train.py`**  
  - `train_fixed_episodes(...)`: A self-play training loop that also integrates an aggressive opponent in 10% of the episodes.
  - `demo_game(...)`: Runs a quick self-play demonstration with the trained agent.

- **`main.py`**  
  - An example main script that demonstrates how to create the environment, instantiate the agent, train it, and optionally run a demo game.  

We also have placeholders for **pretrained agents**:

- `agent_3_3.pkl` (for 3x3 with K=3)
- `agent_4_4.pkl` (for 4x4 with K=4)
- `agent_5_4.pkl` (for 5x5 with K=4)
- `agent_4_3.pkl` (for 4x4 with K=3)

---

## Usage

To run this code locally:

1. **Clone** or **download** this repository.

2. **Install dependencies** (for example, using pip):
   ```bash
   pip install -r requirements.txt
   ```
   or at least install `numpy`, `matplotlib`, and `tqdm`.

3. **Run** `main.py`:
   ```bash
   python main.py
   ```
   This will:
   - Create an environment (default N=4, K=4).
   - Instantiate the TwoStepQLearningShapingAgent.
   - Train for a specified number of episodes (e.g. 100,000).
   - Print the result of a demo self-play game.
   - Display a matplotlib animation if you have a GUI environment. 

To use a **pretrained agent**:

- Place the `.pkl` file (e.g. `agent_4_4.pkl`) in an accessible folder.
- Modify the code to load that pickle into the agent's `Q` dictionary before running the demo game. 
- That way, you can skip training time.

---

## Colab Notebook

If you'd like to run or demo the project in **Google Colab**, we have an **interactive notebook** with `ipywidgets` forms for choosing parameters (board size, number in a row, number of episodes, etc.), plus the ability to load pretrained agents from GitHub. 

[**Open the Colab Notebook here!**](https://colab.research.google.com/drive/1UCl0LNdFOJXsnsSumb6awv7aHiSWzAC6?usp=sharing)

---

## Why This Matters

- **Classic** TicTacToe (3x3) is trivial for any search-based or RL agent. But **generalizing** to NxN with K in a row reveals the importance of multi-step lookahead or advanced reward shaping.
- Pure 1-step Q-Learning fails to anticipate the 2+ moves needed to block or create lines. 
- Our approach with **2-step Q-Learning** (and **aggressive shaping**) addresses these deeper tactical patterns, making the agent significantly better at large or near-large boards.

---

## Contributing

Feel free to open issues or submit pull requests if you find bugs or want to improve the agent’s heuristics and shaping logic. We welcome enhancements like:
- Multi-step expansions (3-step, n-step).
- Further reward shaping improvements.
- More thorough scheduling for exploration.
