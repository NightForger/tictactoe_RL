# TicTacToe \(NxN\) with K-in-a-row using Two-Step Q-Learning

This repository implements a generalized Tic-Tac-Toe environment for an \(NxN\) board, where a player must align \(K\) marks in a row (horizontally, vertically, or diagonally) to win. We focus on **reinforcement learning** approaches, specifically **two-step Q-Learning** with **reward shaping** and an **aggressive opponent** for self-play in 10% of episodes.

## Why this is Hard

In the classic \(3x3\) Tic-Tac-Toe with \(K=3\), simple one-step Q-Learning can eventually learn to block the opponent's immediate threat and/or create a winning alignment. However, when \(N > K\) (for example, a \(5x5\) board with \(K=4\)), **one-step** Q-Learning fails to see multi-step threats. Often, you need to **block** your opponent **two or more moves** in advance. A purely 1-step method:

- Only sees \(r_{t+1}\) or the immediate next state's Q-values.
- Fails to anticipate that ignoring a partially formed diagonal can lead to a forced loss in 2-3 turns.

## Our Solution: Two-Step Q-Learning + Reward Shaping

To address this, we employ a **two-step Q-Learning** approach, which incorporates:

1. **Two-step TD targets**:  
   \[
   Q(s_t, a_t) \leftarrow Q(s_t, a_t) \;+\; \alpha \Bigl[r_{t+1} + \gamma \, r_{t+2} + \gamma^2 \max_a Q(s_{t+2}, a) - Q(s_t, a_t)\Bigr].
   \]
   This allows the agent to learn from outcomes **two** moves away.

2. **Reward shaping** to encourage:
   - **Blocking** an opponent's \(K-1\) or \(K-2\) threat (or penalize ignoring it).
   - **Building**/extending your own line by +1 step.

3. **Aggressive opponent** in 10% of episodes, which forces the agent to handle real threats more often during self-play.

These changes significantly improve the agent's ability to see and defend multi-step threats, especially in boards where \(N>K\).

## Repository Structure

```
.
├── main.py               # Example 'main' script: creates env, agent, runs training & demo
├── tictactoe.py          # The environment class (TicTacToe) for NxN, K-in-a-row
├── agent.py              # The two-step Q-Learning shaping agent; plus 'choose_aggressive_action'
├── utils.py              # Helper function canonical_state(...)
├── train.py              # Training loop (self-play) and a demo_game(...) function
├── README.md             # This file
└── requirements.txt      # (Optional) dependencies list
```

### Key Files

- **`tictactoe.py`**: Implements the environment.  
  - Features *early termination* if neither player can still form \(K\).  
  - `render()` method to animate the game.  
- **`agent.py`**: Contains `TwoStepQLearningShapingAgent` class with two-step Q-Learning logic, shaping rewards, and an \(\epsilon\)-decay schedule. Also includes `choose_aggressive_action(...)`.  
- **`train.py`**: The self-play training function `train_fixed_episodes(env, agent, ...)` that runs a certain number of episodes. Also has a `demo_game(...)` for a quick self-play demonstration.  
- **`main.py`**: Shows how to instantiate environment + agent, run training, and do a final demonstration. This is a simple entry point for local usage.

## Example Usage

1. **Install requirements** (e.g. `numpy`, `matplotlib`, `tqdm`, etc.).  
   ```bash
   pip install -r requirements.txt
   ```
2. **Run `main.py`**  
   ```bash
   python main.py
   ```
   This will:
   - Create a 4x4 environment with `K=4`.
   - Initialize a two-step Q-Learning agent.
   - Train for 100k episodes (with 10% episodes featuring an aggressive O).
   - Show a final demo game result in text and pop up a matplotlib animation if you have a GUI.

## Colab Notebook

You can also check out our **Colab Notebook** version ([**Open in Colab**](https://colab.research.google.com/your-cute-link)) where we use **ipywidgets** to create an interactive form. In the Colab form, you can:

- Choose board size `N`, winning condition `K`.
- Set training hyperparameters (`alpha`, `gamma`, \(\epsilon\)-decay, number of episodes).
- Optionally load a **pretrained agent** from GitHub (for certain `(N,K)` combos).
- Run a self-play demo or even **play manually** against the trained agent.

### Example ipywidgets Screenshot

*(Placeholder for your screenshot!)*

```
![Screenshot of the ipywidgets UI](docs/images/tictactoe_widgets.png)
```

## Pretrained Models

We provide pretrained models in `agents/` subfolder for some typical board sizes:

- **`agent_3_3.pkl`** – \(3\times3\), \(K=3\)
- **`agent_4_4.pkl`** – \(4\times4\), \(K=4\)
- **`agent_5_4.pkl`** – \(5\times5\), \(K=4\)
- **`agent_4_3.pkl`** – \(4\times4\), \(K=3\)

They can be automatically downloaded by the Colab if you enable “Load pretrained from GitHub?” for those exact `(N, K)` combos.

## Why Two-Step is Essential

- **One-step** Q-Learning only looks at immediate reward and \(\max Q(s_{t+1}, a)\).  
- If \(N > K\), you may need to block an opponent's threat *two* or more turns in advance. Otherwise, the agent doesn’t see the negative reward \(-1\) from losing until it’s too late.  
- By using **two-step returns** or **reward shaping**, we propagate the “danger” signals earlier, so the agent learns to block more effectively.

## License and Contributing

You can use or modify this code according to the License in this repository (e.g., MIT). Feel free to open a Pull Request if you have improvements or bug fixes.

Enjoy experimenting with larger boards, or new reward-shaping heuristics for more advanced tactics!
