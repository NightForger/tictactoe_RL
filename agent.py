# agent.py
import random

def count_max_consecutive(board, player, winning_lines):
    """
    For each line in winning_lines, count how many 'player' cells are in that line
    and return the maximum count across them.
    """
    mx = 0
    for line in winning_lines:
        c = sum(board[idx] == player for idx in line)
        if c > mx:
            mx = c
    return mx

def choose_aggressive_action(env, player):
    """
    A simple approach: pick the move that results in the largest
    'max consecutive' line for 'player' after that move.
    """
    valid_actions = env.get_available_actions()
    best_score = -999
    best_act = None
    for act in valid_actions:
        temp_board = env.board[:]
        temp_board[act] = player
        sc = count_max_consecutive(temp_board, player, env.winning_lines)
        if sc > best_score:
            best_score = sc
            best_act = act
    return best_act


class TwoStepQLearningShapingAgent:
    """
    Two-step Q-learning agent with:
      - storing transitions in a buffer
      - once we have 2 consecutive transitions for the same player => 2-step update
      - if an episode ends => flush with 1-step
      - lose penalty (if the current move gets +1 => the previous move of the opponent = -1)
      - shaping for blocking opponent's threats
      - shaping for building own lines
      - decaying epsilon
    """
    def __init__(self, alpha=0.5, gamma=0.9,
                 epsilon_start=0.3, epsilon_end=0.0,
                 N=5, K=4, winning_lines=None,
                 total_episodes=300000):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end   = epsilon_end
        self.total_episodes = total_episodes

        self.Q = {}
        self.buffer = []

        self.N = N
        self.K = K
        self.winning_lines = winning_lines
        self.current_episode = 1  # updated from outside

    def get_epsilon(self):
        frac = min(1.0, self.current_episode / self.total_episodes)
        return self.epsilon_start + frac*(self.epsilon_end - self.epsilon_start)

    def get_Q(self, state):
        if state not in self.Q:
            self.Q[state] = [0.0]*len(state)
        return self.Q[state]

    def choose_action(self, state, valid_actions):
        eps = self.get_epsilon()
        if random.random() < eps:
            return random.choice(valid_actions)
        qvals = self.get_Q(state)
        max_q = max(qvals[a] for a in valid_actions)
        best_acts = [a for a in valid_actions if qvals[a] == max_q]
        return random.choice(best_acts)

    def update(
        self, 
        state, action, env_reward, 
        next_state, next_actions,
        done, current_player_id,
        board_before, board_after
    ):
        # (A) Lose penalty: if current move has env_reward=+1 => the previous move of the opponent = -1
        if env_reward == 1.0:
            for i in reversed(range(len(self.buffer))):
                (p, s, a, r, s_next, va, dd) = self.buffer[i]
                if (not dd) and (p != current_player_id):
                    self.buffer[i] = (p, s, a, -1.0, s_next, va, True)
                    break

        # (B) Shaping
        shaping_block = self.shaping_block(board_before, board_after, current_player_id)
        shaping_build = self.shaping_build(board_before, board_after, current_player_id)
        total_reward = env_reward + shaping_block + shaping_build

        # (C) store transition
        self.buffer.append(
            (current_player_id, state, action, total_reward, next_state, next_actions, done)
        )

        # (D) 2-step update if ready
        self._two_step_update_if_ready()

        # (E) if done => flush
        if done:
            while len(self.buffer) > 0:
                (p0, s0, a0, r0, s1, acts1, d0) = self.buffer.pop(0)
                old_q = self.get_Q(s0)[a0]
                target = r0
                self.Q[s0][a0] = old_q + self.alpha*(target - old_q)

    def _two_step_update_if_ready(self):
        if len(self.buffer) < 2:
            return
        (p0, s0, a0, r0, s1, acts1, done0) = self.buffer[0]
        (p1, s1b, a1b, r1b, s2, acts2, done1) = self.buffer[1]
        if (p0 == p1) and (not done0):
            # 2-step
            if done1:
                target = r0 + self.gamma*r1b
            else:
                nxt_q = self.get_Q(s2)
                max_next = max(nxt_q[a] for a in acts2) if acts2 else 0.0
                target = r0 + self.gamma*r1b + (self.gamma**2)*max_next
        else:
            # 1-step
            target = r0

        old_q = self.get_Q(s0)[a0]
        self.Q[s0][a0] = old_q + self.alpha*(target - old_q)

        self.buffer.pop(0)

    # ------------------------------
    # SHAPING PARTS
    # ------------------------------
    def shaping_block(self, board_before, board_after, current_player):
        """
        Larger shaping for blocking or ignoring threats:
         - if opp has K-1 => ignoring => -2.0, blocking => +1.0
         - if opp has K-2 => ignoring => -1.0, blocking => +0.3
         priority: K-1 lines > K-2 lines
        """
        opp = -current_player
        penalty_kminus1 = -2.0
        reward_block_kminus1 = 1.0
        penalty_kminus2 = -1.0
        reward_block_kminus2 = 0.3

        shaping = 0.0
        lines_kminus1 = []
        lines_kminus2 = []

        # collect lines
        for line in self.winning_lines:
            cnt_opp = sum(board_before[idx] == opp for idx in line)
            cnt_me  = sum(board_before[idx] == current_player for idx in line)
            cnt_emp = sum(board_before[idx] == 0 for idx in line)

            if cnt_me == 0:
                if cnt_opp == self.K - 1 and cnt_emp == 1:
                    lines_kminus1.append(line)
                elif cnt_opp == self.K - 2 and cnt_emp == 2:
                    lines_kminus2.append(line)

        # K-1
        if len(lines_kminus1) > 0:
            blocked_any = False
            failed_any = False
            for line in lines_kminus1:
                cnt_opp_after = sum(board_after[idx] == opp for idx in line)
                cnt_emp_after = sum(board_after[idx] == 0 for idx in line)
                if cnt_opp_after == self.K - 1 and cnt_emp_after == 1:
                    failed_any = True
                else:
                    blocked_any = True
            if failed_any:
                shaping += penalty_kminus1
            elif blocked_any:
                shaping += reward_block_kminus1
        else:
            # K-2
            if len(lines_kminus2) > 0:
                blocked_any = False
                failed_any = False
                for line in lines_kminus2:
                    cnt_opp_after = sum(board_after[idx] == opp for idx in line)
                    cnt_emp_after = sum(board_after[idx] == 0 for idx in line)
                    if cnt_opp_after == self.K - 2 and cnt_emp_after == 2:
                        failed_any = True
                    else:
                        blocked_any = True
                if failed_any:
                    shaping += penalty_kminus2
                elif blocked_any:
                    shaping += reward_block_kminus2

        return shaping

    def shaping_build(self, board_before, board_after, current_player):
        """
        Reward for extending own lines, e.g. +0.1 for each line that we extended by 1 cell.
        """
        reward_for_extension = 0.1
        shaping = 0.0
        for line in self.winning_lines:
            cnt_me_before = sum(board_before[idx] == current_player for idx in line)
            cnt_me_after  = sum(board_after[idx] == current_player for idx in line)
            if cnt_me_after == cnt_me_before + 1:
                shaping += reward_for_extension
        return shaping
