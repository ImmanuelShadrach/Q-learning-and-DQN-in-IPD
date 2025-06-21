import axelrod as axl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

class DQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.model(x)

class DQNPlayer(axl.Player):
    name = "DQN"

    classifier = {
        "memory_depth": float("inf"),
        "stochastic": True,
        "long_run_time": True,
        "inspects_source": False,
        "manipulates_source": False,
        "manipulates_state": False,
    }

    def __init__(self, alpha=0.5, gamma=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

        self.memory = []  # (state, action, reward, next_state) tuples
        self.batch_size = 32
        self.state_size = 4  # (my_last_C, my_last_D, opp_last_C, opp_last_D)
        self.action_size = 2  # C or D

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQNetwork(self.state_size, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        self.loss_fn = nn.MSELoss()

        self.last_state = None
        self.last_action = None

    def encode_state(self, opponent):
        """Convert game state to a fixed-length vector."""
        if len(self.history) == 0:
            return np.array([0, 0, 0, 0], dtype=np.float32)
        my_last = self.history[-1]
        opp_last = opponent.history[-1]
        return np.array([
            1.0 if my_last == axl.Action.C else 0.0,
            1.0 if my_last == axl.Action.D else 0.0,
            1.0 if opp_last == axl.Action.C else 0.0,
            1.0 if opp_last == axl.Action.D else 0.0
        ], dtype=np.float32)

    def choose_action(self, state):
        """Choose action using stable Boltzmann exploration based on number of rounds played."""
        n_rounds = len(self.history)

        # Encode state for network input
        state_tensor = torch.tensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.model(state_tensor).cpu().numpy().flatten()

        # Temperature decay
        t = 5 * (0.999 ** n_rounds)

        # If temperature is too low, go greedy
        if t < 0.01:
            return axl.Action.C if np.argmax(q_values) == 0 else axl.Action.D

        # Compute numerically stable softmax probabilities
        logits = q_values / t
        max_logit = np.max(logits)
        exp_logits = np.exp(logits - max_logit)  # stability trick
        probs = exp_logits / np.sum(exp_logits)

        # Choose action based on softmax probability
        action_idx = np.random.choice([0, 1], p=probs)
        return axl.Action.C if action_idx == 0 else axl.Action.D

    def store_experience(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
        if len(self.memory) > 10000:
            self.memory.pop(0)

    def train_model(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.tensor(np.array(states)).to(self.device)
        next_states = torch.tensor(np.array(next_states)).to(self.device)
        actions = torch.tensor([0 if a == axl.Action.C else 1 for a in actions]).to(self.device)
        rewards = torch.tensor(rewards).float().to(self.device)


        q_values = self.model(states)
        next_q_values = self.model(next_states)

        target_q_values = q_values.clone()
        for i in range(self.batch_size):
            best_next_q = torch.max(next_q_values[i])
            target_q_values[i, actions[i]] = rewards[i] + self.gamma * best_next_q

        loss = self.loss_fn(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def strategy(self, opponent):
        state = self.encode_state(opponent)
        action = self.choose_action(state)

        # Calculate reward from last round
        if len(self.history) > 0:
            last_round = (self.history[-1], opponent.history[-1])
            reward = self.match_attributes["game"].score(last_round)[0]
        else:
            reward = 0

        if self.last_state is not None and self.last_action is not None:
            next_state = state
            self.store_experience(self.last_state, self.last_action, reward, next_state)
            self.train_model()

        self.last_state = state
        self.last_action = action

        # Print Q-table for all 4 possible states
        # self.print_q_table()

        return action

    def print_q_table(self):
        """Print the Q-values for all 4 possible states."""
        state_names = ['CC', 'CD', 'DC', 'DD']
        state_encodings = [
            np.array([1, 0, 1, 0], dtype=np.float32),  # CC
            np.array([1, 0, 0, 1], dtype=np.float32),  # CD
            np.array([0, 1, 1, 0], dtype=np.float32),  # DC
            np.array([0, 1, 0, 1], dtype=np.float32),  # DD
        ]

        print(f"\nQ-values after Round {len(self.history)}:")
        for name, state in zip(state_names, state_encodings):
            state_tensor = torch.tensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.model(state_tensor).cpu().numpy().flatten()
            print(f"  State {name}: C={q_values[0]:.4f}, D={q_values[1]:.4f}")

def check(history, target=(axl.Action.D, axl.Action.D)):
    max_len = 0
    max_start = None

    for start in range(len(history)):
        count = 0
        tolerance = 0

        for i in range(start, len(history)):
            total_so_far = i - start + 1
            current = history[i]

            # Adaptive allowed tolerance = 10% of current streak length (rounded down)
            allowed_tolerance = total_so_far // 10

            if current == target:
                count += 1
            else:
                tolerance += 1
                if tolerance > allowed_tolerance:
                    break  # Too noisy, break this streak

            # Update if this streak beats the max
            if total_so_far > max_len:
                max_len = total_so_far
                max_start = start

    return (max_start, max_len) if max_start is not None else (None, 0)


# Run simulation
dqn_player = DQNPlayer()
tft_player = axl.TitForTat()

match=axl.Match([dqn_player,tft_player],turns=10000)
for i in range(10):
    match_results=match.play()
    print(check(match_results))

# # Play a match of 50 rounds
# match = axl.Match([dqn_player, tft_player], turns=30000)
# match_results = match.play()

# # Print results
# # print("Match History:")
# # print(match_results)

# print(check(match_results)) 

# # Get final scores
# final_scores = match.final_score()

# # Convert history to a list before counting
# dqn_player_history = list(dqn_player.history)
# tft_player_history = list(tft_player.history)

# # Count cooperations and defections for each player
# cooperated_count_dqn_player = dqn_player_history.count(axl.Action.C)
# defected_count_dqn_player = dqn_player_history.count(axl.Action.D)

# cooperated_count_tft_player = tft_player_history.count(axl.Action.C)
# defected_count_tft_player = tft_player_history.count(axl.Action.D)

# # Print final results
# print("Final Results:")

# print(f"DQN Player Final Score: {final_scores[0]}")
# print(f"TitForTat Player Final Score: {final_scores[1]}")

# print(f"DQN Player Cooperated: {cooperated_count_dqn_player} times")
# print(f"DQN Player Defected: {defected_count_dqn_player} times")

# print(f"TitForTat Player Cooperated: {cooperated_count_tft_player} times")
# print(f"TitForTat Player Defected: {defected_count_tft_player} times")