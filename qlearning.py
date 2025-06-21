import axelrod as axl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class QLearningPlayer(axl.Player):
    """Q-learning agent for Iterated Prisoner's Dilemma."""

    name = "Q-Learning"

    classifier = {
        "memory_depth": float("inf"),
        "stochastic": True,
        "long_run_time": False,
        "inspects_source": False,
        "manipulates_source": False,
        "manipulates_state": False,
    }

    def __init__(self, alpha=0.5, gamma=0.6):
        """Initialize Q-learning parameters and Q-table."""
        super().__init__()
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.q_table = {}  # Q-values for state-action pairs
        self.last_state = None
        self.last_action = None

    def get_state(self, opponent):
        """Return current state: (my last move, opponent's last move)."""
        if len(self.history) == 0:
            return (None, None)
        return (self.history[-1], opponent.history[-1])

    def choose_action(self, state, n_rounds):
        """Choose action using stable Boltzmann exploration."""
        if state not in self.q_table:
            self.q_table[state] = {axl.Action.C: np.random.uniform(-0.1, 0.1),
                                axl.Action.D: np.random.uniform(-0.1, 0.1)}

        q_vals = self.q_table[state]
        actions = list(q_vals.keys())

        # Keep temperature from becoming too small
        t = max(5 * (0.999 ** n_rounds), 0.01)
        
        if t < 0.01:
            return max(self.q_table[state], key=self.q_table[state].get)

        # Use NumPy for numerical stability
        logits = np.array([q_vals[a] / t for a in actions])
        max_logit = np.max(logits)
        exp_logits = np.exp(logits - max_logit)  # subtract max for stability
        probs = exp_logits / np.sum(exp_logits)

        return np.random.choice(actions, p=probs)


    def update_q_table(self, reward, next_state):
        """Update Q-values using the Q-learning formula."""
        if self.last_state is not None and self.last_action is not None:
            if next_state not in self.q_table:
                self.q_table[next_state] = {axl.Action.C: 0, axl.Action.D: 0}

            best_next_action = max(self.q_table[next_state], key=self.q_table[next_state].get)
            self.q_table[self.last_state][self.last_action] += self.alpha * (
                reward + self.gamma * self.q_table[next_state][best_next_action] - 
                self.q_table[self.last_state][self.last_action]
            )

    def strategy(self, opponent):
        """Select an action based on Q-learning strategy."""
        if len(self.history) < 3:
            return np.random.choice([axl.Action.C, axl.Action.D])  # Ensure early exploration

        state = self.get_state(opponent)
        action = self.choose_action(state, len(self.history))

        # Get game rewards
        game = self.match_attributes["game"]
        if len(self.history) > 0:
            last_round = (self.history[-1], opponent.history[-1])
            scores = game.score(last_round)
            reward = scores[0]
        else:
            reward = 0

        # Update Q-table
        next_state = self.get_state(opponent)
        self.update_q_table(reward, next_state)

        # Store state-action pair
        self.last_state = state
        self.last_action = action

        # Print Q-values after each round
        # print(f"\nAfter Round {len(self.history)}:")
        # self.print_q_values()

        # Visualize Q-table
        # self.visualize_q_table()

        return action

    def print_q_values(self):
        """Prints the Q-values in a readable format."""
        print("Q-values:")
        for state, actions in self.q_table.items():
            print(f"  State {state}: C={actions[axl.Action.C]:.2f}, D={actions[axl.Action.D]:.2f}")

    def visualize_q_table(self):
        """Plot Q-table heatmap."""
        states = list(self.q_table.keys())
        actions = [axl.Action.C, axl.Action.D]

        q_values = np.zeros((len(states), len(actions)))

        for i, state in enumerate(states):
            for j, action in enumerate(actions):
                q_values[i, j] = self.q_table[state][action]

        plt.figure(figsize=(8, 6))
        sns.heatmap(q_values, annot=True, cmap="coolwarm", xticklabels=["C", "D"], yticklabels=states)
        plt.xlabel("Actions")
        plt.ylabel("States")
        plt.title(f"Q-Table After Round {len(self.history)}")
        plt.show()

def check(history, targets=((axl.Action.C, axl.Action.D), (axl.Action.D, axl.Action.C))):
    max_len = 0
    max_start = None

    for start in range(len(history) - 1):  # need at least one pair to check alternation
        tolerance = 0
        streak_len = 1
        prev = history[start]

        if prev not in targets:
            continue  # skip if starting point isn't (C, D) or (D, C)

        for i in range(start + 1, len(history)):
            current = history[i]
            expected_next = targets[1] if prev == targets[0] else targets[0]

            if current == expected_next:
                streak_len += 1
                prev = current
            else:
                tolerance += 1
                allowed_tolerance = streak_len // 10
                if tolerance > allowed_tolerance:
                    break
                else:
                    streak_len += 1
                    prev = current  # update even if noisy

        if streak_len > max_len:
            max_len = streak_len
            max_start = start

    return (max_start, max_len) if max_start is not None else (None, 0)


# Run simulation
q_player = QLearningPlayer()
tft_player = axl.TitForTat()

match=axl.Match([q_player,tft_player],turns=10000)
for i in range(30):
    match_results=match.play()
    print(check(match_results))

# # Play a match of 50 rounds
# match = axl.Match([q_player, tft_player], turns=30000)
# match_results = match.play()

# # Print results
# # print("Match History:")
# # print(match_results)

# print(check(match_results)) 

# # Get final scores
# final_scores = match.final_score()

# # Convert history to a list before counting
# q_player_history = list(q_player.history)
# tft_player_history = list(tft_player.history)

# # Count cooperations and defections for each player
# cooperated_count_q_player = q_player_history.count(axl.Action.C)
# defected_count_q_player = q_player_history.count(axl.Action.D)

# cooperated_count_tft_player = tft_player_history.count(axl.Action.C)
# defected_count_tft_player = tft_player_history.count(axl.Action.D)

# # Print final results
# print("Final Results:")

# print(f"q Player Final Score: {final_scores[0]}")
# print(f"TitForTat Player Final Score: {final_scores[1]}")

# print(f"q Player Cooperated: {cooperated_count_q_player} times")
# print(f"q Player Defected: {defected_count_q_player} times")

# print(f"TitForTat Player Cooperated: {cooperated_count_tft_player} times")
# print(f"TitForTat Player Defected: {defected_count_tft_player} times")