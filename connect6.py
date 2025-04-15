# -*- coding: utf-8 -*-
import sys
import numpy as np
import random
from numba import njit
import copy
from operator import itemgetter
import math

# --- Constants ---
EMPTY = 0
BLACK = 1
WHITE = 2
DIRECTIONS = np.array([[0, 1], [1, 0], [1, 1], [1, -1]], dtype=np.int32)
BOARD_EVAL_SCORES = np.array([0, 1, 50, 500, 5000, 10000, 1000000], dtype=np.int64)
WINDOW_SIZE = 6
WIN_CONDITION = 6

DEBUG_OUTPUT = True # Control debug prints

# --- Numba Accelerated Board Utilities ---
# (Numba functions remain the same as the previous refactored version)
@njit
def _can_place_window(r: int, c: int, dr: int, dc: int, grid_dim: int, length: int = WINDOW_SIZE) -> bool:
    """Checks if a window of `length` fits starting at (r, c) in direction (dr, dc)."""
    end_r, end_c = r + dr * (length - 1), c + dc * (length - 1)
    return 0 <= end_r < grid_dim and 0 <= end_c < grid_dim

@njit
def _extract_window(grid: np.ndarray, r: int, c: int, dr: int, dc: int, length: int = WINDOW_SIZE) -> np.ndarray:
    """Extracts a window of values from the grid."""
    window = np.zeros(length, dtype=np.int8)
    for k in range(length):
        window[k] = grid[r + k * dr, c + k * dc]
    return window

@njit
def evaluate_line_pattern(window: np.ndarray, player_id: int, score_table: np.ndarray, moves_left_in_turn: int) -> int:
    """
    Analyzes a 6-cell window pattern for scoring.
    Considers connected stones, open ends, and potential based on remaining moves.
    (Keeping the refactored logic here, review if issues persist)
    """
    score = 0
    opponent_id = 3 - player_id

    max_connected = 0
    current_connected = 0
    open_ends = 0
    player_stones = 0
    empty_cells = 0
    opponent_stones = 0

    is_open_start = (window[0] == EMPTY)

    for i in range(WINDOW_SIZE):
        cell = window[i]
        if cell == player_id:
            current_connected += 1
            player_stones += 1
        elif cell == EMPTY:
            if current_connected > 0 and i + 1 < WINDOW_SIZE and window[i + 1] == player_id:
                 open_ends += 1
            max_connected = max(max_connected, current_connected)
            current_connected = 0
            empty_cells += 1
        else:
            max_connected = max(max_connected, current_connected)
            current_connected = 0
            opponent_stones += 1

    max_connected = max(max_connected, current_connected)

    is_open_end = (window[WINDOW_SIZE - 1] == EMPTY)

    if opponent_stones == 0:
        effective_stones = player_stones
        if moves_left_in_turn == 2 and empty_cells > 0 :
             effective_stones = min(player_stones + 1, WINDOW_SIZE)

        score += score_table[effective_stones]

        actual_open_ends = 0
        if is_open_start and player_stones > 0 and window[0] == EMPTY:
             adjacent_player = False
             for k in range(1, WINDOW_SIZE):
                 if window[k] == player_id: adjacent_player = True; break
                 if window[k] == opponent_id: break
             if adjacent_player: actual_open_ends +=1

        if is_open_end and player_stones > 0 and window[WINDOW_SIZE-1] == EMPTY:
             adjacent_player = False
             for k in range(WINDOW_SIZE-2, -1, -1):
                 if window[k] == player_id: adjacent_player = True; break
                 if window[k] == opponent_id: break
             if adjacent_player: actual_open_ends +=1

        actual_open_ends += open_ends

        if max_connected >= 2:
             score += actual_open_ends * (max_connected**2)

    return score

@njit
def score_grid_state(grid: np.ndarray, grid_dim: int, player_id: int, moves_left_in_turn: int) -> tuple:
    """Calculates the overall heuristic score for the grid for both players."""
    my_total_score = 0
    opponent_total_score = 0
    opponent_id = 3 - player_id
    opponent_moves_left = 2 # Assume opponent always has 2 moves for their evaluation

    for r in range(grid_dim):
        for c in range(grid_dim):
            for dr, dc in DIRECTIONS:
                if _can_place_window(r, c, dr, dc, grid_dim):
                    window = _extract_window(grid, r, c, dr, dc)
                    # Evaluate for the player whose perspective we need
                    current_player_score = evaluate_line_pattern(window, player_id, BOARD_EVAL_SCORES, moves_left_in_turn)
                    my_total_score += current_player_score
                    # Evaluate for the opponent
                    current_opp_score = evaluate_line_pattern(window, opponent_id, BOARD_EVAL_SCORES, opponent_moves_left)
                    opponent_total_score += current_opp_score

    return my_total_score, opponent_total_score


@njit
def assess_position_strength(grid: np.ndarray, grid_dim: int, r: int, c: int, player_id: int, moves_left_in_turn: int):
    """
    Evaluates the strategic value of placing a stone at (r, c) for player_id.
    (Keeping the refactored logic here, review if issues persist)
    """
    center_bonus = 5 - (abs(r - grid_dim // 2) + abs(c - grid_dim // 2)) // (grid_dim // 4 + 1)
    score = max(0, center_bonus)
    opponent_id = 3 - player_id

    for dr, dc in DIRECTIONS:
        current_line_score = 0
        consecutive_count = 1
        empty_forward = 0
        empty_backward = 0
        potential_forward = 0
        potential_backward = 0

        # Forward
        rr, cc = r + dr, c + dc
        forward_blocked = False
        spaces_encountered = 0
        while 0 <= rr < grid_dim and 0 <= cc < grid_dim and spaces_encountered < (WINDOW_SIZE - 1):
            if grid[rr, cc] == player_id:
                if spaces_encountered == 0: consecutive_count += 1
                else: potential_forward += 1
            elif grid[rr, cc] == EMPTY:
                spaces_encountered += 1
                empty_forward += 1
            else: forward_blocked = True; break
            if consecutive_count + potential_forward + empty_forward >= WIN_CONDITION: break
            rr += dr; cc += dc

        # Backward
        rr, cc = r - dr, c - dc
        backward_blocked = False
        spaces_encountered = 0
        while 0 <= rr < grid_dim and 0 <= cc < grid_dim and spaces_encountered < (WINDOW_SIZE - 1):
            if grid[rr, cc] == player_id:
                if spaces_encountered == 0: consecutive_count += 1
                else: potential_backward += 1
            elif grid[rr, cc] == EMPTY:
                spaces_encountered += 1
                empty_backward += 1
            else: backward_blocked = True; break
            if consecutive_count + potential_backward + empty_backward >= WIN_CONDITION: break
            rr -= dr; cc -= dc

        total_potential_len = consecutive_count + potential_forward + potential_backward
        total_empty_in_line = empty_forward + empty_backward
        open_ends = (0 if forward_blocked else 1) + (0 if backward_blocked else 1)

        if total_potential_len + total_empty_in_line >= WIN_CONDITION:
             effective_len = total_potential_len
             # *** Original code used remain_turns == 2 check here ***
             # *** Replicating that specific check ***
             if moves_left_in_turn == 2 and total_empty_in_line > 0 :
                  effective_len += 1 # Assume placing another stone if 2 moves available

             # *** Scoring logic similar to original _evaluate_position ***
             s = 0
             if effective_len >= WIN_CONDITION: s += 100000 # Win
             elif effective_len == WIN_CONDITION - 1: s += 1000 * open_ends # Threaten win (original used 1000)
             elif effective_len == WIN_CONDITION - 2: s += 500 * open_ends # Strong setup (original used 500)
             elif effective_len == WIN_CONDITION - 3: s += 200 * open_ends # Decent setup (original used 200)
             elif effective_len == WIN_CONDITION - 4: s += 50 * open_ends  # Basic setup (original used 50)
             # Add raw connection bonus? Original added s, let's add based on length.
             current_line_score += s + consecutive_count * 5

        score += current_line_score
    return score

@njit
def find_winning_moves(grid: np.ndarray, grid_dim: int, player_id: int, moves_needed: int):
    """
    Checks if placing `moves_needed` stones can lead to a win for `player_id`.
    Returns the coordinates of the winning moves if found, otherwise (-1,-1).
    (Logic unchanged)
    """
    winning_sequence = np.full((moves_needed, 2), -1, dtype=np.int32)
    temp_empty_coords = np.zeros((moves_needed, 2), dtype=np.int32)

    for r in range(grid_dim):
        for c in range(grid_dim):
            for dr, dc in DIRECTIONS:
                if _can_place_window(r, c, dr, dc, grid_dim):
                    player_stones_count = 0
                    empty_cells_count = 0
                    empty_idx = 0
                    possible_win = True
                    for k in range(WINDOW_SIZE):
                        cr, cc = r + k * dr, c + k * dc
                        cell_state = grid[cr, cc]
                        if cell_state == player_id:
                            player_stones_count += 1
                        elif cell_state == EMPTY:
                            if empty_cells_count < moves_needed:
                                temp_empty_coords[empty_idx, 0] = cr
                                temp_empty_coords[empty_idx, 1] = cc
                                empty_idx += 1
                            empty_cells_count += 1
                        else:
                            possible_win = False
                            break

                    if possible_win and player_stones_count == WIN_CONDITION - moves_needed and empty_cells_count == moves_needed:
                        for m in range(moves_needed):
                            winning_sequence[m, 0] = temp_empty_coords[m, 0]
                            winning_sequence[m, 1] = temp_empty_coords[m, 1]
                        return winning_sequence
    return winning_sequence

@njit
def get_valid_moves(grid_dim: int, grid: np.ndarray):
    """Returns a list of all empty coordinates (r, c). (Logic unchanged)"""
    count = 0
    for r in range(grid_dim):
        for c in range(grid_dim):
            if grid[r, c] == EMPTY:
                count += 1
    if count == 0:
        return np.empty((0, 2), dtype=np.int32)
    valid_moves = np.empty((count, 2), dtype=np.int32)
    idx = 0
    for r in range(grid_dim):
        for c in range(grid_dim):
            if grid[r, c] == EMPTY:
                valid_moves[idx, 0] = r
                valid_moves[idx, 1] = c
                idx += 1
    return valid_moves

@njit
def get_nearby_valid_moves(grid_dim: int, grid: np.ndarray, search_radius: int = 2):
    """Returns empty cells within `search_radius` of any existing stone. (Logic unchanged)"""
    marked_empty = np.zeros((grid_dim, grid_dim), dtype=np.bool_)
    has_stones = False
    for r in range(grid_dim):
        for c in range(grid_dim):
            if grid[r, c] != EMPTY:
                has_stones = True
                for dr in range(-search_radius, search_radius + 1):
                    for dc in range(-search_radius, search_radius + 1):
                        if dr == 0 and dc == 0: continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < grid_dim and 0 <= nc < grid_dim and grid[nr, nc] == EMPTY:
                            marked_empty[nr, nc] = True
    if not has_stones:
         return get_valid_moves(grid_dim, grid)
    count = 0
    for r in range(grid_dim):
        for c in range(grid_dim):
            if marked_empty[r,c]: count += 1
    if count == 0:
         return get_valid_moves(grid_dim, grid)
    nearby_moves = np.empty((count, 2), dtype=np.int32)
    idx = 0
    for r in range(grid_dim):
        for c in range(grid_dim):
             if marked_empty[r,c]:
                nearby_moves[idx, 0] = r
                nearby_moves[idx, 1] = c
                idx += 1
    return nearby_moves

@njit
def check_victory(grid: np.ndarray, grid_dim: int) -> int:
    """Checks if Black (1) or White (2) has won. Returns winning player ID or 0. (Logic unchanged)"""
    for r in range(grid_dim):
        for c in range(grid_dim):
            player_id = grid[r, c]
            if player_id != EMPTY:
                for dr, dc in DIRECTIONS:
                    prev_r, prev_c = r - dr, c - dc
                    if 0 <= prev_r < grid_dim and 0 <= prev_c < grid_dim and grid[prev_r, prev_c] == player_id:
                         continue
                    count = 0
                    for k in range(WIN_CONDITION + 1):
                        nr, nc = r + k * dr, c + k * dc
                        if 0 <= nr < grid_dim and 0 <= nc < grid_dim and grid[nr, nc] == player_id:
                            count += 1
                        else:
                            break
                    if count >= WIN_CONDITION:
                        return player_id
    return EMPTY

# --- MCTS Components ---
class SearchNode:
    """Represents a node in the Monte Carlo Search Tree. (Unchanged)"""
    def __init__(self, parent_node=None, probability=1.0):
        self.parent = parent_node
        self.children = {}
        self.visit_count = 0
        self.value_score = 0.0
        self.prior_probability = probability

    def expand_children(self, possible_actions, action_probabilities):
        for action, probability in zip(possible_actions, action_probabilities):
            action_tuple = tuple(action)
            if action_tuple not in self.children:
                self.children[action_tuple] = SearchNode(parent_node=self, probability=probability)

    def select_best_child(self, exploration_constant):
        best_score = -np.inf
        best_action = None
        best_child = None
        for action, child in self.children.items():
            score = child.calculate_ucb(exploration_constant)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        return best_action, best_child

    def backpropagate(self, leaf_node_value):
        self.visit_count += 1
        self.value_score += (leaf_node_value - self.value_score) / self.visit_count

    def calculate_ucb(self, exploration_constant):
        if self.visit_count == 0:
            return np.inf
        q_value = self.value_score
        parent_visits = self.parent.visit_count if self.parent else 1
        exploration_bonus = (exploration_constant * self.prior_probability *
                             math.sqrt(parent_visits) / (1 + self.visit_count))
        return q_value + exploration_bonus

    def is_terminal_node(self):
        return len(self.children) == 0

    def is_root_node(self):
        return self.parent is None

# --- Heuristic Policy for MCTS Rollouts ---
# *** MODIFIED to align with original rollout_policy_func ***
def heuristic_policy_selector(game_state: 'GameState', probability_threshold=0.05, search_dist=2):
    """
    Selects promising moves based on heuristic evaluation for rollouts.
    Returns a list of moves and their probabilities.
    *** This version attempts to replicate the original code's policy function more closely. ***
    """
    candidate_moves_np = get_nearby_valid_moves(game_state.dimension, game_state.grid, search_dist)
    candidate_moves_tuples = [tuple(move) for move in candidate_moves_np] # Use tuples internally

    if not candidate_moves_tuples:
        all_moves_np = get_valid_moves(game_state.dimension, game_state.grid)
        candidate_moves_tuples = [tuple(move) for move in all_moves_np]
        if not candidate_moves_tuples:
            return [], np.array([]) # No possible moves left

    moves_left = game_state.get_moves_left_in_turn()
    current_player = game_state.active_player
    opponent_player = 3 - current_player

    scores = []
    # *** Replicate original scoring: sum of player's potential and opponent's potential at that spot ***
    for r, c in candidate_moves_tuples:
        my_eval = assess_position_strength(game_state.grid, game_state.dimension, r, c, current_player, moves_left)
        opp_eval = assess_position_strength(game_state.grid, game_state.dimension, r, c, opponent_player, 2) # Opponent always assumed 2 moves
        combined_score = my_eval + opp_eval # Simple sum as in original
        scores.append(combined_score)

    scores = np.array(scores, dtype=np.float64)

    # *** Replicate original probability generation and filtering ***
    if scores.max() <= 0: # If all scores are zero or negative
        if len(candidate_moves_tuples) > 0:
            # Uniform probability (original code did this too if max was 0)
            probs = np.ones(len(candidate_moves_tuples)) / len(candidate_moves_tuples)
        else:
            return [], np.array([]) # Should not happen if moves exist, but safety check
    else:
        # Softmax-like scaling as in original (exp / 100)
        scaled_scores = np.exp(scores / 100.0) # Temperature scaling

        # Threshold filtering based on max *scaled* score (original used max score before exp)
        # Let's adapt: filter based on max *raw* score * threshold
        score_threshold = scores.max() * probability_threshold

        filtered_moves_indices = [i for i, score in enumerate(scores) if score >= score_threshold]

        if not filtered_moves_indices: # If filtering removes all, take the single best raw score move
             best_raw_idx = np.argmax(scores)
             filtered_moves_indices = [best_raw_idx]

        # Apply scaling and normalization only to filtered moves
        filtered_scaled_scores = scaled_scores[filtered_moves_indices]
        
        # Normalize the scaled scores to get probabilities
        probs_sum = filtered_scaled_scores.sum()
        if probs_sum > 0:
             probs = filtered_scaled_scores / probs_sum
        else: # Handle case where sum is zero (e.g., all scores were negative and exp became tiny)
             probs = np.ones(len(filtered_moves_indices)) / len(filtered_moves_indices)


        # Get the actual move tuples corresponding to the filtered indices
        final_candidate_moves = [candidate_moves_tuples[i] for i in filtered_moves_indices]
        
        # Convert final moves list back to numpy array for consistency if needed by caller
        # (The MCTS expand_children takes an iterable, list of tuples is fine)
        candidate_moves_np = np.array(final_candidate_moves, dtype=np.int32)


    # Ensure probabilities sum to 1
    if not np.isclose(probs.sum(), 1.0):
       probs = probs / probs.sum()

    # Return the filtered moves (as numpy array) and their probabilities
    return candidate_moves_np, probs


class MonteCarloTreeSearch:
    """Implements the MCTS algorithm for move selection. (Unchanged structure)"""
    def __init__(self, exploration_factor=1.41, simulation_limit=1000):
        self.root_node = SearchNode(parent_node=None, probability=1.0)
        self.exploration_factor = exploration_factor
        self.simulation_count = simulation_limit

    def run_simulation(self, current_game_state: 'GameState'):
        sim_game_state = copy.deepcopy(current_game_state)
        node = self.root_node
        # 1. Selection
        while not node.is_terminal_node():
            action, node = node.select_best_child(self.exploration_factor)
            if action is None: break
            sim_game_state.apply_move(action)

        # 2. Expansion
        is_over, _ = sim_game_state.is_finished()
        leaf_value = 0.0
        if not is_over:
            possible_moves, move_probs = heuristic_policy_selector(sim_game_state, probability_threshold=0.05) # Use modified selector
            if possible_moves.shape[0] > 0: # Check if moves exist
                node.expand_children(possible_moves, move_probs)
                # Select child to start simulation (e.g., using UCB or randomly)
                action, node = node.select_best_child(self.exploration_factor)
                if action:
                    sim_game_state.apply_move(action)
                    leaf_value = self.simulate_playout(sim_game_state) # 3. Simulation
                else:
                    leaf_value = self._evaluate_terminal_state(sim_game_state, current_game_state.active_player)
            else: # No moves from this state
                 leaf_value = self._evaluate_terminal_state(sim_game_state, current_game_state.active_player)
        else: # Game ended during selection
            leaf_value = self._evaluate_terminal_state(sim_game_state, current_game_state.active_player)

        # 4. Backpropagation
        current_node = node
        while current_node is not None:
            current_node.backpropagate(leaf_value)
            leaf_value *= -1
            current_node = current_node.parent

    def _evaluate_terminal_state(self, game_state: 'GameState', original_player: int) -> float:
         is_over, winner = game_state.is_finished(check_now=True)
         if not is_over: return 0.0
         if winner == original_player: return 1.0
         elif winner == (3 - original_player): return -1.0
         else: return 0.0 # Draw

    # *** MODIFIED simulate_playout end condition ***
    def simulate_playout(self, game_state: 'GameState', max_depth=10) -> float:
        """
        Simulates a game using heuristic policy.
        *** Returns evaluation based on score comparison at depth limit, closer to original. ***
        """
        playout_start_player = game_state.active_player
        temp_state = game_state # Modify in place for speed during rollout

        for _ in range(max_depth):
            is_over, winner = temp_state.is_finished(check_now=True)
            if is_over:
                return self._evaluate_terminal_state(temp_state, playout_start_player)

            moves_needed = temp_state.get_moves_left_in_turn()
            action_to_play = None

            # Prioritize immediate winning moves
            winning_moves = find_winning_moves(temp_state.grid, temp_state.dimension, temp_state.active_player, moves_needed)
            if not np.all(winning_moves == -1):
                 for i in range(moves_needed):
                     if winning_moves[i, 0] != -1:
                          action_to_play = tuple(winning_moves[i])
                          temp_state.apply_move(action_to_play)
                          if moves_needed == 1: break
                     else: action_to_play = None; break
                 if action_to_play: continue

            # If no immediate win, use heuristic policy (the modified one)
            if action_to_play is None:
                possible_moves, move_probs = heuristic_policy_selector(temp_state, probability_threshold=0.01, search_dist=2)
                if possible_moves.shape[0] == 0: # Use .shape[0] for numpy array
                    break # No moves possible

                if possible_moves.shape[0] == 1:
                     action_to_play = tuple(possible_moves[0]) # Convert np array row to tuple
                else:
                    # np.random.choice works on indices
                    chosen_index = np.random.choice(possible_moves.shape[0], p=move_probs)
                    action_to_play = tuple(possible_moves[chosen_index])

                temp_state.apply_move(action_to_play)

        # *** Revert End-of-Rollout Evaluation to Score Comparison ***
        # If loop finishes due to depth limit, evaluate based on board score difference.
        is_over, winner = temp_state.is_finished(check_now=True) # Final check
        if is_over:
             return self._evaluate_terminal_state(temp_state, playout_start_player)
        else:
            # Use score_grid_state (similar to original evaluate_board)
            # Evaluate from the perspective of the player who started the rollout
            moves_left_at_end = temp_state.get_moves_left_in_turn()
            my_score, opp_score = score_grid_state(temp_state.grid, temp_state.dimension, playout_start_player, moves_left_at_end)

            # Return +1 if my score is higher, -1 if lower, 0 if equal (or very close)
            # Use a small tolerance for floating point? Scores are int here.
            if my_score > opp_score:
                 return 1.0
            elif opp_score > my_score:
                 return -1.0
            else:
                 return 0.0 # Draw or equal evaluation


    # Removed _heuristic_evaluation method as it's replaced by the score comparison above

    def find_best_move(self, game_state: 'GameState'):
        """Runs MCTS simulations and returns the most promising move. (Unchanged structure)"""
        if not self.root_node.children:
            possible_moves, move_probs = heuristic_policy_selector(game_state, probability_threshold=0.0)
            if possible_moves.shape[0] == 0: return None
            self.root_node.expand_children(possible_moves, move_probs)

        for _ in range(self.simulation_count):
            self.run_simulation(game_state)

        if not self.root_node.children:
             if DEBUG_OUTPUT: print("MCTS: No children found at root!", file=sys.stderr)
             return None

        best_action = max(self.root_node.children.items(), key=lambda item: item[1].visit_count)[0]

        if DEBUG_OUTPUT:
            print("--- MCTS Debug Info ---", file=sys.stderr)
            child_info = []
            total_visits = sum(child.visit_count for child in self.root_node.children.values())
            # Sort children by visit count for display
            sorted_children = sorted(self.root_node.children.items(), key=lambda item: item[1].visit_count, reverse=True)
            for move, node in sorted_children:
                 visit_percent = (node.visit_count / total_visits * 100) if total_visits > 0 else 0
                 # Format move tuple for printing
                 move_str = f"({move[0]},{move[1]})" 
                 child_info.append(f"Move: {move_str}, Visits: {node.visit_count} ({visit_percent:.1f}%), Value: {node.value_score:.3f}, Prior: {node.prior_probability:.4f}")
            # Print top N moves or all if fewer than N
            print("\n".join(child_info[:10]), file=sys.stderr) 
            print(f"Selected Best Move: {best_action}", file=sys.stderr)
            print("-----------------------", file=sys.stderr)

        return best_action

    def advance_tree(self, last_played_move):
        """Updates the root of the MCTS tree. (Unchanged structure)"""
        move_tuple = tuple(last_played_move)
        if move_tuple in self.root_node.children:
            if DEBUG_OUTPUT: print(f"MCTS: Advancing tree with move {move_tuple}", file=sys.stderr)
            self.root_node = self.root_node.children[move_tuple]
            self.root_node.parent = None
        else:
            if DEBUG_OUTPUT: print(f"MCTS: Move {move_tuple} not found in children. Resetting tree.", file=sys.stderr)
            self.root_node = SearchNode(parent_node=None, probability=1.0)


# --- Game Engine and GTP Interface ---
class GameState:
    """Manages the Connect6 game state. (Unchanged structure)"""
    def __init__(self, dimension=19):
        self.dimension = dimension
        self.grid = np.zeros((dimension, dimension), dtype=np.int8)
        self.active_player = BLACK
        self.is_game_finished = False
        self.winner = EMPTY
        self.total_moves = 0
        self.moves_this_turn = 0

    def reset(self, dimension=None):
        if dimension is not None: self.dimension = dimension
        self.grid = np.zeros((self.dimension, self.dimension), dtype=np.int8)
        self.active_player = BLACK
        self.is_game_finished = False
        self.winner = EMPTY
        self.total_moves = 0
        self.moves_this_turn = 0

    def get_moves_left_in_turn(self) -> int:
         if self.total_moves == 0 and self.active_player == BLACK: return 1
         else: return 2 - self.moves_this_turn

    def apply_move(self, move_coord: tuple):
        r, c = move_coord
        if not (0 <= r < self.dimension and 0 <= c < self.dimension and self.grid[r, c] == EMPTY):
             print(f"Warning: Invalid move {move_coord} applied to game state.", file=sys.stderr)
             return

        self.grid[r, c] = self.active_player
        self.total_moves += 1
        self.moves_this_turn += 1

        if self.check_and_update_winner(): return

        if (self.total_moves == 1 and self.active_player == BLACK) or self.moves_this_turn == 2:
            self.active_player = 3 - self.active_player
            self.moves_this_turn = 0

        if self.total_moves == self.dimension * self.dimension and not self.is_game_finished:
             self.is_game_finished = True
             self.winner = EMPTY

    def check_and_update_winner(self) -> bool:
        if self.is_game_finished: return True
        if self.total_moves < (WIN_CONDITION * 2 - 1): return False
        potential_winner = check_victory(self.grid, self.dimension)
        if potential_winner != EMPTY:
            self.is_game_finished = True
            self.winner = potential_winner
            return True
        return False

    def is_finished(self, check_now=False) -> tuple:
        if check_now and not self.is_game_finished:
             self.check_and_update_winner()
             if not self.is_game_finished and self.total_moves == self.dimension * self.dimension :
                 self.is_game_finished = True
                 self.winner = EMPTY
        return self.is_game_finished, self.winner

class GameEngine:
    """Handles GTP commands and orchestrates the game and AI. (Unchanged structure, uses modified MCTS)"""
    def __init__(self, board_size=19, ai_simulations=1000, exploration=1.41): # Default params
        self.game_state = GameState(board_size)
        # Ensure exploration factor matches original if needed (e.g., 1.41 or 5)
        # Original MCTS used c_puct=5, let's try that
        # Original MCTS used n_playout=2000, let's try that
        self.ai_player = MonteCarloTreeSearch(exploration_factor=exploration, simulation_limit=ai_simulations) 
        self.last_opponent_moves = []

        self.gtp_commands = {
            "boardsize": self._gtp_boardsize, "clear_board": self._gtp_clear_board,
            "play": self._gtp_play, "genmove": self._gtp_genmove,
            "showboard": self._gtp_showboard, "list_commands": self._gtp_list_commands,
            "name": self._gtp_name, "version": self._gtp_version,
            "protocol_version": self._gtp_protocol_version, "quit": self._gtp_quit,
        }

    def _col_to_char(self, col_index):
        if col_index < 0 or col_index >= self.game_state.dimension: return "?"
        return chr(ord('A') + col_index + (1 if col_index >= 8 else 0))

    def _char_to_col(self, col_char):
        char = col_char.upper()
        if not 'A' <= char <= 'T' or char == 'I': return -1
        return ord(char) - ord('A') - (1 if char > 'I' else 0)

    def _move_to_gtp(self, move_coord):
        if move_coord is None: return "PASS"
        row, col = move_coord
        return f"{self._col_to_char(col)}{row + 1}"

    def _gtp_to_move(self, gtp_coord):
        gtp_coord = gtp_coord.strip().upper()
        if len(gtp_coord) < 2: return None
        col_char = gtp_coord[0]
        row_str = gtp_coord[1:]
        col = self._char_to_col(col_char)
        try: row = int(row_str) - 1
        except ValueError: return None
        if not (0 <= row < self.game_state.dimension and 0 <= col < self.game_state.dimension): return None
        return (row, col)

    def _respond_ok(self, message=""): print(f"= {message}\n", flush=True)
    def _respond_err(self, message=""): print(f"? {message}\n", flush=True)
    def _gtp_name(self, args): self._respond_ok("RefactoredConnect6AI-v2")
    def _gtp_version(self, args): self._respond_ok("1.2-reverted-sim")
    def _gtp_protocol_version(self, args): self._respond_ok("2")
    def _gtp_list_commands(self, args): self._respond_ok("\n".join(self.gtp_commands.keys()))
    def _gtp_quit(self, args): self._respond_ok(); sys.exit(0)

    def _gtp_boardsize(self, args):
        try:
            size = int(args[0])
            if size < 6 or size > 19: raise ValueError("Size out of range")
            self.game_state.reset(dimension=size)
            self.ai_player = MonteCarloTreeSearch(self.ai_player.exploration_factor, self.ai_player.simulation_count)
            self._respond_ok()
        except (IndexError, ValueError) as e: self._respond_err(f"Invalid board size: {e}")

    def _gtp_clear_board(self, args):
        self.game_state.reset()
        self.ai_player = MonteCarloTreeSearch(self.ai_player.exploration_factor, self.ai_player.simulation_count)
        self._respond_ok()

    def _gtp_play(self, args):
        # (GTP Play logic unchanged from previous refactored version)
        if len(args) < 2: self._respond_err("Syntax error: play COLOR COORD[,COORD]"); return
        color_str = args[0].upper()
        player_id = BLACK if color_str.startswith("B") else WHITE if color_str.startswith("W") else EMPTY
        if player_id == EMPTY: self._respond_err("Invalid player color"); return
        moves_str = args[1].split(',')
        move_coords = []
        for move_s in moves_str:
            coord = self._gtp_to_move(move_s)
            if coord is None: self._respond_err(f"Invalid coordinate format: {move_s}"); return
            if self.game_state.grid[coord[0], coord[1]] != EMPTY: self._respond_err(f"Illegal move - position occupied: {move_s}"); return
            move_coords.append(coord)
        if len(move_coords) == 2 and move_coords[0] == move_coords[1]: self._respond_err("Illegal move - cannot place both stones in the same location"); return

        self.last_opponent_moves = []
        for coord in move_coords:
             self.ai_player.advance_tree(coord)
             self.game_state.apply_move(coord)
             self.last_opponent_moves.append(coord)
             if self.game_state.is_game_finished: break

        if DEBUG_OUTPUT and not self.game_state.is_game_finished:
             moves_left = self.game_state.get_moves_left_in_turn()
             p1_score, p2_score = score_grid_state(self.game_state.grid, self.game_state.dimension, BLACK, moves_left if self.game_state.active_player == BLACK else 2)
             print(f"Score after PLAY: B={p1_score}, W={p2_score}", file=sys.stderr)
        self._respond_ok()

    def _gtp_genmove(self, args):
        # (GTP Genmove logic for win/block/first move/MCTS call unchanged)
        if len(args) < 1: self._respond_err("Syntax error: genmove COLOR"); return
        color_str = args[0].upper()
        player_id = BLACK if color_str.startswith("B") else WHITE if color_str.startswith("W") else EMPTY
        if player_id == EMPTY: self._respond_err("Invalid player color"); return
        if player_id != self.game_state.active_player: self._respond_err(f"AI cannot generate move for {color_str} - it's {['BLACK','WHITE'][self.game_state.active_player-1]}'s turn"); return
        if self.game_state.is_game_finished: self._respond_err("Game is already over"); return

        num_moves_to_gen = self.game_state.get_moves_left_in_turn()
        generated_moves_gtp = []

        for i in range(num_moves_to_gen):
            best_move = None
            # First move logic
            if self.game_state.total_moves == 0 and player_id == BLACK:
                 center = (self.game_state.dimension - 1) / 2.0
                 std_dev = self.game_state.dimension / 6.0
                 while True:
                     r = int(np.round(np.random.normal(loc=center, scale=std_dev)))
                     c = int(np.round(np.random.normal(loc=center, scale=std_dev)))
                     if 0 <= r < self.game_state.dimension and 0 <= c < self.game_state.dimension and self.game_state.grid[r, c] == EMPTY:
                         best_move = (r, c); break
            else:
                 # Check immediate win
                 win_seq = find_winning_moves(self.game_state.grid, self.game_state.dimension, player_id, num_moves_to_gen - i)
                 if not np.all(win_seq == -1):
                      best_move = tuple(win_seq[0])
                      if DEBUG_OUTPUT: print(f"AI: Found winning move {best_move}", file=sys.stderr)
                 else:
                     # Check immediate block
                     opponent_id = 3 - player_id
                     block_seq = find_winning_moves(self.game_state.grid, self.game_state.dimension, opponent_id, 2)
                     if not np.all(block_seq == -1):
                         block_move = tuple(block_seq[0])
                         if self.game_state.grid[block_move[0], block_move[1]] == EMPTY:
                              best_move = block_move
                              if DEBUG_OUTPUT: print(f"AI: Found blocking move {best_move}", file=sys.stderr)
                         else:
                              if DEBUG_OUTPUT: print(f"AI: Blocking move {block_move} is occupied, using MCTS", file=sys.stderr)
                              # Fall through to MCTS if block spot taken
                     
                     # Use MCTS if no win/block found or block failed
                     if best_move is None:
                          if DEBUG_OUTPUT: print(f"AI: Using MCTS (reverted sim logic) to find move {i+1}/{num_moves_to_gen}...", file=sys.stderr)
                          best_move = self.ai_player.find_best_move(self.game_state)

            if best_move is None:
                 fallback_moves = get_valid_moves(self.game_state.dimension, self.game_state.grid)
                 if fallback_moves.shape[0] > 0: best_move = tuple(fallback_moves[0])
                 else: self._respond_err("cannot generate move"); return

            self.game_state.apply_move(best_move)
            self.ai_player.advance_tree(best_move)
            generated_moves_gtp.append(self._move_to_gtp(best_move))
            if self.game_state.is_game_finished: break

        print(f"= {','.join(generated_moves_gtp)}\n", flush=True)
        if DEBUG_OUTPUT and not self.game_state.is_game_finished:
             moves_left = self.game_state.get_moves_left_in_turn()
             p1_score, p2_score = score_grid_state(self.game_state.grid, self.game_state.dimension, BLACK, moves_left if self.game_state.active_player == BLACK else 2)
             print(f"Score after GENMOVE: B={p1_score}, W={p2_score}", file=sys.stderr)

    def _gtp_showboard(self, args):
        # (Showboard logic unchanged)
        header = "   " + " ".join(self._col_to_char(c) for c in range(self.game_state.dimension))
        board_str = [header]
        symbols = {EMPTY: '.', BLACK: 'X', WHITE: 'O'}
        for r in range(self.game_state.dimension - 1, -1, -1):
            row_str = f"{r+1:<2d} " + " ".join(symbols[self.game_state.grid[r, c]] for c in range(self.game_state.dimension))
            board_str.append(row_str)
        board_str.append(f"Turn: {['BLACK', 'WHITE'][self.game_state.active_player-1]} ({self.game_state.get_moves_left_in_turn()} moves left)")
        if self.game_state.is_game_finished:
             winner_str = {BLACK: "Black wins", WHITE: "White wins", EMPTY: "Draw"}[self.game_state.winner]
             board_str.append(f"Game Over: {winner_str}")
        self._respond_ok("\n" + "\n".join(board_str))

    def run_gtp_server(self):
        # (GTP Loop unchanged)
        if DEBUG_OUTPUT: print("GTP Engine Ready (Simulation logic reverted).", file=sys.stderr)
        while True:
            try:
                line = sys.stdin.readline().strip()
                if not line: break
                parts = line.split(maxsplit=1)
                command = parts[0].lower()
                args = parts[1].split() if len(parts) > 1 else []
                cmd_id = ""
                # Handle optional GTP command ID
                if command.isdigit():
                    cmd_id = command
                    if not args: self._respond_err(f"Command expected after ID {cmd_id}"); continue
                    command = args[0].lower()
                    args = args[1:]
                
                response_prefix = f"{cmd_id} " if cmd_id else "" # Prepare response prefix if ID exists

                if command in self.gtp_commands:
                     if DEBUG_OUTPUT: print(f"Received GTP: {line}", file=sys.stderr)
                     # Pass the prefix to handlers? No, handlers print =/? directly. Modify handlers if needed.
                     # For now, handlers ignore the ID and print standard =/?
                     self.gtp_commands[command](args) 
                else:
                     self._respond_err(f"Unknown command: {command}")
            except EOFError: break
            except KeyboardInterrupt: print("\nInterrupted.", file=sys.stderr); break
            except Exception as e:
                print(f"? Error: {str(e)}\n", flush=True)
                if DEBUG_OUTPUT: import traceback; traceback.print_exc(file=sys.stderr)

if __name__ == "__main__":
    # You might want to match parameters from the original code if they were different
    # Example: Original used c_puct=5, n_playout=2000
    engine = GameEngine(board_size=19, ai_simulations=2000, exploration=5.0) 
    engine.run_gtp_server()