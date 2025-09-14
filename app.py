import time
from collections import deque
import heapq
from typing import Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from flask import Flask, jsonify, render_template, request

# constants
TRAINING_EPOCHS = 150
LEARNING_RATE = 0.001

# deep learning model
class HeuristicCNN(nn.Module):
    def __init__(self):
        super(HeuristicCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x

# a-star algorithm implementations
class AStar:
    def __init__(self):
        self.nodes_expanded = 0

    def find_path(self, grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        self.nodes_expanded = 0
        heuristic_map = self.get_heuristic(grid, goal)
        rows, cols = grid.shape
        open_set = [(0, start)]
        came_from = {}

        g_score = np.full((rows, cols), float('inf'))
        g_score[start] = 0

        f_score = np.full((rows, cols), float('inf'))
        f_score[start] = heuristic_map[start]

        while open_set:
            _, current = heapq.heappop(open_set)
            self.nodes_expanded += 1

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (current[0] + dr, current[1] + dc)

                if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and grid[neighbor] == 0:
                    tentative_g_score = g_score[current] + 1
                    if tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + heuristic_map[neighbor]
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return None

    def get_heuristic(self, grid: np.ndarray, goal: Tuple[int, int]) -> np.ndarray:
        raise NotImplementedError

class AStarDeepLearning(AStar):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.model.eval()

    def get_heuristic(self, grid: np.ndarray, goal: Tuple[int, int]) -> np.ndarray:
        grid_tensor = torch.from_numpy(grid).unsqueeze(0).unsqueeze(0).float()
        with torch.no_grad():
            heuristic_map = self.model(grid_tensor).squeeze().numpy()
        return heuristic_map

class AStarManhattan(AStar):
    def get_heuristic(self, grid: np.ndarray, goal: Tuple[int, int]) -> np.ndarray:
        height, width = grid.shape
        y, x = np.mgrid[0:height, 0:width]
        return np.abs(y - goal[0]) + np.abs(x - goal[1])

# core logic
def generate_target_heuristic(grid: np.ndarray, goal: Tuple[int, int]) -> np.ndarray:
    rows, cols = grid.shape
    target_heuristic = np.full((rows, cols), float('inf'))
    q = deque([(goal, 0)])
    target_heuristic[goal] = 0
    visited = {goal}

    while q:
        (r, c), cost = q.popleft()
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0 and (nr, nc) not in visited:
                visited.add((nr, nc))
                target_heuristic[nr, nc] = cost + 1
                q.append(((nr, nc), cost + 1))
    return target_heuristic

def train_heuristic_model(grid: np.ndarray, goal: Tuple[int, int]) -> HeuristicCNN:
    target_heuristic_np = generate_target_heuristic(grid, goal)
    model = HeuristicCNN()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    grid_tensor = torch.from_numpy(grid).unsqueeze(0).unsqueeze(0).float()
    target_tensor = torch.from_numpy(target_heuristic_np).unsqueeze(0).unsqueeze(0).float()
    
    mask = ~torch.isinf(target_tensor)

    for _ in range(TRAINING_EPOCHS):
        optimizer.zero_grad()
        predicted_tensor = model(grid_tensor)
        loss = loss_fn(predicted_tensor[mask], target_tensor[mask])
        loss.backward()
        optimizer.step()
    
    return model

def run_pathfinding_analysis(algorithm: AStar, grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> dict:
    start_time = time.perf_counter()
    path = algorithm.find_path(grid, start, goal)
    end_time = time.perf_counter()
    
    execution_time_ms = (end_time - start_time) * 1000
    path_length = len(path) - 1 if path else 'N/A'

    return {
        "path": path,
        "nodes_expanded": algorithm.nodes_expanded,
        "time": f"{execution_time_ms:.2f}",
        "length": path_length,
    }

# flask app
app = Flask(__name__)
a_star_manhattan = AStarManhattan()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/path', methods=['POST'])
def train_and_find_path_endpoint():
    data = request.json
    numpy_grid = np.array(data['grid'])
    start_pos = tuple(data['start'])
    goal_pos = tuple(data['goal'])

    # train new model
    trained_model = train_heuristic_model(numpy_grid, goal_pos)
    a_star_dl_trained = AStarDeepLearning(trained_model)

    # analyze both algorithms
    dl_results = run_pathfinding_analysis(a_star_dl_trained, numpy_grid, start_pos, goal_pos)
    manhattan_results = run_pathfinding_analysis(a_star_manhattan, numpy_grid, start_pos, goal_pos)

    # send results
    return jsonify({
        'path_dl': dl_results['path'],
        'nodes_expanded_dl': dl_results['nodes_expanded'],
        'time_dl': dl_results['time'],
        'length_dl': dl_results['length'],
        'path_manhattan': manhattan_results['path'],
        'nodes_expanded_manhattan': manhattan_results['nodes_expanded'],
        'time_manhattan': manhattan_results['time'],
        'length_manhattan': manhattan_results['length']
    })

if __name__ == '__main__':
    app.run(debug=True)