import sys
import heapq
import time
from PIL import Image, ImageDraw
from collections import deque
import random

class Node:
    def __init__(self, state, parent, action, cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost

    def __lt__(self, other):
        return self.cost < other.cost


class PriorityQueue:
    def __init__(self):
        self.frontier = []

    def add(self, node):
        heapq.heappush(self.frontier, node)

    def contains_state(self, state):
        return any(node.state == state for node in self.frontier)

    def empty(self):
        return len(self.frontier) == 0

    def remove(self):
        if self.empty():
            raise Exception("empty frontier")
        return heapq.heappop(self.frontier)


class Maze:
    def __init__(self, filename=None, height=10, width=10, obstacle_percentage=0.3):
        if filename:
            with open(filename) as f:
                contents = f.read()

            if contents.count("A") != 1 or contents.count("B") != 1:
                raise Exception("Maze must have exactly one start and one goal")

            contents = contents.splitlines()
            self.height = len(contents)
            self.width = max(len(line) for line in contents)

            self.walls = []
            for i in range(self.height):
                row = []
                for j in range(self.width):
                    try:
                        if contents[i][j] == "A":
                            self.start = (i, j)
                            row.append(False)
                        elif contents[i][j] == "B":
                            self.goal = (i, j)
                            row.append(False)
                        elif contents[i][j] == " ":
                            row.append(False)
                        else:
                            row.append(True)
                    except IndexError:
                        row.append(False)
                self.walls.append(row)
        else:
            self.height = height
            self.width = width
            self.start = (0, 0)
            self.goal = (self.height - 1, self.width - 1)
            self.walls = self.generate_maze(obstacle_percentage)

        self.solution = None

    def generate_maze(self, obstacle_percentage):
        maze = [[False for _ in range(self.width)] for _ in range(self.height)]
        for i in range(self.height):
            for j in range(self.width):
                if random.random() < obstacle_percentage:
                    maze[i][j] = True
        maze[self.start[0]][self.start[1]] = False
        maze[self.goal[0]][self.goal[1]] = False
        return maze

    def manhattan_distance(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def chebyshev_distance(self, a, b):
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    def neighbors(self, state):
        row, col = state
        candidates = [
            ("up", (row - 1, col)),
            ("down", (row + 1, col)),
            ("left", (row, col - 1)),
            ("right", (row, col + 1))
        ]
        result = []
        for action, (r, c) in candidates:
            if 0 <= r < self.height and 0 <= c < self.width and not self.walls[r][c]:
                result.append((action, (r, c)))
        return result

    def solve_greedy(self, heuristic):
        self.num_explored = 0
        frontier = PriorityQueue()
        frontier.add(Node(self.start, None, None, heuristic(self.start, self.goal)))
        self.explored = set()

        while not frontier.empty():
            node = frontier.remove()
            self.num_explored += 1

            if node.state == self.goal:
                actions, cells = [], []
                while node.parent:
                    actions.append(node.action)
                    cells.append(node.state)
                    node = node.parent
                actions.reverse()
                cells.reverse()
                self.solution = (actions, cells)
                return

            self.explored.add(node.state)
            for action, state in self.neighbors(node.state):
                if not frontier.contains_state(state) and state not in self.explored:
                    frontier.add(Node(state, node, action, heuristic(state, self.goal)))

        print("No solution found using Greedy with the current setup")
        return None

    def solve_a_star(self, heuristic):
        self.num_explored = 0
        frontier = PriorityQueue()
        frontier.add(Node(self.start, None, None, 0))
        self.explored = set()
        g_costs = {self.start: 0}

        while not frontier.empty():
            node = frontier.remove()
            self.num_explored += 1

            if node.state == self.goal:
                actions, cells = [], []
                while node.parent:
                    actions.append(node.action)
                    cells.append(node.state)
                    node = node.parent
                actions.reverse()
                cells.reverse()
                self.solution = (actions, cells)
                return

            self.explored.add(node.state)
            for action, state in self.neighbors(node.state):
                new_cost = g_costs[node.state] + 1
                if state not in g_costs or new_cost < g_costs[state]:
                    g_costs[state] = new_cost
                    f_cost = new_cost + heuristic(state, self.goal)
                    frontier.add(Node(state, node, action, f_cost))

        print("No solution found using A* with the current setup")
        return None

    def output_image(self, filename, show_solution=True, show_explored=False):
        cell_size = 20
        cell_border = 2

        img = Image.new("RGBA", (self.width * cell_size, self.height * cell_size), "black")
        draw = ImageDraw.Draw(img)

        solution = self.solution[1] if self.solution is not None else None
        for i, row in enumerate(self.walls):
            for j, col in enumerate(row):
                if col:
                    fill = (40, 40, 40)
                elif (i, j) == self.start:
                    fill = (255, 0, 0)
                elif (i, j) == self.goal:
                    fill = (0, 171, 28)
                elif solution is not None and show_solution and (i, j) in solution:
                    fill = (220, 235, 113)
                elif show_explored and (i, j) in self.explored:
                    fill = (212, 97, 85)
                else:
                    fill = (237, 240, 252)

                draw.rectangle(
                    [(j * cell_size + cell_border, i * cell_size + cell_border),
                     ((j + 1) * cell_size - cell_border, (i + 1) * cell_size - cell_border)],
                    fill=fill
                )

        filename = filename.replace("*", "_")
        img.save(filename)
        print(f"Saved maze solution as {filename}")


def compare_performance(maze_sizes):
    heuristics = {"Manhattan": "manhattan_distance", "Chebyshev": "chebyshev_distance"}
    algorithms = {"Greedy": "solve_greedy", "A*": "solve_a_star"}

    for height, width in maze_sizes:
        print(f"\nComparing performance for maze size {height}x{width}...\n")
        m = Maze(height=height, width=width, obstacle_percentage=0.2)
        for h_name, h_method in heuristics.items():
            for algo_name, method in algorithms.items():
                print(f"Running {algo_name} with {h_name} heuristic for maze {height}x{width}...")
                start_time = time.time()
                getattr(m, method)(getattr(m, h_method))  
                end_time = time.time()
                m.output_image(f"{height}x{width}_{algo_name.lower()}_{h_name.lower()}_solution.png", show_explored=True)
                print(f"{algo_name} ({h_name}) Performance for {height}x{width}:")
                print(f"  - Nodes Expanded: {m.num_explored}")
                print(f"  - Path Cost: {len(m.solution[1]) if m.solution else None}")
                print(f"  - Execution Time: {end_time - start_time:.6f} seconds")

# Define maze sizes
maze_sizes = [(30, 30), (35, 35), (40, 40)]  # Sizes to compare
compare_performance(maze_sizes)
