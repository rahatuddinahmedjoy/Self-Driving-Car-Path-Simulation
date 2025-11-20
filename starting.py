"""
self_driving_sim.py

Self-Driving Car Path Simulation (advanced-ish):
- Grid map with static obstacles
- A* path planning on grid
- Simple kinematic car following path (pure-pursuit style)
- Replanning if obstacle appears on path
- Visualization with pygame

Run:
    python self_driving_sim.py

Requirements:
    pip install pygame numpy
"""

import pygame
import sys
import math
import heapq
import random
import numpy as np
from collections import deque

# --------------------------
# Configuration
# --------------------------
GRID_W = 40            # grid columns
GRID_H = 25            # grid rows
CELL = 24              # pixels per grid cell
SCREEN_W = GRID_W * CELL
SCREEN_H = GRID_H * CELL
FPS = 60

# Planner/config
OBSTACLE_DENSITY = 0.12   # fraction of grid cells blocked initially
ALLOW_DYNAMIC_OBSTACLES = True
DYNAMIC_OBS_FREQ = 4.0    # seconds between toggling a dynamic obstacle (or 0 to disable)

# Car params (kinematic)
MAX_STEER = math.radians(30)   # max steering angle
LOOKAHEAD = 3.0                 # lookahead distance in cells for pure-pursuit
SPEED = 2.5                     # cells per second
CAR_RADIUS = 0.35               # for collision check (in cells)

# Colors
WHITE = (255,255,255)
BLACK = (0,0,0)
GRAY = (200,200,200)
BLUE = (40,120,255)
RED = (220,40,40)
GREEN = (50,200,50)
ORANGE = (255,160,0)
PURPLE = (180,60,180)

# --------------------------
# Utility - A* planner on grid
# --------------------------
def heuristic(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def neighbors(node):
    x,y = node
    for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
        yield (x+dx, y+dy)

def astar(start, goal, grid_blocked):
    # start/goal: integer grid coords (x, y)
    # grid_blocked: set of blocked coords
    frontier = []
    heapq.heappush(frontier, (0, start))
    came = {start: None}
    cost = {start: 0}
    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            break
        for n in neighbors(current):
            nx, ny = n
            if not (0 <= nx < GRID_W and 0 <= ny < GRID_H):
                continue
            if n in grid_blocked:
                continue
            new_cost = cost[current] + 1
            if n not in cost or new_cost < cost[n]:
                cost[n] = new_cost
                priority = new_cost + heuristic(n, goal)
                heapq.heappush(frontier, (priority, n))
                came[n] = current
    if goal not in came:
        return None
    # reconstruct path
    path = []
    cur = goal
    while cur != start:
        path.append(cur)
        cur = came[cur]
    path.append(start)
    path.reverse()
    return path

# --------------------------
# Kinematic car & controller
# --------------------------
class Car:
    def __init__(self, x, y, theta=0.0):
        # x,y in grid coordinates (float)
        self.x = float(x)
        self.y = float(y)
        self.theta = theta   # heading in radians (0 = right, pi/2 = down)
        self.speed = SPEED   # cells / second
        self.path = []       # list of grid cells to follow (tuples)
        self.path_goal_idx = 0

    def world_to_pixel(self, cx, cy):
        return int(cx * CELL + CELL/2), int(cy * CELL + CELL/2)

    def step(self, dt, grid_blocked):
        # follow path using a simple lookahead target (pure pursuit style)
        if not self.path:
            return
        # compute lookahead point along path measured in Euclidean distance
        px, py = self.x, self.y
        # find first path point beyond lookahead
        target = None
        for i in range(self.path_goal_idx, len(self.path)):
            gx, gy = self.path[i]
            dist = math.hypot(gx - px, gy - py)
            if dist >= LOOKAHEAD or i == len(self.path)-1:
                target = (gx, gy)
                self.path_goal_idx = i
                break
        if target is None:
            target = self.path[-1]
            self.path_goal_idx = len(self.path)-1

        tx, ty = target
        # vector from car to target in local frame
        dx = tx - px
        dy = ty - py
        # desired heading
        desired_theta = math.atan2(dy, dx)
        # steering: smallest angular difference
        angle_diff = (desired_theta - self.theta + math.pi) % (2*math.pi) - math.pi
        steer = max(-MAX_STEER, min(MAX_STEER, angle_diff))
        # update pose (simple bicycle-like forward step, but with instantaneous heading change limited by steer)
        # rotate heading by a fraction depending on steer
        # For simplicity, we map steer to heading change proportional to dt
        heading_change = steer * 1.5 * dt  # factor to make turning responsive
        self.theta += heading_change
        # move forward
        self.x += math.cos(self.theta) * self.speed * dt
        self.y += math.sin(self.theta) * self.speed * dt

        # collision check with blocked cells: if collided, report True
        if check_collision((self.x, self.y), grid_blocked):
            return True
        return False

# --------------------------
# Collision detection helper
# --------------------------
def check_collision(pos, grid_blocked):
    # pos: (x,y) float in grid coords
    x,y = pos
    # check neighbors within radius
    r = CAR_RADIUS
    minx = int(math.floor(x-r))
    maxx = int(math.floor(x+r))
    miny = int(math.floor(y-r))
    maxy = int(math.floor(y+r))
    for gx in range(minx, maxx+1):
        for gy in range(miny, maxy+1):
            if (gx,gy) in grid_blocked:
                # check precise collision using circle-rect (cell centered)
                cx = gx + 0.5
                cy = gy + 0.5
                # closest point on rect
                closest_x = max(gx, min(x, gx+1))
                closest_y = max(gy, min(y, gy+1))
                dist = math.hypot(x-closest_x, y-closest_y)
                if dist < CAR_RADIUS:
                    return True
    return False

# --------------------------
# Visualization
# --------------------------
def draw_grid(screen, blocked, start, goal, path=None):
    # draw cells
    for x in range(GRID_W):
        for y in range(GRID_H):
            rect = pygame.Rect(x*CELL, y*CELL, CELL, CELL)
            pygame.draw.rect(screen, GRAY, rect, 1)
    # blocked cells
    for (x,y) in blocked:
        rect = pygame.Rect(x*CELL, y*CELL, CELL, CELL)
        pygame.draw.rect(screen, BLACK, rect)
    # path
    if path:
        for i in range(len(path)-1):
            a = path[i]; b = path[i+1]
            ax = a[0]*CELL + CELL/2; ay = a[1]*CELL + CELL/2
            bx = b[0]*CELL + CELL/2; by = b[1]*CELL + CELL/2
            pygame.draw.line(screen, ORANGE, (ax,ay), (bx,by), 3)
    # start/goal
    sx = start[0]*CELL + CELL/2; sy = start[1]*CELL + CELL/2
    gx = goal[0]*CELL + CELL/2; gy = goal[1]*CELL + CELL/2
    pygame.draw.circle(screen, GREEN, (sx,sy), CELL//3)
    pygame.draw.circle(screen, RED, (gx,gy), CELL//3)

def draw_car(screen, car):
    px = int(car.x*CELL + CELL/2)
    py = int(car.y*CELL + CELL/2)
    # draw heading arrow
    head = (int(px + math.cos(car.theta)*CELL*0.7), int(py + math.sin(car.theta)*CELL*0.7))
    pygame.draw.circle(screen, BLUE, (px,py), int(CELL*0.35))
    pygame.draw.line(screen, WHITE, (px,py), head, 3)

# --------------------------
# Main simulation
# --------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    clock = pygame.time.Clock()
    pygame.display.set_caption("Self-Driving Car Path Simulation")

    # build random map with obstacles
    blocked = set()
    for x in range(GRID_W):
        for y in range(GRID_H):
            if random.random() < OBSTACLE_DENSITY:
                blocked.add((x,y))
    # clear a corridor near edges to avoid impossible starts
    for x in range(3):
        for y in range(GRID_H):
            blocked.discard((x,y))
    for x in range(GRID_W-3, GRID_W):
        for y in range(GRID_H):
            blocked.discard((x,y))

    # random start and goal in free space
    def random_free():
        while True:
            a = (random.randrange(2, GRID_W-2), random.randrange(2, GRID_H-2))
            if a not in blocked:
                return a
    start = random_free()
    goal = random_free()

    # Instantiate car at start (center)
    car = Car(start[0]+0.5, start[1]+0.5, theta=0.0)

    # initial planning
    path = astar((int(math.floor(car.x)), int(math.floor(car.y))), goal, blocked)
    if path is None:
        print("Initial path not found — regenerate map")
        pygame.quit(); sys.exit(0)
    car.path = path
    car.path_goal_idx = 0

    # dynamic obstacles toggle
    last_dyn_time = 0.0
    dynamic_obs = set()

    running = True
    total_time = 0.0
    reroute_count = 0

    font = pygame.font.SysFont(None, 20)

    while running:
        dt = clock.tick(FPS) / 1000.0
        total_time += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # allow resetting start/goal by mouse click
            if event.type == pygame.MOUSEBUTTONDOWN:
                mx,my = pygame.mouse.get_pos()
                gx = mx // CELL; gy = my // CELL
                if 0 <= gx < GRID_W and 0 <= gy < GRID_H:
                    if event.button == 1:  # left click sets goal
                        goal = (gx, gy)
                        # replan immediately
                        path = astar((int(math.floor(car.x)), int(math.floor(car.y))), goal, blocked)
                        if path:
                            car.path = path; car.path_goal_idx = 0
                            reroute_count += 1

        # dynamic obstacles: occasionally flip a cell blocked/unblocked
        if ALLOW_DYNAMIC_OBSTACLES and DYNAMIC_OBS_FREQ > 0:
            if total_time - last_dyn_time > DYNAMIC_OBS_FREQ:
                last_dyn_time = total_time
                # choose a random free cell and make it blocked (or toggle)
                rx = random.randrange(2, GRID_W-2)
                ry = random.randrange(2, GRID_H-2)
                if (rx,ry) not in blocked and (rx,ry) != (int(car.x), int(car.y)) and (rx,ry) != goal:
                    blocked.add((rx,ry))
                    dynamic_obs.add((rx,ry))
                else:
                    # maybe remove a dynamic obstacle
                    if dynamic_obs:
                        removed = dynamic_obs.pop()
                        blocked.discard(removed)

        # step car
        collision = car.step(dt, blocked)
        if collision:
            # if collided, stop and replan from nearest grid cell
            cx = int(math.floor(car.x)); cy = int(math.floor(car.y))
            path2 = astar((cx, cy), goal, blocked)
            if path2:
                car.path = path2; car.path_goal_idx = 0
                reroute_count += 1
        else:
            # check if car's current grid cell is blocked by new obstacles
            cur_cell = (int(math.floor(car.x)), int(math.floor(car.y)))
            # if next path node is blocked, replan
            needs_replan = False
            for p in car.path[car.path_goal_idx:]:
                if p in blocked:
                    needs_replan = True; break
            if needs_replan:
                path2 = astar(cur_cell, goal, blocked)
                if path2:
                    car.path = path2; car.path_goal_idx = 0
                    reroute_count += 1

        # draw
        screen.fill((30,30,30))
        draw_grid(screen, blocked, start, goal, path=car.path)
        draw_car(screen, car)

        # HUD
        txt1 = font.render(f"Start: {start}  Goal: {goal}", True, WHITE)
        txt2 = font.render(f"Car pos: ({car.x:.2f},{car.y:.2f})  Theta: {math.degrees(car.theta):.1f} deg", True, WHITE)
        txt3 = font.render(f"Reroutes: {reroute_count}", True, WHITE)
        screen.blit(txt1, (6, 6))
        screen.blit(txt2, (6, 26))
        screen.blit(txt3, (6, 46))

        pygame.display.flip()

        # check success (car reached goal cell)
        gx, gy = goal
        if math.hypot(car.x - (gx+0.5), car.y - (gy+0.5)) < 0.6:
            print("Goal reached!")
            running = False

    pygame.quit()
    print("Simulation ended.")

if __name__ == "__main__":
    main()
