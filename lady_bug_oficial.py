import pygame
import math
import random
import numpy as np
import matplotlib.pyplot as plt

class Envir:
    def __init__(self, dimensions):
        # colors
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.green = (0, 255, 0)
        self.blue = (0, 0, 255)
        self.red = (255, 0, 0)
        self.yellow = (255, 255, 0)
        # maps dims
        self.height = dimensions[0]
        self.width = dimensions[1]
        # window settings
        self.MapWindowName = "DDR Controlled"
        pygame.display.set_caption(self.MapWindowName)
        self.map = pygame.display.set_mode((self.width,
                                            self.height))
        # Text Variables
        self.font = pygame.font.Font('freesansbold.ttf', 20)
        self.text = self.font.render('default', True, self.white, self.black)
        self.textRect = self.text.get_rect()
        self.textRect.center = (dimensions[1] - 400, dimensions[0] - dimensions[0]/7)
        self.textRect_cur = self.text.get_rect()
        self.textRect_cur.center = (dimensions[1] - 400, dimensions[0] - dimensions[0]/7 - 40)
        self.textRect_goal = self.text.get_rect()
        self.textRect_goal.center = (dimensions[1] - 400, dimensions[0] - dimensions[0]/7 - 20)
        self.textRect_reward = self.text.get_rect()
        self.textRect_reward.center = (dimensions[1] - 400, dimensions[0] - dimensions[0] + 20)

        # Trail
        self.trail_set = []
        # self.goal0 = (600, 100, math.radians(270))
        # self.goal1 = (800, 300, math.radians(0))
        # self.goal2 = (600, 500, math.radians(90))
        # self.goal3 = (400, 300, math.radians(180))
        # Points
        self.point0 = (600, 100, math.radians(270))
        self.point1 = (800, 300, math.radians(0))
        self.point2 = (600, 500, math.radians(90))
        self.point3 = (400, 300, math.radians(180))
        self.points = (self.point0, self.point1, self.point2, self.point3)
        self.b = random.randint(0, 3)
        # self.b = 1
        self.point_ch = self.points[self.b]
        self.bandit = np.zeros(4)
        self.bandit[self.b] = 1
        self.point_size = (10, 10)


    def write_info(self, v, omega):
        txt = f"v = {v}, omega = {omega}"
        self.text = self.font.render(txt, True, self.white, self.black)
        self.map.blit(self.text, self.textRect)

    def write_info_cur(self, x, y, theta):
        txt_cur = f"x = {x}, y = {y}, theta = {int(math.degrees(theta))}"
        self.text = self.font.render(txt_cur, True, self.white, self.black)
        self.map.blit(self.text, self.textRect_cur)

    def write_info_goal(self, g_x, g_y, g_theta):
        txt_goal = f"goal_x = {g_x}, goal_y = {g_y}, goal_theta = {int(math.degrees(g_theta))}"
        self.text = self.font.render(txt_goal, True, self.white, self.black)
        self.map.blit(self.text, self.textRect_goal)

    def write_info_reward(self, n, reward):
        txt_goal = f"n = {n}, reward = {reward}"
        self.text = self.font.render(txt_goal, True, self.white, self.black)
        self.map.blit(self.text, self.textRect_reward)

    def trail(self, pos):
        for i in range(0, len(self.trail_set) - 1):
            pygame.draw.line(self.map, self.yellow, (self.trail_set[i][0], self.trail_set[i][1]),
                             (self.trail_set[i + 1][0], self.trail_set[i + 1][1]))
        if self.trail_set.__sizeof__() > 2000:
            self.trail_set.pop(0)
        self.trail_set.append(pos)

    def robot_frame(self, pos, rotation):
        n = 80

        centerx, centery = pos

        x_axis = (centerx + n * math.cos(rotation), centery + n * math.sin(rotation))
        y_axis = (centerx + n * math.cos(rotation + math.pi/2), centery + n * math.sin(rotation + math.pi/2))
        pygame.draw.line(self.map, self.red, (centerx, centery), x_axis, 3)
        pygame.draw.line(self.map, self.green, (centerx, centery), y_axis, 3)

    def coord(self):
        x, y = 0, 0
        pygame.draw.line(self.map, self.white, (x, y), (x+100, y), 3)
        pygame.draw.line(self.map, self.white, (x, y), (x, y+100), 3)

    def point(self):
        pygame.draw.rect(self.map, self.white, self.point_ch[0:2]+self.point_size)


class Robot:
    def __init__(self, startpos, goalpos, robotImg, width):
        self.m2p = 3779.52  # meter to pixels

        # Robot dimension
        self.w = width
        self.x = startpos[0]
        self.y = startpos[1]
        self.theta = startpos[2]
        self.goalx = goalpos[0]
        self.goaly = goalpos[1]
        self.goaltheta = goalpos[2]
        self.a = 20
        self.v = 0 * self.m2p
        self.omega = 1 * self.m2p
        self.maxspeed = 1 * self.m2p
        self.minspeed = - 1 * self.m2p
        self.reward = 0
        # self.nodes = 0

        # Parameters of egreedy

        self.last_atitude = "Random"
        self.action = 4
        self.n = 0
        self.A = []
        self.R = []
        self.Q = []
        self.N = []
        self.n_arms = np.arange(4)
        self.OCC = np.zeros(4)

        # Graphics ######### ACERTAR A IMAGEM DO ROBO
        self.img = pygame.image.load(robotImg)
        self.rotated = self.img
        self.rect = self.rotated.get_rect(center=(self.x, self.y))

        # goals
        self.centerpoint = (600, 300)
        self.goal0 = (600, 100, math.radians(270))
        self.goal1 = (800, 300, math.radians(0))
        self.goal2 = (600, 500, math.radians(90))
        self.goal3 = (400, 300, math.radians(180))
        self.goals = (self.goal0, self.goal1, self.goal2, self.goal3)

    def ajusteAngulo(self, angulo):
        angulo = np.mod(angulo, 2 * np.pi)
        if angulo > np.pi:
            angulo = angulo - 2 * np.pi
        return angulo

    def Controller(self, Krho=2/11, Kalpha=345/823, Kbeta=-2/11, forward=False): #Krho=2/11, Kalpha=345/823, Kbeta=-2/11,

        self.Krho = Krho
        self.Kalpha = Kalpha
        self.Kbeta = Kbeta

        Dx = self.goalx - self.x
        Dy = self.goaly - self.y
        Dth = self.goaltheta - self.theta

        # Control just for line
        #self.v = Dx * math.cos(self.theta) + Dy * math.sin(self.theta)
        #self.omega = (-1/self.a) * math.sin(self.theta) * Dx + (1/self.a) * math.cos(self.theta) * Dy

        # P control

        rho = np.sqrt(Dx**2 + Dy **2)
        gamma = self.ajusteAngulo(math.atan2(Dy, Dx))

        alpha = self.ajusteAngulo(gamma - self.theta)

        beta = self.ajusteAngulo(self.goaltheta - gamma)

        self.v = min(self.Krho * rho, self.maxspeed)
        # Walk forward or backward
        if forward == False:
            if np.abs(alpha) > math.pi / 2:
                 self.v = -self.v
                 alpha = self.ajusteAngulo(alpha + math.pi)
                 beta = self.ajusteAngulo(beta + math.pi)
        self.omega = self.Kalpha * alpha + self.Kbeta * beta

    def Reinforcement(self, pointok, bandit, epsilon):

        self.Controller()

        self.epsilon = epsilon
        nmax = 500

        Dx = self.goalx - self.x
        Dy = self.goaly - self.y

        self.point_x = pointok[0]
        self.point_y = pointok[1]
        dx_point = self.x - self.point_x
        dy_point = self.y - self.point_y
        rho_point = np.sqrt(dx_point**2 + dy_point**2)

        rho = np.sqrt(Dx**2 + Dy**2)

        rho_center = np.sqrt((self.x - self.centerpoint[0])**2 + (self.y - self.centerpoint[1])**2)

        if self.goalx == self.goal0[0] and robot.goaly == self.goal0[1]:
            self.action = 0
        elif self.goalx == self.goal1[0] and robot.goaly == self.goal1[1]:
            self.action = 1
        elif self.goalx == self.goal2[0] and robot.goaly == self.goal2[1]:
            self.action = 2
        elif self.goalx == self.goal3[0] and robot.goaly == self.goal3[1]:
            self.action = 3

        if rho <= 0.01 and rho_center >= 0.01:
            # self.Controller(forward=False)
            self.goalx = self.centerpoint[0]
            self.goaly = self.centerpoint[1]

            self.A.append(self.action)
            self.R.append(bandit[self.action])
            self.N.append(self.n)
            if self.n == 0:
                self.Q.append((1 / (self.n + 1)) * self.R[self.n])
            else:
                self.Q.append(self.Q[self.n - 1] + (1 / self.n) * (self.R[self.n] - self.Q[self.n - 1]))
            self.OCC[self.action] += 1
            self.n = self.n + 1
            # self.nodes = self.nodes + 1

            if rho_point <= 0.02:
                self.reward = self.reward + 1

        if rho <= 0.1 and rho_center <= 0.01:
            # self.Controller(forward=True)
            choice = random.uniform(0, 1) < epsilon #0.14~
            if choice == 0:
                self.last_atitude = "exploit"
            else:
                self.last_atitude = "explore"

            if choice == 0:
                b = max(self.R)
                a1 = self.R.index(b)
                self.action = self.A[a1]
            else:
                self.action = random.randint(0, 3)

            print(self.action)
            print(self.last_atitude)
            print(self.A)

            ch_goal = self.goals[self.action]
            self.goalx = ch_goal[0]
            self.goaly = ch_goal[1]
            self.goaltheta = ch_goal[2]

        if self.n >= nmax:
            pygame.quit()

    def draw(self, map):
        map.blit(self.rotated, self.rect)

    def move(self, event=None):

        self.x += (self.v * math.cos(self.theta)) * dt #- (self.a * math.sin(self.theta) * self.omega) * dt
        self.y += (self.v * math.sin(self.theta)) * dt #+ (self.a * math.cos(self.theta) * self.omega) * dt
        self.theta += self.omega * dt
        # Reset theta over 1 lap complete
        if self.theta > 2 * math.pi or self.theta < - 2 * math.pi:
            self.theta = 0

        self.rotated = pygame.transform.rotozoom(self.img, math.degrees(-self.theta), 1)
        self.rect = self.rotated.get_rect(center=(self.x, self.y))

        # self.Controller()
        self.Reinforcement(environment.point_ch, environment.bandit, 1)

        if event is not None:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_KP4:
                    self.goalx += 0.001 * self.m2p
                elif event.key == pygame.K_KP1:
                    self.goalx -= 0.001 * self.m2p
                elif event.key == pygame.K_KP6:
                    self.goaly += 0.001 * self.m2p
                elif event.key == pygame.K_KP3:
                    self.goaly -= 0.001 * self.m2p
                elif event.key == pygame.K_KP7:
                    self.goaltheta += math.degrees(0.00033)
                elif event.key == pygame.K_KP9:
                    self.goaltheta -= math.degrees(0.00033)


# Initialisation
pygame.init()

# Start Position
start = (600, 300, math.radians(0))
goal = (600, 100, math.radians(-90))

# Dimensions
dims = (600, 1200)

# Score Points
point_size = (10, 10)
point0 = (600, 100, math.radians(270))
point1 = (800, 300, math.radians(0))
point2 = (600, 500, math.radians(90))
point3 = (400, 300, math.radians(180))
points = (point0, point1, point2, point3)
center_point = (600, 300, math.radians(0))
sorted_point = random.choice(points)
goal = sorted_point

# Running or not
running = True

# The enviroment
environment = Envir(dims)

# The robot
robot = Robot(start, goal, "/home/julio/Documentos/Projetos/TCC/testerobozinho.png", 0.01 * 3779.52)

dt = 0
lasttime = pygame.time.get_ticks()
# Simulation loop
while running:
    # activate the quit button
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
        robot.move(event)

    dt = (pygame.time.get_ticks() - lasttime) / 100
    lasttime = pygame.time.get_ticks()
    pygame.display.update()
    environment.map.fill(environment.black)
    robot.move()
    robot.draw(environment.map)
    environment.point()
    environment.trail((robot.x, robot.y))
    environment.robot_frame((robot.x, robot.y), robot.theta)
    environment.write_info('%.3f'%(robot.v), '%.4f'%(robot.omega))
    environment.write_info_cur(int(robot.x), int(robot.y), robot.theta)
    environment.write_info_goal(int(robot.goalx), int(robot.goaly), robot.goaltheta)
    environment.write_info_reward(robot.n, robot.reward)
    environment.coord()

# Plot do robô
fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 6))
    # Plot de recompensa ao longo da run
ax1.set_xlabel("Passo")
ax1.set_ylabel("Recompensa média")
ax1.plot(robot.N, robot.Q)
    # Plot ações ao longo da run
labels = ["Norte", "Leste", "Sul", "Oeste"]
ticks = np.arange(len(labels))
ax2.set_yticks(ticks)
ax2.set_yticklabels(labels)
s = 8 * np.ones(len(robot.A))
ax2.scatter(robot.N, robot.A, s=s)

fig.tight_layout()
plt.show()

# Plot de ocorrência por direção
fig, ax = plt.subplots()
width = 0.35
ax.set_title("Ocorrência em cada ponto")
labels = ["Norte", "Leste", "Sul", "Oeste"]
ticks = np.arange(len(labels))
ax.set_xticks(ticks)
ax.set_xticklabels(labels)
ax.bar(robot.n_arms, robot.OCC, width=width)

fig.tight_layout()
plt.show()
