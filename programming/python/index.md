import os
import random
import time
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
#                 SPIDER AI FC - CONFIG
# ============================================================

ACTIONS = [
    "UP",
    "DOWN",
    "LEFT",
    "RIGHT",
    "PASS",
    "SHOOT",
    "DRIBBLE",
    "DEFEND"
]

STATE_SIZE = 16
ACTION_SIZE = len(ACTIONS)

GRID_W = 14
GRID_H = 8

MAX_STEPS = 500

GAMMA = 0.98
LEARNING_RATE = 0.0005

MEMORY_SIZE = 50000
BATCH_SIZE = 64

EPSILON = 1.0
EPSILON_MIN = 0.03
EPSILON_DECAY = 0.997

TARGET_UPDATE = 20

MODEL_FILE = "SPIDER_AI_FC.pt"


# ============================================================
#                    DQN NETWORK
# ============================================================

class DQN(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(STATE_SIZE, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, ACTION_SIZE)
        )

    def forward(self, x):

        return self.network(x)


# ============================================================
#                  REPLAY MEMORY
# ============================================================

class ReplayMemory:

    def __init__(self):

        self.memory = deque(
            maxlen=MEMORY_SIZE
        )

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        self.memory.append(
            (
                state.copy(),
                action,
                reward,
                next_state.copy(),
                done
            )
        )

    def sample(self):

        batch = random.sample(
            self.memory,
            BATCH_SIZE
        )

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for item in batch:

            s, a, r, ns, d = item

            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)

        return (
            np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_states, dtype=np.float32),
            np.asarray(dones, dtype=np.float32)
        )

    def __len__(self):

        return len(self.memory)


# ============================================================
#                     SPIDER BRAIN
# ============================================================

class SpiderBrain:

    def __init__(self):

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            "SPIDER AI DEVICE:",
            self.device
        )

        self.policy = DQN().to(
            self.device
        )

        self.target = DQN().to(
            self.device
        )

        self.target.load_state_dict(
            self.policy.state_dict()
        )

        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=LEARNING_RATE
        )

        self.memory = ReplayMemory()

        self.epsilon = EPSILON

        self.training_steps = 0

    # --------------------------------------------------------

    def choose_action(
        self,
        state,
        training=True
    ):

        if training:

            if random.random() < self.epsilon:

                return random.randrange(
                    ACTION_SIZE
                )

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():

            values = self.policy(
                state_tensor
            )

        return int(
            values.argmax(
                dim=1
            ).item()
        )

    # --------------------------------------------------------

    def get_probabilities(self, state):

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():

            q_values = self.policy(
                state_tensor
            )

            probabilities = torch.softmax(
                q_values,
                dim=1
            )

        return probabilities[
            0
        ].cpu().numpy()

    # --------------------------------------------------------

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        self.memory.add(
            state,
            action,
            reward,
            next_state,
            done
        )

    # --------------------------------------------------------

    def learn(self):

        if len(self.memory) < BATCH_SIZE:

            return 0.0

        (
            states,
            actions,
            rewards,
            next_states,
            dones
        ) = self.memory.sample()

        states = torch.tensor(
            states,
            dtype=torch.float32,
            device=self.device
        )

        actions = torch.tensor(
            actions,
            dtype=torch.long,
            device=self.device
        ).unsqueeze(1)

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32,
            device=self.device
        )

        next_states = torch.tensor(
            next_states,
            dtype=torch.float32,
            device=self.device
        )

        dones = torch.tensor(
            dones,
            dtype=torch.float32,
            device=self.device
        )

        current_q = self.policy(
            states
        ).gather(
            1,
            actions
        ).squeeze(1)

        with torch.no_grad():

            next_q = self.target(
                next_states
            ).max(
                1
            ).values

            target_q = (
                rewards
                +
                GAMMA * next_q * (1 - dones)
            )

        loss = nn.functional.smooth_l1_loss(
            current_q,
            target_q
        )

        self.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(),
            5.0
        )

        self.optimizer.step()

        self.training_steps += 1

        self.epsilon = max(
            EPSILON_MIN,
            self.epsilon * EPSILON_DECAY
        )

        return float(
            loss.item()
        )

    # --------------------------------------------------------

    def update_target(self):

        self.target.load_state_dict(
            self.policy.state_dict()
        )

    # --------------------------------------------------------

    def save(self):

        torch.save(
            {
                "policy":
                    self.policy.state_dict(),

                "target":
                    self.target.state_dict(),

                "epsilon":
                    self.epsilon,

                "training_steps":
                    self.training_steps
            },
            MODEL_FILE
        )

        print(
            "MODEL SAVED:",
            MODEL_FILE
        )

    # --------------------------------------------------------

    def load(self):

        if not os.path.exists(
            MODEL_FILE
        ):

            return False

        data = torch.load(
            MODEL_FILE,
            map_location=self.device
        )

        self.policy.load_state_dict(
            data["policy"]
        )

        self.target.load_state_dict(
            data["target"]
        )

        self.epsilon = data.get(
            "epsilon",
            EPSILON_MIN
        )

        self.training_steps = data.get(
            "training_steps",
            0
        )

        print(
            "MODEL LOADED"
        )

        return True


# ============================================================
#                    SAFETY SYSTEM
# ============================================================

class SafetySystem:

    def validate(
        self,
        action,
        state,
        probabilities
    ):

        # منع قرارات غير منطقية ذات ثقة منخفضة

        confidence = float(
            probabilities[action]
        )

        if confidence < 0.05:

            return 0

        # لا تسمح بالتسديد في ظروف بعيدة
        # state[9] = المسافة التقريبية للمرمى

        if action == 5:

            goal_distance = state[9]

            if goal_distance > 0.55:

                return 4

        return action


# ============================================================
#                  CONTROLLER - SIMULATION
# ============================================================

class SafeController:

    def send(
        self,
        action_id
    ):

        action = ACTIONS[
            action_id
        ]

        print(
            "[SIMULATED CONTROLLER]",
            action
        )


# ============================================================
#                  FOOTBALL ENVIRONMENT
# ============================================================

class SoccerEnvironment:

    def reset(self):

        self.player = np.array(
            [2.0, 4.0],
            dtype=np.float32
        )

        self.ball = np.array(
            [5.0, 4.0],
            dtype=np.float32
        )

        self.opponent = np.array(
            [
                8.0,
                float(
                    random.randint(
                        1,
                        GRID_H - 2
                    )
                )
            ],
            dtype=np.float32
        )

        self.goal = np.array(
            [13.0, 4.0],
            dtype=np.float32
        )

        self.steps = 0

        self.last_action = 0

        return self.get_state()

    # --------------------------------------------------------

    def normalize(
        self,
        position
    ):

        return position / np.array(
            [
                GRID_W - 1,
                GRID_H - 1
            ],
            dtype=np.float32
        )

    # --------------------------------------------------------

    def get_state(self):

        player = self.normalize(
            self.player
        )

        ball = self.normalize(
            self.ball
        )

        opponent = self.normalize(
            self.opponent
        )

        goal = self.normalize(
            self.goal
        )

        ball_distance = (
            np.linalg.norm(
                self.player -
                self.ball
            ) / 15.0
        )

        goal_distance = (
            np.linalg.norm(
                self.ball -
                self.goal
            ) / 15.0
        )

        opponent_distance = (
            np.linalg.norm(
                self.player -
                self.opponent
            ) / 15.0
        )

        return np.array(
            [

                player[0],
                player[1],

                ball[0],
                ball[1],

                opponent[0],
                opponent[1],

                goal[0],
                goal[1],

                ball_distance,
                goal_distance,

                opponent_distance,

                self.steps /
                MAX_STEPS,

                1.0
                if self.last_action == 4
                else 0.0,

                1.0
                if self.last_action == 5
                else 0.0,

                1.0
                if self.last_action == 6
                else 0.0,

                1.0
                if self.last_action == 7
                else 0.0

            ],
            dtype=np.float32
        )

    # --------------------------------------------------------

    def step(
        self,
        action
    ):

        reward = -0.01

        self.steps += 1

        self.last_action = action

        # ---------------- MOVEMENT ----------------

        if action == 0:

            self.player[1] -= 1

        elif action == 1:

            self.player[1] += 1

        elif action == 2:

            self.player[0] -= 1

        elif action == 3:

            self.player[0] += 1

        # حدود الملعب

        self.player[0] = np.clip(
            self.player[0],
            0,
            GRID_W - 1
        )

        self.player[1] = np.clip(
            self.player[1],
            0,
            GRID_H - 1
        )

        # ---------------- DRIBBLE ----------------

        if action == 6:

            distance = np.linalg.norm(
                self.player -
                self.ball
            )

            if distance < 2.0:

                self.ball[0] = min(
                    GRID_W - 1,
                    self.ball[0] + 1
                )

                reward += 0.25

            else:

                reward -= 0.15

        # ---------------- PASS ----------------

        elif action == 4:

            distance = np.linalg.norm(
                self.player -
                self.ball
            )

            if distance < 2.0:

                self.ball[0] = min(
                    GRID_W - 1,
                    self.ball[0] + 2
                )

                reward += 0.35

            else:

                reward -= 0.20

        # ---------------- SHOOT ----------------

        elif action == 5:

            ball_distance = np.linalg.norm(
                self.player -
                self.ball
            )

            goal_distance = np.linalg.norm(
                self.ball -
                self.goal
            )

            if (
                ball_distance < 2.0
                and
                goal_distance < 5.0
            ):

                reward += 1.5

                # احتمال تسجيل الهدف

                chance = max(
                    0.05,
                    1.0 -
                    goal_distance / 7.0
                )

                if random.random() < chance:

                    reward += 10.0

                    return (
                        self.get_state(),
                        reward,
                        True,
                        "GOAL"
                    )

            else:

                reward -= 0.25

        # ---------------- DEFEND ----------------

        elif action == 7:

            distance = np.linalg.norm(
                self.player -
                self.opponent
            )

            if distance < 2.0:

                reward += 0.4

        # ---------------- OPPONENT ----------------

        if random.random() < 0.45:

            movement = random.choice(
                [-1, 0, 1]
            )

            self.opponent[1] += movement

            self.opponent[1] = np.clip(
                self.opponent[1],
                0,
                GRID_H - 1
            )

        # ---------------- BALL REWARD ----------------

        distance_to_ball = np.linalg.norm(
            self.player -
            self.ball
        )

        reward += max(
            0,
            0.10 -
            distance_to_ball * 0.01
        )

        done = (
            self.steps >=
            MAX_STEPS
        )

        return (
            self.get_state(),
            reward,
            done,
            None
        )


# ============================================================
#                    OPENCV DASHBOARD
# ============================================================

class Dashboard:

    def __init__(self):

        self.width = 1100
        self.height = 650

    def draw(
        self,
        env,
        probabilities,
        reward,
        loss,
        episode
    ):

        screen = np.zeros(
            (
                self.height,
                self.width,
                3
            ),
            dtype=np.uint8
        )

        # ---------------- FIELD ----------------

        left = 30
        top = 40
        cell = 55

        field_width = (
            GRID_W * cell
        )

        field_height = (
            GRID_H * cell
        )

        # Grid

        for x in range(
            GRID_W + 1
        ):

            cv2.line(
                screen,
                (
                    left + x * cell,
                    top
                ),
                (
                    left + x * cell,
                    top + field_height
                ),
                (60, 60, 60),
                1
            )

        for y in range(
            GRID_H + 1
        ):

            cv2.line(
                screen,
                (
                    left,
                    top + y * cell
                ),
                (
                    left + field_width,
                    top + y * cell
                ),
                (60, 60, 60),
                1
            )

        # ---------------- OBJECTS ----------------

        def position(p):

            return (
                left +
                int(
                    p[0] * cell +
                    cell / 2
                ),

                top +
                int(
                    p[1] * cell +
                    cell / 2
                )
            )

        # Player

        cv2.circle(
            screen,
            position(env.player),
            16,
            (255, 180, 0),
            -1
        )

        # Ball

        cv2.circle(
            screen,
            position(env.ball),
            10,
            (255, 255, 255),
            -1
        )

        # Opponent

        cv2.circle(
            screen,
            position(env.opponent),
            16,
            (0, 0, 255),
            -1
        )

        # Goal

        cv2.rectangle(
            screen,

            (
                left +
                int(
                    env.goal[0] *
                    cell
                ),

                top +
                int(
                    (env.goal[1] - 1) *
                    cell
                )
            ),

            (
                left +
                int(
                    env.goal[0] *
                    cell
                ) + cell,

                top +
                int(
                    (env.goal[1] + 1) *
                    cell
                )
            ),

            (0, 255, 0),

            2
        )

        # ---------------- PANEL ----------------

        panel_x = 800

        cv2.putText(
            screen,
            "SPIDER AI FC",
            (
                panel_x,
                50
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )

        cv2.putText(
            screen,
            "AI DECISION ENGINE",
            (
                panel_x,
                80
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (150, 150, 150),
            1
        )

        # ---------------- PROBABILITIES ----------------

        order = np.argsort(
            probabilities
        )[::-1]

        for i, index in enumerate(
            order
        ):

            value = float(
                probabilities[index]
            )

            y = 130 + i * 45

            cv2.putText(
                screen,

                f"{ACTIONS[index]:8s} "
                f"{value * 100:5.1f}%",

                (
                    panel_x,
                    y
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.65,

                (230, 230, 230),

                2
            )

        # ---------------- INFO ----------------

        y = 510

        cv2.putText(
            screen,
            f"Episode: {episode}",
            (
                panel_x,
                y
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            screen,
            f"Reward: {reward:.3f}",
            (
                panel_x,
                y + 35
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            screen,
            f"Loss: {loss:.4f}",
            (
                panel_x,
                y + 70
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        return screen


# ============================================================
#                      MAIN TRAINING
# ============================================================

def main():

    print()
    print(
        "======================================"
    )

    print(
        "        SPIDER AI FC"
    )

    print(
        "     TRAINING EDITION"
    )

    print(
        "======================================"
    )

    print()

    env = SoccerEnvironment()

    brain = SpiderBrain()

    safety = SafetySystem()

    controller = SafeController()

    dashboard = Dashboard()

    # محاولة تحميل النموذج السابق

    brain.load()

    episodes = 1000

    for episode in range(
        1,
        episodes + 1
    ):

        state = env.reset()

        total_reward = 0

        last_loss = 0

        for step in range(
            MAX_STEPS
        ):

            # AI يختار القرار

            action = brain.choose_action(
                state,
                training=True
            )

            # احتمالات القرارات

            probabilities = (
                brain.get_probabilities(
                    state
                )
            )

            # Safety Layer

            action = safety.validate(
                action,
                state,
                probabilities
            )

            # تنفيذ تجريبي

            controller.send(
                action
            )

            # البيئة

            (
                next_state,
                reward,
                done,
                event
            ) = env.step(
                action
            )

            # Memory

            brain.remember(
                state,
                action,
                reward,
                next_state,
                done
            )

            # Learning

            last_loss = brain.learn()

            state = next_state

            total_reward += reward

            # Dashboard

            probabilities = (
                brain.get_probabilities(
                    state
                )
            )

            frame = dashboard.draw(
                env,
                probabilities,
                reward,
                last_loss,
                episode
            )

            cv2.imshow(
                "SPIDER AI FC",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            # Q أو ESC = خروج

            if key in (
                ord("q"),
                27
            ):

                brain.save()

                cv2.destroyAllWindows()

                print(
                    "SPIDER AI stopped."
                )

                return

            if done:

                break

        # Target Network

        if episode % TARGET_UPDATE == 0:

            brain.update_target()

        # حفظ النموذج

        if episode % 25 == 0:

            brain.save()

        print(
            f"Episode {episode:04d} | "
            f"Reward {total_reward:8.2f} | "
            f"Epsilon {brain.epsilon:.3f} | "
            f"Memory {len(brain.memory)}"
        )

    brain.save()

    cv2.destroyAllWindows()


# ============================================================
#                       START
# ============================================================

if __name__ == "__main__":

    main()
