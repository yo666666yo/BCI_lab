import pygame
import time
import numpy as np
import os  # 用于设置窗口位置

# --- 颜色定义 (保持原版) ---
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 0, 0)
BLUE = (0, 0, 200)
GREEN = (0, 200, 0)
ORANGE = (255, 140, 0)
YELLOW = (255, 255, 0)
GRAY = (180, 180, 180)
DARK_GRAY = (50, 50, 50)


class MazeCalibrationGame:
    def __init__(self, data_buffer, input_window=500, training_queue=None):
        self.data_buffer = data_buffer
        self.input_window = input_window
        self.training_queue = training_queue # <--- 新增：接收一个队列

        # 收集到的数据容器
        self.collected_data = []
        self.collected_labels = []

        # 标签映射: 0:Stop, 1:Forward, 2:Backward, 3:Left, 4:Right
        self.key_map = {
            pygame.K_UP: 1,
            pygame.K_DOWN: 2,
            pygame.K_LEFT: 3,
            pygame.K_RIGHT: 4,
            pygame.K_SPACE: 0
        }

        self.last_collection_time = 0
        self.collection_cooldown = 0.5

        # --- 完全还原原版关卡数据 ---
        self.levels = [
                {   # Level 1
                    "corridors": [
                        pygame.Rect(50, 50, 40, 500),
                        pygame.Rect(50, 50, 500, 40),
                        pygame.Rect(510, 50, 40, 500),
                        pygame.Rect(50, 510, 500, 40)
                    ],
                    "checkpoints": [
                        pygame.Rect(50, 50, 40, 40),
                        pygame.Rect(510, 50, 40, 40),
                        pygame.Rect(510, 510, 40, 40),
                        pygame.Rect(50, 510, 40, 40)
                    ],
                    "player_start": [70, 70]
                },
                {   # Level 2
                    "corridors": [
                        pygame.Rect(100, 50, 40, 200),
                        pygame.Rect(100, 250, 200, 40),
                        pygame.Rect(300, 50, 40, 240),
                        pygame.Rect(300, 250, 200, 40),
                        pygame.Rect(500, 50, 40, 240)
                    ],
                    "checkpoints": [
                        pygame.Rect(100, 50, 40, 40),
                        pygame.Rect(300, 50, 40, 40),
                        pygame.Rect(500, 50, 40, 40)
                    ],
                    "player_start": [120, 70]
                },
                {   # Level 3
                    "corridors": [
                        pygame.Rect(50, 50, 40, 500),
                        pygame.Rect(50, 250, 500, 40),
                        pygame.Rect(510, 50, 40, 240)
                    ],
                    "checkpoints": [
                        pygame.Rect(50, 50, 40, 40),
                        pygame.Rect(510, 50, 40, 40),
                        pygame.Rect(510, 250, 40, 40),
                        pygame.Rect(50, 290, 40, 40)
                    ],
                    "player_start": [70, 70]
                },
                {   # Level 4
                    "corridors": [
                        pygame.Rect(50, 50, 40, 200),
                        pygame.Rect(50, 250, 200, 40),
                        pygame.Rect(200, 250, 40, 200),
                        pygame.Rect(200, 450, 200, 40),
                        pygame.Rect(400, 450, 40, 100)
                    ],
                    "checkpoints": [
                        pygame.Rect(50, 50, 40, 40),
                        pygame.Rect(200, 250, 40, 40),
                        pygame.Rect(400, 450, 40, 40)
                    ],
                    "player_start": [70, 70]
                },
                {   # Level 5
                    "corridors": [
                        pygame.Rect(50, 50, 40, 500),
                        pygame.Rect(50, 50, 500, 40),
                        pygame.Rect(510, 50, 40, 500),
                        pygame.Rect(50, 510, 500, 40),
                        pygame.Rect(200, 50, 40, 300),
                        pygame.Rect(350, 250, 40, 260)
                    ],
                    "checkpoints": [
                        pygame.Rect(50, 50, 40, 40),
                        pygame.Rect(200, 200, 40, 40),
                        pygame.Rect(350, 450, 40, 40),
                        pygame.Rect(510, 510, 40, 40)
                    ],
                    "player_start": [70, 70]
                }
            ]

    def run_game(self):
        # --- 强制窗口居中 ---
        os.environ['SDL_VIDEO_CENTERED'] = '1'

        pygame.init()
        WIDTH, HEIGHT = 600, 600
        SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("MazeminiGame")
        CLOCK = pygame.time.Clock()

        # 字体定义
        self.number_font = pygame.font.SysFont(None, 24)
        self.font = pygame.font.SysFont(None, 32)

        # 游戏参数
        player_radius = 8
        player_speed = 5  # 恢复原版速度
        checkpoint_hold_time = 1.5  # 恢复原版时间

        # 游戏状态
        level_index = 0
        current_level = self.levels[level_index]
        player_pos = list(current_level["player_start"])
        current_checkpoint = 0
        holding_space_start = None

        running = True

        while running:
            dt = CLOCK.tick(60)
            keys = pygame.key.get_pressed()

            # 事件处理
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                # 按键按下采集
                if event.type == pygame.KEYDOWN:
                    if event.key in self.key_map:
                        self.capture_eeg(self.key_map[event.key])

            # 持续按键移动 & 采集
            move_x, move_y = 0, 0
            current_time = time.time()
            active_label = -1  # <--- 【修改】设为 -1，因为 0 现在代表 Stop 标签

            if keys[pygame.K_LEFT]:
                move_x -= player_speed
                active_label = 3
            if keys[pygame.K_RIGHT]:
                move_x += player_speed
                active_label = 4
            if keys[pygame.K_UP]:
                move_y -= player_speed
                active_label = 1
            if keys[pygame.K_DOWN]:
                move_y += player_speed
                active_label = 2

            # 【新增】如果没有按方向键移动，但是按住了空格（比如在检查点停留），则判定为正在“想象停止”
            if move_x == 0 and move_y == 0 and keys[pygame.K_SPACE]:
                active_label = 0

            # 【修改】判断条件改为 != -1
            if active_label != -1 and (current_time - self.last_collection_time > self.collection_cooldown):
                self.capture_eeg(active_label)

            # 移动逻辑
            if move_x != 0:
                test_pos_x = [player_pos[0] + move_x, player_pos[1]]
                if self.inside_corridors(test_pos_x, current_level["corridors"], player_radius):
                    player_pos[0] += move_x

            if move_y != 0:
                test_pos_y = [player_pos[0], player_pos[1] + move_y]
                if self.inside_corridors(test_pos_y, current_level["corridors"], player_radius):
                    player_pos[1] += move_y

            # 检查点逻辑
            if self.at_checkpoint(player_pos, current_level["checkpoints"], current_checkpoint, player_radius):
                if keys[pygame.K_SPACE]:
                    if holding_space_start is None:
                        holding_space_start = time.time()
                    elif time.time() - holding_space_start >= checkpoint_hold_time:
                        current_checkpoint += 1
                        holding_space_start = None
                else:
                    holding_space_start = None
            else:
                holding_space_start = None

            # 关卡切换
            if current_checkpoint >= len(current_level["checkpoints"]):
                # 简单的关卡切换动画/逻辑
                level_index += 1
                if level_index >= len(self.levels):
                    print("所有关卡完成")
                    running = False
                else:
                    current_level = self.levels[level_index]
                    player_pos = list(current_level["player_start"])
                    current_checkpoint = 0
                    time.sleep(0.5)  # 稍微停顿一下

            # --- 绘图 ---
            self.draw_maze(SCREEN, current_level, current_checkpoint, holding_space_start, player_pos, player_radius)
            pygame.display.flip()

        pygame.quit()
        return self.collected_data, self.collected_labels

    def draw_maze(self, screen, level, current_checkpoint, holding_space_start, player_pos, player_radius):
        """完全还原原版的绘制逻辑"""
        screen.fill(WHITE)

        # 1. 绘制走廊
        for corridor in level["corridors"]:
            pygame.draw.rect(screen, BLACK, corridor)

        # 2. 绘制检查点
        for i, cp in enumerate(level["checkpoints"]):
            if i < current_checkpoint:
                color = GREEN
            elif i == current_checkpoint:
                color = BLUE
            else:
                color = GRAY

            pygame.draw.rect(screen, color, cp)

            # 绘制数字
            num_text = self.number_font.render(str(i + 1), True, WHITE)
            text_rect = num_text.get_rect(center=cp.center)
            screen.blit(num_text, text_rect)

        # 3. 绘制进度条
        if holding_space_start:
            # 原版进度条逻辑
            progress = min((time.time() - holding_space_start) / 1.5, 1)  # 1.5是hold_time
            bar_width = 40
            bar_height = 6
            bar_x = player_pos[0] - bar_width // 2
            bar_y = player_pos[1] - player_radius - 15

            pygame.draw.rect(screen, DARK_GRAY, (bar_x, bar_y, bar_width, bar_height))
            pygame.draw.rect(screen, ORANGE, (bar_x, bar_y, bar_width * progress, bar_height))
            pygame.draw.rect(screen, BLACK, (bar_x, bar_y, bar_width, bar_height), 1)

        # 4. 绘制玩家
        pygame.draw.circle(screen, RED, (int(player_pos[0]), int(player_pos[1])), player_radius)

        # 5. 绘制额外的采集信息
        info_text = self.font.render(f"Samples: {len(self.collected_data)}", True, BLUE)
        screen.blit(info_text, (10, 10))

    def capture_eeg(self, label):
        """截取当前的 EEG 数据并保存/发送"""
        if len(self.data_buffer) < self.input_window:
            return

        recent_data = list(self.data_buffer)[-self.input_window:]

        # 1. 本地保存一份
        self.collected_data.append(recent_data)
        self.collected_labels.append(label)

        # 2. 如果有队列，立即发送给后台训练
        if self.training_queue:
            # 数据需转置：(Time, Chan) -> (Chan, Time) 以匹配模型输入
            data_array = np.array(recent_data).T
            self.training_queue.put((data_array, label))
            print(f"[Game] 样本已发送至后台训练队列 (Label: {label})")

        self.last_collection_time = time.time()

    def inside_corridors(self, pos, corridors, radius):
        check_radius = radius * 0.8
        player_rect = pygame.Rect(pos[0] - check_radius, pos[1] - check_radius, check_radius * 2, check_radius * 2)
        for corridor in corridors:
            if player_rect.colliderect(corridor):
                return True
        return False

    def at_checkpoint(self, pos, checkpoints, idx, radius):
        if idx >= len(checkpoints): return False
        check_radius = radius * 1.5
        player_rect = pygame.Rect(pos[0] - check_radius, pos[1] - check_radius, check_radius * 2, check_radius * 2)
        return player_rect.colliderect(checkpoints[idx])