import pygame
import sys
import time
import socket
import struct
import threading
import csv
from datetime import datetime

# ================= 脑电记录配置 =================
EEG_HOST = "127.0.0.1"
EEG_PORT = 8712
NUM_CHANNELS = 21
BYTES_PER_SAMPLE = (NUM_CHANNELS + 1) * 4
CHANNEL_NAMES = [
    "T7", "T8", "TP7", "TP8", "P7", "P5", "P3", "Pz", "P4", "P6",
    "P8", "PO7", "PO5", "PO3", "POz", "PO4", "PO6", "PO8", "O1", "Oz", "O2"
]

# 采样率配置
TARGET_SAMPLING_RATE = 250.0

# 按键与触发码映射
KEY_TRIGGER_MAP = {
    'none': 0, 'up': 1, 'down': 2, 'left': 3, 'right': 4, 'space': 5
}


class SyncState:
    def __init__(self):
        self.running = True
        self.kb_trigger = 0
        self.level_id = 0
        self.game_state = "START"
        self.lock = threading.Lock()


state = SyncState()


# ================= 脑电后台记录线程 =================
class EEGRecorder(threading.Thread):
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self.daemon = True
        self.sample_count = 0  # 记录收到的总样本数

    def run(self):
        print(f"尝试连接脑电服务器 {EEG_HOST}:{EEG_PORT}...")
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect((EEG_HOST, EEG_PORT))
            print(f"连接成功！预期采样率: {TARGET_SAMPLING_RATE}Hz。数据存至: {self.filename}")
        except Exception as e:
            print(f"连接失败: {e}")
            state.running = False
            return

        # 使用 perf_counter 代替 time()，精度更高且不受系统对时影响
        self.start_record_time = time.perf_counter()

        # 采样率监控相关的变量
        last_check_time = self.start_record_time
        last_check_count = 0

        with open(self.filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            # 同时记录 样本序号(index)、理想时间(ideal_time) 和 网络接收时间(recv_time)
            header = ['sample_index', 'ideal_time', 'recv_time'] + CHANNEL_NAMES + ['raw_trigger', 'kb_trigger',
                                                                                    'level_id', 'state_code']
            writer.writerow(header)

            buffer = b''
            while state.running:
                try:
                    # 设定短暂超时，防止阻塞导致无法退出线程
                    sock.settimeout(1.0)
                    data = sock.recv(4096)
                    if not data: break
                    buffer += data

                    # 记录这一批次数据“到达”操作系统的时间
                    batch_recv_time = time.perf_counter() - self.start_record_time

                    while len(buffer) >= BYTES_PER_SAMPLE:
                        sample_bytes = buffer[:BYTES_PER_SAMPLE]
                        buffer = buffer[BYTES_PER_SAMPLE:]
                        values = struct.unpack('<22f', sample_bytes)

                        channels = list(values[:NUM_CHANNELS])
                        raw_trigger = values[NUM_CHANNELS]

                        # 【核心防错：策略1】基于采样率推算的“绝对无抖动”时间
                        ideal_time = self.sample_count / TARGET_SAMPLING_RATE

                        with state.lock:
                            kbt = state.kb_trigger
                            lvl = state.level_id
                            st_name = state.game_state

                        st_code = {"START": 0, "LEVEL_SELECT": 1, "PLAYING": 2, "REST": 3, "END": 4}.get(st_name, -1)

                        # 写入数据
                        writer.writerow([
                                            self.sample_count,
                                            round(ideal_time, 4),  # 喂给深度学习/MNE的标准时间
                                            round(batch_recv_time, 4)  # 仅供debug网络延迟用的时间
                                        ] + channels + [raw_trigger, kbt, lvl, st_code])

                        self.sample_count += 1

                    # 每隔 5 秒监控一次真实的采样率
                    current_time = time.perf_counter()
                    time_elapsed = current_time - last_check_time
                    if time_elapsed >= 5.0:
                        samples_in_interval = self.sample_count - last_check_count
                        actual_fps = samples_in_interval / time_elapsed

                        # 如果实际采样率与目标偏离超过 5%，报警！
                        if abs(actual_fps - TARGET_SAMPLING_RATE) > (TARGET_SAMPLING_RATE * 0.05):
                            print(
                                f"\n[警告] 网络掉包或采样率异常！当前实际接收速率: {actual_fps:.1f} Hz (预期: {TARGET_SAMPLING_RATE} Hz)")

                        last_check_time = current_time
                        last_check_count = self.sample_count

                except socket.timeout:
                    continue  # 超时没关系，继续等下一次
                except Exception as e:
                    print(f"\n[X] 记录数据时发生错误: {e}")
                    break

        sock.close()

        # 实验结束后，打印全局采样率报告
        total_time = time.perf_counter() - self.start_record_time
        print(f"\n采集结束。总耗时: {total_time:.2f} 秒, 总样本数: {self.sample_count}")
        print(f"全局平均有效采样率: {self.sample_count / total_time:.2f} Hz")



# MazeMiniGame
pygame.init()
WIDTH, HEIGHT = 600, 600
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Maze EEG Sync Collector")
CLOCK = pygame.time.Clock()

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 0, 0)
BLUE = (0, 0, 200)
GREEN = (0, 200, 0)
ORANGE = (255, 140, 0)
YELLOW = (255, 255, 0)
GRAY = (180, 180, 180)

player_radius = 8
player_speed = 4
checkpoint_hold_time = 2
font = pygame.font.SysFont(None, 32)
number_font = pygame.font.SysFont(None, 24)

levels = [
    {"corridors": [pygame.Rect(50, 50, 40, 500), pygame.Rect(50, 50, 500, 40), pygame.Rect(510, 50, 40, 500),
                   pygame.Rect(50, 510, 500, 40)],
     "checkpoints": [pygame.Rect(50, 50, 40, 40), pygame.Rect(510, 50, 40, 40), pygame.Rect(510, 510, 40, 40),
                     pygame.Rect(50, 510, 40, 40)],
     "player_start": [70, 70]},
    {"corridors": [pygame.Rect(100, 50, 40, 200), pygame.Rect(100, 250, 200, 40), pygame.Rect(300, 50, 40, 240),
                   pygame.Rect(300, 250, 200, 40), pygame.Rect(500, 50, 40, 240)],
     "checkpoints": [pygame.Rect(100, 50, 40, 40), pygame.Rect(300, 50, 40, 40), pygame.Rect(500, 50, 40, 40)],
     "player_start": [120, 70]},
    {"corridors": [pygame.Rect(50, 50, 40, 500), pygame.Rect(50, 250, 500, 40), pygame.Rect(510, 50, 40, 240)],
     "checkpoints": [pygame.Rect(50, 50, 40, 40), pygame.Rect(510, 50, 40, 40), pygame.Rect(510, 250, 40, 40),
                     pygame.Rect(50, 290, 40, 40)],
     "player_start": [70, 70]},
    {"corridors": [pygame.Rect(50, 50, 40, 200), pygame.Rect(50, 250, 200, 40), pygame.Rect(200, 250, 40, 200),
                   pygame.Rect(200, 450, 200, 40), pygame.Rect(400, 450, 40, 100)],
     "checkpoints": [pygame.Rect(50, 50, 40, 40), pygame.Rect(200, 250, 40, 40), pygame.Rect(400, 450, 40, 40)],
     "player_start": [70, 70]},
    {"corridors": [pygame.Rect(50, 50, 40, 500), pygame.Rect(50, 50, 500, 40), pygame.Rect(510, 50, 40, 500),
                   pygame.Rect(50, 510, 500, 40), pygame.Rect(200, 50, 40, 300), pygame.Rect(350, 250, 40, 260)],
     "checkpoints": [pygame.Rect(50, 50, 40, 40), pygame.Rect(200, 200, 40, 40), pygame.Rect(350, 450, 40, 40),
                     pygame.Rect(510, 510, 40, 40)],
     "player_start": [70, 70]},
    {  # Level 6 - 蛇形回廊 (Snake Winding)
        "corridors": [
            pygame.Rect(50, 50, 500, 40),  # 第一横行
            pygame.Rect(510, 50, 40, 150),  # 右侧连接
            pygame.Rect(100, 160, 450, 40),  # 第二横行
            pygame.Rect(100, 160, 40, 150),  # 左侧连接
            pygame.Rect(100, 310, 450, 40),  # 第三横行
            pygame.Rect(510, 310, 40, 150),  # 右侧连接
            pygame.Rect(50, 460, 500, 40)  # 第四横行
        ],
        "checkpoints": [
            pygame.Rect(50, 50, 40, 40),  # 起点
            pygame.Rect(510, 50, 40, 40),  # 第1拐点
            pygame.Rect(100, 160, 40, 40),  # 第2拐点
            pygame.Rect(510, 310, 40, 40),  # 第3拐点
            pygame.Rect(50, 460, 40, 40)  # 终点
        ],
        "player_start": [70, 70]
    }
]


def draw_maze(corridors, checkpoints, current_checkpoint, holding_space_start):
    SCREEN.fill(WHITE)
    for corridor in corridors:
        pygame.draw.rect(SCREEN, BLACK, corridor)
    for i, cp in enumerate(checkpoints):
        if i < current_checkpoint:
            color = GREEN
        elif i == current_checkpoint:
            if holding_space_start:
                progress = min((time.time() - holding_space_start) / checkpoint_hold_time, 1)
                # 你的原版颜色插值算法
                color = (int(ORANGE[0] * progress + BLUE[0] * (1 - progress)),
                         int(ORANGE[1] * progress + BLUE[1] * (1 - progress)),
                         int(ORANGE[2] * progress + BLUE[2] * (1 - progress)))
            else:
                color = BLUE
        else:
            color = BLUE
        pygame.draw.rect(SCREEN, color, cp)
        num_text = number_font.render(str(i + 1), True, WHITE)
        text_rect = num_text.get_rect(center=cp.center)
        SCREEN.blit(num_text, text_rect)


def inside_corridors(pos, corridors):
    # 计算玩家的外接矩形
    player_rect = pygame.Rect(
        pos[0] - player_radius,
        pos[1] - player_radius,
        player_radius * 2,
        player_radius * 2
    )

    # 检查玩家矩形的四个顶点
    corners = [
        player_rect.topleft,
        player_rect.topright,
        player_rect.bottomleft,
        player_rect.bottomright
    ]

    # 逻辑：对于每一个角点，它必须至少位于其中一个走廊矩形内
    for corner in corners:
        point_in_any_corridor = False
        for corridor in corridors:
            if corridor.collidepoint(corner):
                point_in_any_corridor = True
                break
        # 如果有一个角点不在任何走廊里，说明出界了
        if not point_in_any_corridor:
            return False

    return True


def at_checkpoint(player_pos, checkpoints, current_checkpoint):
    if current_checkpoint >= len(checkpoints):
        return False
    player_rect = pygame.Rect(player_pos[0] - player_radius, player_pos[1] - player_radius, player_radius * 2,
                              player_radius * 2)
    return player_rect.colliderect(checkpoints[current_checkpoint])


def draw_start_screen(selected_option=None):
    SCREEN.fill(WHITE)
    title_text = font.render("Maze Checkpoint Game", True, BLACK)
    SCREEN.blit(title_text, (WIDTH // 2 - title_text.get_width() // 2, 100))
    options = ["Start from Beginning", "Select Level", "Exit"]
    for i, option in enumerate(options):
        color = ORANGE if selected_option == i else GRAY
        size = 40 if selected_option == i else 32
        option_font = pygame.font.SysFont(None, size, bold=(selected_option == i))
        option_text = option_font.render(option, True, color)
        SCREEN.blit(option_text, (WIDTH // 2 - option_text.get_width() // 2, 200 + i * 70))
    return options


def draw_level_selection(selected_level=None):
    SCREEN.fill(WHITE)
    title_text = font.render("Select Level", True, BLACK)
    SCREEN.blit(title_text, (WIDTH // 2 - title_text.get_width() // 2, 100))
    for i in range(len(levels)):
        color = ORANGE if selected_level == i else GRAY
        level_font = pygame.font.SysFont(None, 36 if selected_level == i else 32, bold=(selected_level == i))
        level_text = level_font.render(f"Level {i + 1}", True, color)
        SCREEN.blit(level_text, (WIDTH // 2 - level_text.get_width() // 2, 200 + i * 60))



def main():

    csv_fn = f"EEG_Maze_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    recorder = EEGRecorder(csv_fn)
    recorder.start()

    game_state = "start"
    selected_option = 0
    selected_level = 0
    level_index = 0
    player_pos = [0, 0]
    current_checkpoint = 0
    holding_space_start = None
    rest_start_time = None
    REST_TIME = 5

    while state.running:
        dt = CLOCK.tick(60)
        keys = pygame.key.get_pressed()

        curr_kb = KEY_TRIGGER_MAP['none']
        if keys[pygame.K_UP]:
            curr_kb = KEY_TRIGGER_MAP['up']
        elif keys[pygame.K_DOWN]:
            curr_kb = KEY_TRIGGER_MAP['down']
        elif keys[pygame.K_LEFT]:
            curr_kb = KEY_TRIGGER_MAP['left']
        elif keys[pygame.K_RIGHT]:
            curr_kb = KEY_TRIGGER_MAP['right']
        elif keys[pygame.K_SPACE]:
            curr_kb = KEY_TRIGGER_MAP['space']

        with state.lock:
            state.kb_trigger = curr_kb
            state.level_id = level_index + 1
            state.game_state = game_state.upper()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                state.running = False
            if event.type == pygame.KEYDOWN:
                if game_state == "start":
                    if event.key == pygame.K_UP: selected_option = (selected_option - 1) % 3
                    if event.key == pygame.K_DOWN: selected_option = (selected_option + 1) % 3
                    if event.key == pygame.K_RETURN:
                        if selected_option == 0:
                            level_index = 0
                            player_pos = levels[level_index]["player_start"][:]
                            current_checkpoint = 0
                            game_state = "playing"
                        elif selected_option == 1:
                            game_state = "level_select"
                        elif selected_option == 2:
                            state.running = False
                elif game_state == "level_select":
                    if event.key == pygame.K_UP: selected_level = (selected_level - 1) % len(levels)
                    if event.key == pygame.K_DOWN: selected_level = (selected_level + 1) % len(levels)
                    if event.key == pygame.K_RETURN:
                        level_index = selected_level
                        player_pos = levels[level_index]["player_start"][:]
                        current_checkpoint = 0
                        game_state = "playing"
                elif game_state == "end":
                    if event.key == pygame.K_RETURN: game_state = "start"

        if game_state == "playing":
            # --- 尝试水平移动 ---
            new_pos_x = [player_pos[0], player_pos[1]]
            if keys[pygame.K_LEFT]: new_pos_x[0] -= player_speed
            if keys[pygame.K_RIGHT]: new_pos_x[0] += player_speed

            if inside_corridors(new_pos_x, levels[level_index]["corridors"]):
                player_pos[0] = new_pos_x[0]

            # --- 尝试垂直移动 ---
            new_pos_y = [player_pos[0], player_pos[1]]
            if keys[pygame.K_UP]: new_pos_y[1] -= player_speed
            if keys[pygame.K_DOWN]: new_pos_y[1] += player_speed

            if inside_corridors(new_pos_y, levels[level_index]["corridors"]):
                player_pos[1] = new_pos_y[1]

            if at_checkpoint(player_pos, levels[level_index]["checkpoints"], current_checkpoint):
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

            draw_maze(levels[level_index]["corridors"], levels[level_index]["checkpoints"], current_checkpoint,
                      holding_space_start)
            pygame.draw.circle(SCREEN, RED, player_pos, player_radius)

            if current_checkpoint >= len(levels[level_index]["checkpoints"]):
                rest_start_time = time.time()
                game_state = "rest"

        elif game_state == "rest":
            SCREEN.fill(WHITE)
            elapsed = time.time() - rest_start_time
            if elapsed < REST_TIME:
                txt = font.render(f"Level {level_index + 1} completed! Next in {int(REST_TIME - elapsed)}...", True,
                                  BLUE)
                SCREEN.blit(txt, (50, HEIGHT // 2 - 20))
            else:
                if level_index < len(levels) - 1:
                    level_index += 1
                    player_pos = levels[level_index]["player_start"][:]
                    current_checkpoint = 0
                    game_state = "playing"
                else:
                    game_state = "end"

        elif game_state == "start":
            draw_start_screen(selected_option)

        elif game_state == "level_select":
            draw_level_selection(selected_level)

        elif game_state == "end":
            SCREEN.fill(WHITE)
            end_text = font.render("All levels completed! Press Enter to start.", True, YELLOW)
            SCREEN.blit(end_text, (50, HEIGHT // 2 - 20))

        pygame.display.flip()

    pygame.quit()
    print(f"采集完成，数据已安全保存至 {csv_fn}")


if __name__ == "__main__":
    main()