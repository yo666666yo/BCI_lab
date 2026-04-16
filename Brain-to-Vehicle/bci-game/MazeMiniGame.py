import pygame
import sys
import time

# --- Game Setup ---
pygame.init()
WIDTH, HEIGHT = 600, 600
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Maze Checkpoint Game")
CLOCK = pygame.time.Clock()

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 0, 0)
BLUE = (0, 0, 200)
GREEN = (0, 200, 0)
ORANGE = (255, 140, 0)
YELLOW = (255, 255, 0)
GRAY = (180, 180, 180)

# Player
player_radius = 8

# --- Speed intensity levels (matches the BCI 3-class intensity decoder) ---
# Index 0/1/2 correspond to Slow / Medium / Fast motor-imagery intensity.
# The medium speed keeps the original game feel (4 px/frame).
SPEED_LEVELS = [2, 4, 7]
SPEED_LABELS = ["Slow", "Medium", "Fast"]
SPEED_COLORS = [(0, 120, 255), (200, 0, 0), (160, 0, 200)]
DEFAULT_SPEED_IDX = 1
current_speed_idx = DEFAULT_SPEED_IDX

# Checkpoint hold time
checkpoint_hold_time = 2  # seconds

# Fonts
font = pygame.font.SysFont(None, 32)
number_font = pygame.font.SysFont(None, 24)

# --- Levels ---
levels = [
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
    },
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

# --- Helper Functions ---
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
                color = (int(ORANGE[0]*progress + BLUE[0]*(1-progress)),
                         int(ORANGE[1]*progress + BLUE[1]*(1-progress)),
                         int(ORANGE[2]*progress + BLUE[2]*(1-progress)))
            else:
                color = BLUE
        else:
            color = BLUE
        pygame.draw.rect(SCREEN, color, cp)
        num_text = number_font.render(str(i+1), True, WHITE)
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
    player_rect = pygame.Rect(player_pos[0]-player_radius, player_pos[1]-player_radius, player_radius*2, player_radius*2)
    return player_rect.colliderect(checkpoints[current_checkpoint])

def draw_start_screen(selected_option=None):
    SCREEN.fill(WHITE)
    title_text = font.render("Maze Checkpoint Game", True, BLACK)
    SCREEN.blit(title_text, (WIDTH//2 - title_text.get_width()//2, 100))
    
    options = ["Start from Beginning", "Select Level", "Exit"]
    
    for i, option in enumerate(options):
        if selected_option == i:
            color = ORANGE
            size = 40
        else:
            color = GRAY
            size = 32
        
        option_font = pygame.font.SysFont(None, size, bold=(selected_option==i))
        option_text = option_font.render(option, True, color)
        SCREEN.blit(option_text, (WIDTH//2 - option_text.get_width()//2, 200 + i*70))
    
    # Display instruction for current tab
    if selected_option is not None:
        instruction_text = font.render(f"Press Enter to {options[selected_option]}", True, BLUE)
        SCREEN.blit(instruction_text, (WIDTH//2 - instruction_text.get_width()//2, 430))

    # Controls hint: directions + 3-class speed intensity
    hint_lines = [
        "Controls: Arrow keys = Move   Space = Stop/Hold at checkpoint",
        "Speed intensity: 1 = Slow    2 = Medium    3 = Fast",
    ]
    for i, line in enumerate(hint_lines):
        hint_text = number_font.render(line, True, BLACK)
        SCREEN.blit(hint_text, (WIDTH//2 - hint_text.get_width()//2, 490 + i * 22))

    return options

def draw_level_selection(selected_level=None):
    SCREEN.fill(WHITE)
    title_text = font.render("Select Level", True, BLACK)
    SCREEN.blit(title_text, (WIDTH//2 - title_text.get_width()//2, 100))
    for i in range(len(levels)):
        color = ORANGE if selected_level == i else GRAY
        level_font = pygame.font.SysFont(None, 36 if selected_level==i else 32, bold=(selected_level==i))
        level_text = level_font.render(f"Level {i+1}", True, color)
        SCREEN.blit(level_text, (WIDTH//2 - level_text.get_width()//2, 200 + i*60))
    return [f"Level {i+1}" for i in range(len(levels))]

# --- Game States ---
STATE_START = "start"
STATE_LEVEL_SELECT = "level_select"
STATE_PLAYING = "playing"
STATE_REST = "rest"
STATE_END = "end"

game_state = STATE_START
selected_option = 0
selected_level = 0
level_index = 0
player_pos = [0,0]
current_checkpoint = 0
holding_space_start = None
rest_start_time = None
REST_TIME = 5

# --- Main Loop ---
running = True
while running:
    dt = CLOCK.tick(60)
    keys = pygame.key.get_pressed()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if game_state == STATE_START:
                if event.key == pygame.K_UP:
                    selected_option = (selected_option -1) % 3
                if event.key == pygame.K_DOWN:
                    selected_option = (selected_option +1) % 3
                if event.key == pygame.K_RETURN:
                    if selected_option == 0:
                        level_index = 0
                        player_pos = levels[level_index]["player_start"][:]
                        current_checkpoint = 0
                        holding_space_start = None
                        current_speed_idx = DEFAULT_SPEED_IDX
                        game_state = STATE_PLAYING
                    elif selected_option == 1:
                        game_state = STATE_LEVEL_SELECT
                        selected_level = 0
                    elif selected_option == 2:
                        running = False
            elif game_state == STATE_LEVEL_SELECT:
                if event.key == pygame.K_UP:
                    selected_level = (selected_level -1) % len(levels)
                if event.key == pygame.K_DOWN:
                    selected_level = (selected_level +1) % len(levels)
                if event.key == pygame.K_RETURN:
                    level_index = selected_level
                    player_pos = levels[level_index]["player_start"][:]
                    current_checkpoint = 0
                    holding_space_start = None
                    current_speed_idx = DEFAULT_SPEED_IDX
                    game_state = STATE_PLAYING
            elif game_state == STATE_PLAYING:
                # Speed intensity switching — corresponds to the 3-class
                # intensity decoder head in the BCI pipeline. Pressing the
                # number key is meant to accompany the subject's motor-imagery
                # of "slow / medium / fast" movement.
                if event.key == pygame.K_1:
                    current_speed_idx = 0
                elif event.key == pygame.K_2:
                    current_speed_idx = 1
                elif event.key == pygame.K_3:
                    current_speed_idx = 2
            elif game_state == STATE_END:
                if event.key == pygame.K_RETURN:
                    game_state = STATE_START
                    selected_option = 0

    # --- Game Logic ---
    if game_state == STATE_PLAYING:
        speed = SPEED_LEVELS[current_speed_idx]
        new_pos = player_pos.copy()
        if keys[pygame.K_UP]: new_pos[1] -= speed
        if keys[pygame.K_DOWN]: new_pos[1] += speed
        if keys[pygame.K_LEFT]: new_pos[0] -= speed
        if keys[pygame.K_RIGHT]: new_pos[0] += speed
        if inside_corridors(new_pos, levels[level_index]["corridors"]):
            player_pos = new_pos

        if at_checkpoint(player_pos, levels[level_index]["checkpoints"], current_checkpoint):
            if keys[pygame.K_SPACE]:
                if holding_space_start is None:
                    holding_space_start = time.time()
                else:
                    if time.time() - holding_space_start >= checkpoint_hold_time:
                        current_checkpoint += 1
                        holding_space_start = None
            else:
                holding_space_start = None
        else:
            holding_space_start = None

        draw_maze(levels[level_index]["corridors"], levels[level_index]["checkpoints"], current_checkpoint, holding_space_start)
        pygame.draw.circle(SCREEN, SPEED_COLORS[current_speed_idx], player_pos, player_radius)

        # --- Speed intensity HUD ---
        speed_label = SPEED_LABELS[current_speed_idx]
        hud_text = number_font.render(
            f"Speed [1/2/3]: {speed_label}  ({SPEED_LEVELS[current_speed_idx]} px/f)",
            True, SPEED_COLORS[current_speed_idx]
        )
        SCREEN.blit(hud_text, (10, HEIGHT - 26))
        # Small tick bar to make the active intensity level visually obvious
        for i, _ in enumerate(SPEED_LEVELS):
            tick_color = SPEED_COLORS[i] if i == current_speed_idx else GRAY
            tick_rect = pygame.Rect(WIDTH - 110 + i * 32, HEIGHT - 26, 26, 16)
            pygame.draw.rect(SCREEN, tick_color, tick_rect)
            tick_num = number_font.render(str(i + 1), True, WHITE)
            SCREEN.blit(tick_num, tick_num.get_rect(center=tick_rect.center))

        if current_checkpoint >= len(levels[level_index]["checkpoints"]):
            rest_start_time = time.time()
            game_state = STATE_REST

    elif game_state == STATE_REST:
        SCREEN.fill(WHITE)
        rest_elapsed = time.time() - rest_start_time
        if rest_elapsed < REST_TIME:
            rest_text = font.render(f"Level {level_index+1} completed! Next in {int(REST_TIME-rest_elapsed)}...", True, BLUE)
            SCREEN.blit(rest_text, (50, HEIGHT//2 - 20))
        else:
            if level_index < len(levels)-1:
                level_index += 1
                player_pos = levels[level_index]["player_start"][:]
                current_checkpoint = 0
                holding_space_start = None
                game_state = STATE_PLAYING
            else:
                game_state = STATE_END

    elif game_state == STATE_START:
        draw_start_screen(selected_option)

    elif game_state == STATE_LEVEL_SELECT:
        draw_level_selection(selected_level)

    elif game_state == STATE_END:
        SCREEN.fill(WHITE)
        end_text = font.render("All levels completed! Press Enter to start.", True, YELLOW)
        SCREEN.blit(end_text, (50, HEIGHT//2 - 20))

    pygame.display.flip()

pygame.quit()
sys.exit()
