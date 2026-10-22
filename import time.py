import time
import cv2
import threading
import numpy as np
import rospy
import math
import sys
from enum import Enum, auto

# 用户自定义模块
from image_Tag_converter import ImgConverter, TagConverter

sys.path.append("/home/lemon/catkin_ws/src/aelos_smart_ros")
from leju import base_action


# ========================== 全局参数（可调） ==========================
COLOR_RANGE = {
    'green': [(38, 92, 91), (87, 255, 255)],
}

# 箱子抓取坐标阈值（图像像素）- 根据实际接触时的图像位置调整
X_TARGET_LO = 340
X_TARGET_HI = 370
Y_TARGET_LO = 300
Y_TARGET_HI = 380

# 横向死区：x 偏差小于此值时不再左右移动
X_DEAD_ZONE = 15

# 强制退出参数：连续前进3次后直接抱箱
MAX_FORWARD_BEFORE_PICK = 3

# 默认 Tag 导航阈值（作为后备）
TAG_X_THRESH = 0.15
TAG_Y_THRESH = 0.15
TAG_THETA_THRESH = 18.0

# Tag yaw 角度补偿：TagConverter 返回的 yaw 与机器人朝向存在 90° 固定偏差
TAG_YAW_OFFSET_DEG = 90.0
# 反向对正（返回路径）时的 theta 目标偏移（机器人掉头 180°）
TAG_REVERSE_THETA_OFFSET = 180.0
# 前进过程中"Tag 已大致对齐即中断"的宽松阈值倍数
TAG_INTERRUPT_THRESH_MULT = 2.5
TAG_INTERRUPT_THETA_MULT = 2.0

# 动作执行默认等待时间（秒）
ACTION_WAIT = 0.3
ACTION_TIMEOUT = 5.0

# 连续无Tag次数阈值，触发转向
NO_TAG_TURN_THRESH = 10

# ==================== 显示优化参数（减轻画面卡顿） ====================
DISPLAY_SCALE = 0.6            # 显示缩放比例：1.0=原尺寸，0.6=缩到60%（像素量减64%，大幅降CPU）
DISPLAY_DEBUG_INTERVAL = 3     # 调试帧绘制间隔：1=每帧都画（最卡），3=每3帧画一次

# ==================== 低算力优化开关（芯片算力不足时优先调这里） ====================
DISPLAY_ENABLED = True         # 显示线程总开关：比赛/无人值守设 False，省掉整帧 resize+绘制+imshow 的开销
DISPLAY_FPS = 5                # 显示线程目标帧率（只影响显示，不影响控制）；越低越省 CPU
MAIN_LOOP_HZ = 10              # 主循环频率：机器人动作步长 0.3-1s，可降到 5 再省近一半视觉计算
WORK_SCALE = 1.0               # 工作分辨率缩放（0.5=半分辨率，像素量降到 1/4）；<1 时需同步缩小所有像素阈值

# ==================== Tag 导航定位策略（仅 Tag，不用白线） ====================
# 单摄像头（胸部）只负责 Tag 识别，导航定位完全靠 Tag 码（ID 1~7）
#   检测到 Tag -> 听 Tag 指令（对正 / 跳转 / 前进中断）
#   无 Tag     -> 备用盲目前进 + 超时转向找 Tag
# ==============================================================

# ==================== Tag 独立导航配置（ID 1~7） ====================
TAG_NAV_CONFIG = {
    1: {'x_offset': 0.08, 'y_offset': 0.0, 'theta_offset': 2.0,
        'x_thresh': 0.12, 'y_thresh': 0.12, 'theta_thresh': 15.0},
    2: {'x_offset': 0.12, 'y_offset': 0.0, 'theta_offset': 0.0,
        'x_thresh': 0.10, 'y_thresh': 0.10, 'theta_thresh': 12.0},
    3: {'x_offset': 0.12, 'y_offset': 0.11, 'theta_offset': 0.0,
        'x_thresh': 0.08, 'y_thresh': 0.08, 'theta_thresh': 10.0},
    4: {'x_offset': 0.12, 'y_offset': -0.10, 'theta_offset': 0.0,
        'x_thresh': 0.08, 'y_thresh': 0.08, 'theta_thresh': 10.0},
    5: {'x_offset': 0.12, 'y_offset': 0.0, 'theta_offset': 0.0,
        'x_thresh': 0.08, 'y_thresh': 0.08, 'theta_thresh': 10.0},
    6: {'x_offset': 0.05, 'y_offset': -0.10, 'theta_offset': 0.0,
        'x_thresh': 0.08, 'y_thresh': 0.08, 'theta_thresh': 10.0},
    7: {'x_offset': 0.06, 'y_offset': 0.10, 'theta_offset': 0.0,
        'x_thresh': 0.08, 'y_thresh': 0.08, 'theta_thresh': 10.0},
}

# ========================== 预分配形态学核（避免每帧重复 np.ones 分配） ==========================
_KERNEL4 = np.ones((4, 4), np.uint8)   # find_box() 箱子去噪/膨胀核


# ========================== 任务状态机 ==========================
class Phase(Enum):
    """任务阶段（原 level 字符串的语义不变，仅类型化）。"""
    START_BOX = "start_box"          # 抓箱前
    END_BOX = "end_box"              # 已抓箱，待移动到 1 号 Tag
    START_MOVING = "start_moving"    # 搬运中
    REVERSE_MOVING = "reverse_moving"  # 返程（空载）


class Step(Enum):
    OUTBOUND = 1   # 去程（抱箱）
    INBOUND = 2    # 返程（空载）


class Flow(Enum):
    """Tag 分派 handler 的返回，显式化原代码里的 continue 语义。"""
    FALLTHROUGH = auto()   # 走到底部 rate.sleep()
    CONTINUE = auto()      # 等价原 continue（跳过 rate.sleep()）


class TaskState:
    """任务状态机，取代散落的 ID / step / level 三个全局变量。"""

    def __init__(self):
        self.tag_id = 0
        self.step = Step.OUTBOUND
        self.phase = Phase.START_BOX
        self.has_box = False   # 真实负载：以实际 grip/lift 与 put_down 为准

    def reset(self):
        self.tag_id = 0
        self.step = Step.OUTBOUND
        self.phase = Phase.START_BOX
        self.has_box = False


state = TaskState()


# ========================== 线程安全图像缓存 ==========================
class ImageBuffer:
    def __init__(self):
        self._img = None
        self._lock = threading.Lock()
        self._update_time = 0.0  # 最近一次更新时间戳

    def update(self, img):
        with self._lock:
            self._img = img
            self._update_time = time.time()

    def get(self):
        with self._lock:
            return None if self._img is None else self._img.copy()

    def get_age(self):
        """返回缓存图像的存在时间（秒），None=无图像"""
        with self._lock:
            if self._img is None:
                return None
            return time.time() - self._update_time

    def is_fresh(self, max_age=1.0):
        """检查缓存图像是否在指定秒数内更新过"""
        age = self.get_age()
        if age is None:
            return False
        return age < max_age

img_buffer = ImageBuffer()          # 胸部相机图像缓存（用于 Tag 识别 + 主画面显示）
tag_reader = None       # 全局 Tag 阅读器（在 main() 中初始化），供 move_forward() 检查 Tag 中断

# ========================== 动作统一执行器 ==========================
class ActionExecutor:
    _action_lock = threading.Lock()  # 动作串行锁：防止多条电机指令重叠执行

    ACTION_MAP = {
        "put_down": ("PutCubeDown", 0.5),
        "box_back": ("BoxBack1Run", 0.5),
        "box_back2": ("Back2Run", 0.5),
        "box_forward_small": ("前进70", 0.5),
        "box_forward_1s": ("BoxForward1s", 1.0),
        "box_forward_2s": ("BoxForward2s", 1.0),
        "box_forward_3s": ("BoxForward3s", 1.0),
        "box_left_move": ("左70", 1.0),
        "box_right_move": ("右70", 1.0),
        "box_left_turn": ("BoxLeftTurn1s", 1.0),
        "box_right_turn": ("BoxRightTurn1s", 1.0),
        "grip_close": ("GardCup004", 0.8),
        "lift_up": ("LiftCubeUp", 0.8),
        "forward_small": ("Forwalk01", 0.5),
        "forward_1s": ("FastForward1s", 0.5),
        "forward_2s": ("FastForward2s", 0.5),
        "forward_3s": ("FastForward3s", 0.5),
        "back_1s": ("Back1Run", 0.5),
        "back_2s": ("Back2Run", 0.5),
        "left_move": ("move_left", 0.5),
        "right_move": ("move_right", 0.5),
        "left_turn": ("LeftTurn1s", 0.5),
        "right_turn": ("RightTurn1s", 0.5),
    }

    @staticmethod
    def execute(action_name, repeat=1, wait=ACTION_WAIT, timeout=ACTION_TIMEOUT):
        if timeout is None:
            timeout = 5.0
        for i in range(repeat):
            act_key, _ = ActionExecutor.ACTION_MAP.get(action_name, (action_name, 0.5))
            rospy.loginfo(f"[动作] {act_key} (第{i+1}/{repeat}次)")
            done = threading.Event()

            def run():
                try:
                    base_action.action(act_key)
                except Exception as e:
                    rospy.logerr(f"动作执行异常 {act_key}: {e}")
                finally:
                    done.set()

            # 超时只用于告警，不抛弃动作：宁可阻塞等待，也不让下一条动作与它重叠
            with ActionExecutor._action_lock:
                thr = threading.Thread(target=run, daemon=True)
                thr.start()
                if not done.wait(timeout=timeout):
                    rospy.logwarn(f"动作超时 {act_key} > {timeout}s，等待其结束以防动作叠加")
                    done.wait()
                thr.join()
            time.sleep(wait)

# ========================== 动作快捷函数 ==========================
# 以下都是 ActionExecutor.execute() 的薄封装，用「简短函数名 + 步数参数」调用，
# 让上层控制代码能写成 go_fast(1)、BoxR_turn2(2) 这类紧凑写法。
# move1/turn1 系列与 move2/turn2 系列等价（1 表示小步、2 表示大步），下方统一做别名映射。
def Box_Down(n): ActionExecutor.execute("put_down", n)
def Box_Back(n): ActionExecutor.execute("box_back", n)
def Box_Back1(n): ActionExecutor.execute("box_back2", n)
def Back2(n): ActionExecutor.execute("back_2s", n)
def Back1(n): ActionExecutor.execute("back_1s", n)
def go_fast(n): ActionExecutor.execute("forward_small", n)
def go_fast1(n): ActionExecutor.execute("forward_1s", n)
def go_fast2(n): ActionExecutor.execute("forward_2s", n)
def go_fast3(n): ActionExecutor.execute("forward_3s", n)
def box_go(n): ActionExecutor.execute("box_forward_small", n)
def box_go1(n): ActionExecutor.execute("box_forward_1s", n)
def box_go2(n): ActionExecutor.execute("box_forward_2s", n)
def box_go3(n): ActionExecutor.execute("box_forward_3s", n)
def L_move2(n): ActionExecutor.execute("left_move", n)
def R_move2(n): ActionExecutor.execute("right_move", n)
def BoxL_move2(n): ActionExecutor.execute("box_left_move", n)
def BoxR_move2(n): ActionExecutor.execute("box_right_move", n)
def L_turn2(n): ActionExecutor.execute("left_turn", n)
def R_turn2(n): ActionExecutor.execute("right_turn", n)
def BoxL_turn2(n): ActionExecutor.execute("box_left_turn", n)
def BoxR_turn2(n): ActionExecutor.execute("box_right_turn", n)
L_move1 = L_move2
R_move1 = R_move2
BoxL_move1 = BoxL_move2
BoxR_move1 = BoxR_move2
L_turn1 = L_turn2
R_turn1 = R_turn2
BoxL_turn1 = BoxL_turn2
BoxR_turn1 = BoxR_turn2

# ========================== 视觉追踪与滤波 ==========================
class BoxTracker:
    def __init__(self, alpha=0.4):
        self.alpha = alpha
        self.filtered_x = None
        self.filtered_y = None

    def update(self, x, y):
        if self.filtered_x is None:
            self.filtered_x = x
            self.filtered_y = y
        else:
            self.filtered_x = self.alpha * x + (1 - self.alpha) * self.filtered_x
            self.filtered_y = self.alpha * y + (1 - self.alpha) * self.filtered_y
        return self.filtered_x, self.filtered_y

    def reset(self):
        self.filtered_x = None
        self.filtered_y = None

tracker = BoxTracker(alpha=0.4)


# ======================== 综合调试：Tag 可视化 ========================
_combined_debug_window_name = "[综合调试] Tag 状态"


class _DisplayState:
    """线程安全的显示状态，由主循环更新、独立显示线程读取渲染"""
    def __init__(self):
        self._lock = threading.Lock()
        self.tag_data = None          # 最新 Tag 数据 [id, x, y, yaw]
        self.nav_id = 0               # 当前导航 ID
        self.nav_step = 1             # 当前导航 step
        self.nav_level = "start_box"  # 当前导航 level
        self.running = True           # 是否运行

    def update_tag(self, tag_data):
        with self._lock:
            self.tag_data = tag_data

    def update_nav(self, nav_id=None, nav_step=None, nav_level=None):
        with self._lock:
            if nav_id is not None:    self.nav_id = nav_id
            if nav_step is not None:  self.nav_step = nav_step
            if nav_level is not None: self.nav_level = nav_level

    def snapshot(self):
        with self._lock:
            return (self.tag_data, self.nav_id, self.nav_step, self.nav_level)

    def stop(self):
        with self._lock:
            self.running = False


_combined_debug_state = _DisplayState()


def show_combined_debug(tag_data=None):
    """更新综合调试状态，实际渲染由独立显示线程完成"""
    if tag_data is not None:
        _combined_debug_state.update_tag(tag_data)
    _combined_debug_state.update_nav(nav_id=state.tag_id, nav_step=state.step.value,
                                     nav_level=state.phase.value)


def display_thread_func():
    """
    独立显示线程：单摄像头（胸部）综合调试画面。

    🖥️ 布局说明：
      主画面 = 胸部相机（Tag 识别 + 状态栏）

    🚀 性能优化：
    1. 显示缩放到 DISPLAY_SCALE（默认60%）→ 像素量仅为 36%
    2. 胸部相机为底图，Tag 信息叠加其上
    """
    win = _combined_debug_window_name
    cv2.namedWindow(win)
    rospy.loginfo(f"[显示] 单摄像头显示线程启动（缩放={DISPLAY_SCALE:.0%}）")

    _frame = 0  # 局部帧计数器

    while _combined_debug_state.running and not rospy.is_shutdown():
        # —— 获取胸部相机图像（主画面） ——
        chest_img = img_buffer.get()
        if chest_img is None:
            time.sleep(0.03)
            continue

        tag_data, nav_id, nav_step, nav_level = _combined_debug_state.snapshot()
        _frame += 1

        # —— 缩放胸部相机图像到 DISPLAY_SCALE ——
        if DISPLAY_SCALE < 1.0:
            new_size = (int(chest_img.shape[1] * DISPLAY_SCALE), int(chest_img.shape[0] * DISPLAY_SCALE))
            debug = cv2.resize(chest_img, new_size, interpolation=cv2.INTER_NEAREST)
        else:
            debug = chest_img.copy()
        h, w = debug.shape[:2]
        fs = max(0.35, DISPLAY_SCALE)  # 字体缩放基准

        # ── 1. 顶部状态栏 ──
        bar_h = int(28 * fs)
        cv2.rectangle(debug, (0, 0), (w, bar_h), (30, 30, 30), -1)
        label = f"ID={nav_id} | step={nav_step} | {nav_level}"
        cv2.putText(debug, label, (8, bar_h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, fs * 0.5, (200, 200, 200), 1)

        # ── 2. Tag 状态（右侧面板） ──
        tw = int(170 * fs)
        if tag_data is not None and len(tag_data) >= 4:
            tag_id = int(tag_data[0])
            tag_x = tag_data[1]
            tag_y = tag_data[2]
            tag_yaw = tag_data[3]

            cv2.rectangle(debug, (w - tw - 8, int(38 * fs)), (w - 4, int(120 * fs)), (0, 55, 0), -1)
            cv2.putText(debug, f"Tag #{tag_id}",
                        (w - tw + 4, int(56 * fs)), cv2.FONT_HERSHEY_SIMPLEX, fs * 0.6, (0, 255, 0), 2)
            cv2.putText(debug, f"x={tag_x:+.3f}", (w - tw + 4, int(74 * fs)),
                        cv2.FONT_HERSHEY_SIMPLEX, fs * 0.5, (150, 255, 150), 1)
            cv2.putText(debug, f"y={tag_y:+.3f}", (w - tw + 4, int(90 * fs)),
                        cv2.FONT_HERSHEY_SIMPLEX, fs * 0.5, (150, 255, 150), 1)
            cv2.putText(debug, f"θ={tag_yaw:+.1f}°", (w - tw + 4, int(106 * fs)),
                        cv2.FONT_HERSHEY_SIMPLEX, fs * 0.5, (150, 255, 150), 1)

            # 画面中央大字
            if _frame % 2 == 0:  # 隔帧绘制，减少闪烁
                cv2.putText(debug, f"[Tag {tag_id}]", (w // 2 - int(55 * fs), h // 2 - int(10 * fs)),
                            cv2.FONT_HERSHEY_SIMPLEX, fs * 1.1, (0, 255, 0), 2)
                cv2.putText(debug, f"({tag_x:.2f}, {tag_y:.2f}, {tag_yaw:.0f}°)",
                            (w // 2 - int(75 * fs), h // 2 + int(14 * fs)),
                            cv2.FONT_HERSHEY_SIMPLEX, fs * 0.45, (0, 255, 0), 1)
        else:
            cv2.rectangle(debug, (w - tw - 8, int(38 * fs)), (w - 4, int(88 * fs)), (55, 0, 0), -1)
            cv2.putText(debug, "Tag: 未识别",
                        (w - tw + 4, int(56 * fs)), cv2.FONT_HERSHEY_SIMPLEX, fs * 0.6, (0, 0, 255), 2)
            cv2.putText(debug, "等待检测中...", (w - tw + 4, int(74 * fs)),
                        cv2.FONT_HERSHEY_SIMPLEX, fs * 0.45, (200, 100, 100), 1)
            if _frame % DISPLAY_DEBUG_INTERVAL == 0:
                cv2.putText(debug, "[等待 Tag...]", (w // 2 - int(75 * fs), h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, fs * 0.9, (0, 0, 255), 2)

        # ── 3. 底部图例（每 5 帧画一次） ──
        if _frame % 5 == 1:
            cv2.putText(debug, "绿=Tag | 红=未识别 | Q=退出",
                        (8, h - 4), cv2.FONT_HERSHEY_SIMPLEX, fs * 0.35, (170, 170, 170), 1)

        cv2.imshow(win, debug)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            rospy.loginfo("[显示] 用户按 Q 退出程序")
            _combined_debug_state.stop()
            rospy.signal_shutdown("用户按 Q 退出")
            break

        # —— FPS 节流：显示只给人看，不需要满速渲染，省 CPU ——
        time.sleep(1.0 / DISPLAY_FPS)

    cv2.destroyWindow(win)
    rospy.loginfo("[显示] 独立显示线程退出")


def close_combined_debug():
    """关闭综合调试窗口"""
    _combined_debug_state.stop()


def find_box(img, color_name='green', debug=False):
    """返回 (x, y) 坐标，无面积"""
    if img is None:
        return None, None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, COLOR_RANGE[color_name][0], COLOR_RANGE[color_name][1])
    # 原实现对三通道 HSV 做高斯模糊再阈值，开销大且对色相模糊不合理；
    # 改为在二值 mask 上做开运算去噪 + 膨胀，更快且对二值图更合理
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _KERNEL4)
    opened = cv2.dilate(opened, _KERNEL4, iterations=2)
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    areas = [cv2.contourArea(c) for c in contours]
    max_idx = np.argmax(areas)
    (cx, cy), radius = cv2.minEnclosingCircle(contours[max_idx])
    cx_f, cy_f = tracker.update(cx, cy)
    if debug:
        disp = img.copy()
        cv2.circle(disp, (int(cx_f), int(cy_f)), int(radius), (0, 0, 255), 2)
        cv2.imshow("Box Detected", disp)
        cv2.waitKey(1)
    return cx_f, cy_f

def get_latest_image():
    """获取胸部相机最新图像，最多重试 5 次（每次间隔 0.1s），失败返回 None。"""
    for _ in range(5):
        img = img_buffer.get()
        if img is not None:
            return img
        time.sleep(0.1)
    return None

# ========================== Tag 间前进（仅靠 Tag 中断，不做白线循迹） ==========================
def move_forward(total_steps, step_mode='box', check_tag_ids=None):
    """
    沿行进方向前进指定步数，仅靠 Tag 码判断是否已到达下一个目标。
    每步前进前先检查目标 Tag：若检测到且大致对齐 → 立即中断前进并返回该 Tag ID。

    返回：
        None = 正常走完全部步数
        int  = 被中断的 Tag ID（主循环直接处理该 Tag）
    """
    rospy.loginfo(f"[前进] ▶ 开始前进 ({total_steps}步, {step_mode}模式)"
                  f"{'  🏷 监听Tag中断: '+str(check_tag_ids) if check_tag_ids else ''}")
    for i in range(total_steps):
        rospy.loginfo(f"[前进] ── 第 {i+1}/{total_steps} 步 ──")

        # ===== Tag 中断检查：前进之前，先看 Tag =====
        # 如果 Tag 已经大致对齐，就不用继续前进了，直接交给主循环处理
        if check_tag_ids is not None and tag_reader is not None:
            markers = tag_reader.get_nearest_marker()
            if len(markers) >= 4:
                tid = int(markers[0])
                if tid in check_tag_ids:
                    tag_x, tag_y, tag_yaw = markers[1], markers[2], markers[3] + TAG_YAW_OFFSET_DEG
                    cfg = TAG_NAV_CONFIG.get(tid, {})
                    x_off = cfg.get('x_offset', 0.0)
                    y_off = cfg.get('y_offset', 0.0)
                    th_off = cfg.get('theta_offset', 0.0)
                    x_th = cfg.get('x_thresh', TAG_X_THRESH) * TAG_INTERRUPT_THRESH_MULT  # 宽松阈值
                    y_th = cfg.get('y_thresh', TAG_Y_THRESH) * TAG_INTERRUPT_THRESH_MULT
                    th_th = cfg.get('theta_thresh', TAG_THETA_THRESH) * TAG_INTERRUPT_THETA_MULT

                    x_err = abs(tag_x - x_off)
                    y_err = abs(tag_y - y_off)
                    th_err = abs(tag_yaw - th_off)

                    rospy.loginfo(f"[前进] 📍 检测到Tag #{tid}  "
                                  f"x_err={x_err:.3f} y_err={y_err:.3f} th_err={th_err:.1f}  "
                                  f"阈值: x<{x_th:.3f} y<{y_th:.3f} th<{th_th:.1f}")

                    if x_err <= x_th and y_err <= y_th and th_err <= th_th:
                        rospy.loginfo(f"[前进] ⏭  Tag #{tid} 已大致对齐，中断前进！")
                        return tid  # 直接把 Tag ID 返回给主循环
                    else:
                        rospy.loginfo(f"[前进] 检测到Tag #{tid} 但未对齐，继续前进")

        # 前进一步
        if step_mode == 'box':
            box_go(1)
        else:
            go_fast(1)
        time.sleep(0.1)  # 等动作完成 + Tag 检测更新

    rospy.loginfo(f"[前进] ✓ 前进完成 ({total_steps}步)")
    return None


# ========================== 抱箱函数（前进三次后强制抱箱） ==========================
def approach_and_pick():
    """
    简化抱箱流程：识别方块 → 前进3步 → 夹紧 → 抬升。
    """
    rospy.loginfo("开始简化抱箱流程：识别方块 → 前进3步 → 夹紧 → 抬升")
    tracker.reset()

    img = get_latest_image()
    if img is None:
        rospy.logwarn("获取图像失败，仍继续执行抱箱")
    else:
        x, y = find_box(img, 'green')
        if x is None:
            rospy.logwarn("未检测到绿色方块，仍继续执行抱箱")
        else:
            rospy.loginfo(f"检测到方块坐标: x={x:.1f}, y={y:.1f}")

    rospy.loginfo("前进3步接近方块")
    for i in range(3):
        go_fast(1)
        time.sleep(0.2)

    rospy.loginfo("执行抱箱动作：夹紧 → 抬升")
    ActionExecutor.execute("grip_close", 1, wait=0.8)
    ActionExecutor.execute("lift_up", 1, wait=0.6)

    rospy.loginfo("抱箱动作完成")
    return True

# ========================== Tag 导航对正（使用传入的阈值，已改为前进） ==========================
def turn_to_tag(dis_x, dis_y, theta,
                x_offset=0.0, y_offset=0.0, theta_offset=0.0,
                x_thresh=TAG_X_THRESH, y_thresh=TAG_Y_THRESH,
                theta_thresh=TAG_THETA_THRESH, box_mode=True):
    x_err = dis_x - x_offset
    y_err = dis_y - y_offset
    theta_err = theta - theta_offset
    rospy.loginfo_throttle(1.0, f"误差: x={x_err:.3f}, y={y_err:.3f}, theta={theta_err:.1f}")
    if box_mode:
        if theta_err > 35:
            BoxL_turn2(1)
        elif theta_err < -35:
            BoxR_turn2(1)
        elif y_err > y_thresh + 0.05:
            BoxL_move2(1)
        elif y_err < -y_thresh - 0.05:
            BoxR_move2(1)
        # 修改：极近直接完成，否则前进（不再后退）
        elif x_err < 0.02:
            rospy.loginfo("距离极近，认为对正完成")
            return True
        elif x_err < x_thresh - 0.03:
            box_go(1)        # 原为Box_Back(1)
        elif x_err > x_thresh + 0.2:
            box_go2(1)
        elif x_err > x_thresh + 0.12:
            box_go1(1)
        elif x_err > x_thresh + 0.06:
            box_go(1)
        elif theta_err > theta_thresh:
            BoxL_turn1(1)
        elif theta_err < -theta_thresh:
            BoxR_turn1(1)
        elif y_err > y_thresh:
            BoxL_move1(1)
        elif y_err < -y_thresh:
            BoxR_move1(1)
        else:
            rospy.loginfo("抱箱模式对正完成")
            return True
    else:
        if theta_err > 35:
            L_turn2(1)
        elif theta_err < -35:
            R_turn2(1)
        elif y_err > y_thresh + 0.05:
            L_move2(1)
        elif y_err < -y_thresh - 0.05:
            R_move2(1)
        elif x_err < 0.02:
            rospy.loginfo("距离极近，认为对正完成")
            return True
        elif x_err < x_thresh - 0.10:
            go_fast(1)       # 原为Back1(1)
        elif x_err > x_thresh + 0.2:
            go_fast2(1)
        elif x_err > x_thresh + 0.12:
            go_fast1(1)
        elif x_err > x_thresh + 0.06:
            go_fast(1)
        elif theta_err > theta_thresh:
            L_turn1(1)
        elif theta_err < -theta_thresh:
            R_turn1(1)
        elif y_err > y_thresh:
            L_move1(1)
        elif y_err < -y_thresh:
            R_move1(1)
        else:
            rospy.loginfo("空载模式对正完成")
            return True
    return False

# ========================== 图像采集线程 ==========================
def image_capture_thread():
    reader = ImgConverter()
    while not rospy.is_shutdown():
        ret, frame = reader.chest_image()
        if frame is not None:
            # 低算力：把工作分辨率一次性缩小，后续箱子/显示都在小图上做，
            # 像素量按 WORK_SCALE² 下降。注意 <1 时需同步缩小顶部所有像素阈值。
            if WORK_SCALE < 1.0:
                frame = cv2.resize(frame, None, fx=WORK_SCALE, fy=WORK_SCALE,
                                   interpolation=cv2.INTER_NEAREST)
            img_buffer.update(frame)
        else:
            rospy.logwarn_throttle(5, "等待胸部相机图像")
        time.sleep(0.1)  # ~10fps，机器人动作步长 0.3-1s，完全够用


# ========================== Tag 分派 handler ==========================
def _handle_outbound_tag(tag_id, tag_x, tag_y, tag_yaw):
    """去程（抱箱）Tag 分派。返回 Flow 以控制主循环。"""
    cfg = TAG_NAV_CONFIG.get(tag_id, {})
    x_offset = cfg.get('x_offset', 0.0)
    y_offset = cfg.get('y_offset', 0.0)
    theta_offset = cfg.get('theta_offset', 0.0)
    x_thresh = cfg.get('x_thresh', TAG_X_THRESH)
    y_thresh = cfg.get('y_thresh', TAG_Y_THRESH)
    theta_thresh = cfg.get('theta_thresh', TAG_THETA_THRESH)

    if tag_id == 1:
        if state.tag_id == 1:
            result = turn_to_tag(tag_x, tag_y, tag_yaw,
                                 x_offset=x_offset, y_offset=y_offset,
                                 theta_offset=theta_offset,
                                 x_thresh=x_thresh, y_thresh=y_thresh,
                                 theta_thresh=theta_thresh, box_mode=True)
            if result:
                rospy.loginfo("一号 Tag 对正完成，前进至二号")
                state.tag_id = 2
                intr = move_forward(5, 'box', check_tag_ids=[2, 3, 4, 5])
                if intr is not None:
                    rospy.loginfo(f"[Tag] ⏭ 前进被 Tag #{intr} 中断，直接跳转")
                    state.tag_id = intr
                    return Flow.CONTINUE
        else:
            box_go2(2)
    elif tag_id == 2:
        if state.tag_id == 2:
            result = turn_to_tag(tag_x, tag_y, tag_yaw,
                                 x_offset=x_offset, y_offset=y_offset,
                                 theta_offset=theta_offset,
                                 x_thresh=x_thresh, y_thresh=y_thresh,
                                 theta_thresh=theta_thresh, box_mode=True)
            if result:
                rospy.loginfo("二号 Tag 对正完成，前进至三号")
                state.tag_id = 3
                intr = move_forward(3, 'box', check_tag_ids=[3, 4, 5])
                if intr is not None:
                    rospy.loginfo(f"[Tag] ⏭ 前进被 Tag #{intr} 中断")
                    state.tag_id = intr
                    return Flow.CONTINUE
                BoxR_move2(2)
        else:
            BoxR_move2(1)
    elif tag_id == 3:
        if state.tag_id == 3:
            result = turn_to_tag(tag_x, tag_y, tag_yaw,
                                 x_offset=x_offset, y_offset=y_offset,
                                 theta_offset=theta_offset,
                                 x_thresh=x_thresh, y_thresh=y_thresh,
                                 theta_thresh=theta_thresh, box_mode=True)
            if result:
                rospy.loginfo("三号 Tag 对正完成，右移至四号")
                state.tag_id = 4
                BoxR_move1(2)
                BoxL_turn2(1)
                intr = move_forward(6, 'box', check_tag_ids=[4, 5])
                if intr is not None:
                    rospy.loginfo(f"[Tag] ⏭ 前进被 Tag #{intr} 中断，跳过后续动作")
                    state.tag_id = intr
                    return Flow.CONTINUE
                BoxL_turn2(1)
                intr = move_forward(4, 'box', check_tag_ids=[4, 5])
                if intr is not None:
                    rospy.loginfo(f"[Tag] ⏭ 前进被 Tag #{intr} 中断，跳过后续动作")
                    state.tag_id = intr
                    return Flow.CONTINUE
                BoxL_turn2(1)
                box_go2(1)
                BoxL_move1(1)
        else:
            BoxR_move2(1)
    elif tag_id == 4:
        if state.tag_id == 4:
            result = turn_to_tag(tag_x, tag_y, tag_yaw,
                                 x_offset=x_offset, y_offset=y_offset,
                                 theta_offset=theta_offset,
                                 x_thresh=x_thresh, y_thresh=y_thresh,
                                 theta_thresh=theta_thresh, box_mode=True)
            if result:
                rospy.loginfo("四号 Tag 对正完成，左移至五号")
                state.tag_id = 5
                BoxL_move1(1)
        else:
            BoxL_move1(1)
    elif tag_id == 5:
        if state.tag_id == 5:
            result = turn_to_tag(tag_x, tag_y, tag_yaw,
                                 x_offset=x_offset, y_offset=y_offset,
                                 theta_offset=theta_offset,
                                 x_thresh=x_thresh, y_thresh=y_thresh,
                                 theta_thresh=theta_thresh, box_mode=True)
            if result:
                rospy.loginfo("五号 Tag 对正完成，前进放箱")
                # ⚡ 对正后不要再监听 Tag #5（否则刚对正又看到 Tag #5 → 无限循环），
                # 只监听后续 Tag #6（还没到放箱位就看到 #6 → 原地放箱再继续）
                intr = move_forward(3, 'box', check_tag_ids=[6])
                if intr is not None:
                    rospy.loginfo(f"[Tag] ⏭ 前进被 Tag #{intr} 中断，原地放箱")
                    Box_Down(1)
                    state.has_box = False
                    state.step = Step.INBOUND
                    state.tag_id = intr
                    return Flow.CONTINUE
                Box_Down(1)
                state.has_box = False
                rospy.loginfo("放下箱子，空载前进并转向")
                intr = move_forward(3, 'empty', check_tag_ids=[6])
                if intr is not None:
                    rospy.loginfo(f"[Tag] ⏭ 空载前进被 Tag #{intr} 中断")
                    state.step = Step.INBOUND
                    state.tag_id = intr
                    return Flow.CONTINUE
                R_turn2(3)
                state.step = Step.INBOUND
        # 非当前目标：忽略
    return Flow.FALLTHROUGH


def _handle_inbound_tag(tag_id, tag_x, tag_y, tag_yaw):
    """返程（空载）Tag 分派。返回 Flow 以控制主循环。"""
    cfg = TAG_NAV_CONFIG.get(tag_id, {})
    x_offset = cfg.get('x_offset', 0.0)
    y_offset = cfg.get('y_offset', 0.0)
    theta_offset = cfg.get('theta_offset', 0.0)
    x_thresh = cfg.get('x_thresh', TAG_X_THRESH)
    y_thresh = cfg.get('y_thresh', TAG_Y_THRESH)
    theta_thresh = cfg.get('theta_thresh', TAG_THETA_THRESH)

    if tag_id == 5:
        result = turn_to_tag(tag_x, tag_y, tag_yaw,
                             x_offset=0.08, y_offset=0.0, theta_offset=TAG_REVERSE_THETA_OFFSET,
                             x_thresh=0.12, y_thresh=0.12, theta_thresh=15.0,
                             box_mode=False)
        if result:
            state.phase = Phase.REVERSE_MOVING
            rospy.loginfo("五号反向对正完成，移至六号")
            state.tag_id = 6
            intr = move_forward(3, 'empty', check_tag_ids=[6])
            if intr is not None:
                rospy.loginfo(f"[Tag] ⏭ 前进被 Tag #{intr} 中断")
                state.tag_id = intr
                return Flow.CONTINUE
    elif tag_id == 6:
        if state.tag_id == 6:
            result = turn_to_tag(tag_x, tag_y, tag_yaw,
                                 x_offset=x_offset, y_offset=y_offset,
                                 theta_offset=theta_offset,
                                 x_thresh=x_thresh, y_thresh=y_thresh,
                                 theta_thresh=theta_thresh, box_mode=False)
            if result:
                rospy.loginfo("六号对正完成，左移至七号（前进）")
                state.tag_id = 7
                L_move2(6)
                L_turn2(1)
                intr = move_forward(5, 'empty', check_tag_ids=[7])
                if intr is not None:
                    rospy.loginfo(f"[Tag] ⏭ 前进被 Tag #{intr} 中断")
                    state.tag_id = intr
                    return Flow.CONTINUE
        # 非当前目标：忽略
    elif tag_id == 7:
        if state.tag_id == 7:
            result = turn_to_tag(tag_x, tag_y, tag_yaw,
                                 x_offset=x_offset, y_offset=y_offset,
                                 theta_offset=theta_offset,
                                 x_thresh=x_thresh, y_thresh=y_thresh,
                                 theta_thresh=theta_thresh, box_mode=False)
            if result:
                rospy.loginfo("七号对正完成，右移回一号")
                R_move2(5)
        # 非当前目标：忽略
    elif tag_id == 1:
        result = turn_to_tag(tag_x, tag_y, tag_yaw,
                             x_offset=0.1, y_offset=0.0, theta_offset=TAG_REVERSE_THETA_OFFSET,
                             x_thresh=0.12, y_thresh=0.12, theta_thresh=15.0,
                             box_mode=False)
        if result:
            rospy.loginfo("一号反向对正完成，前进抓块")
            move_forward(4, 'empty')
            state.reset()
    return Flow.FALLTHROUGH


# ========================== 主程序 ==========================
def main():
    global tag_reader
    rospy.init_node('aelos_smart_optimized', anonymous=False)
    tag_reader = TagConverter()
    time.sleep(0.5)

    th = threading.Thread(target=image_capture_thread, daemon=True)
    th.start()

    # ── 启动诊断 ──
    rospy.loginfo("=" * 45)
    rospy.loginfo("[诊断] 系统启动诊断")
    try:
        topics = rospy.get_published_topics()
        rospy.loginfo(f"[诊断] ROS topics ({len(topics)}):")
        for n, t in topics:
            rospy.loginfo(f"  ├─ {n}  ({t})")
    except Exception as e:
        rospy.logwarn(f"[诊断] 获取 ROS topics 失败: {e}")
    try:
        methods = [m for m in dir(ImgConverter) if not m.startswith('_')]
        rospy.loginfo(f"[诊断] ImgConverter 可用方法: {methods}")
    except Exception as e:
        rospy.logwarn(f"[诊断] 检查 ImgConverter 失败: {e}")
    rospy.loginfo("=" * 45)

    # 等待胸部相机图像就绪（带超时，避免相机异常时无限刷屏）
    chest_wait_start = time.time()
    while img_buffer.get() is None:
        if time.time() - chest_wait_start > 10.0:
            rospy.logwarn("[启动] ⏰ 等待胸部相机超时(10s)，继续启动")
            break
        rospy.loginfo_throttle(2, "等待胸部相机图像数据...")
        time.sleep(0.2)

    # ── 启动独立显示线程：画面从此稳定刷新 ──
    if DISPLAY_ENABLED:
        display_th = threading.Thread(target=display_thread_func, daemon=True)
        display_th.start()
        rospy.loginfo("[显示] 独立显示线程已启动（单摄像头模式）")
    else:
        rospy.loginfo("[显示] 显示线程已禁用（DISPLAY_ENABLED=False，节省算力）")

    state.reset()

    rospy.loginfo("启动完成，前进两步")
    go_fast2(2)

    rate = rospy.Rate(MAIN_LOOP_HZ)
    no_tag_count = 0

    while not rospy.is_shutdown():
        try:
            if state.tag_id == 0:
                # 综合调试：装箱阶段无 Tag
                show_combined_debug(None)

                if state.phase is Phase.START_BOX:
                    approach_and_pick()
                    state.has_box = True
                    state.phase = Phase.END_BOX
                elif state.phase is Phase.END_BOX:
                    if state.step is Step.OUTBOUND:
                        box_go2(5)
                        BoxR_turn2(3)
                        BoxL_move2(1)
                        state.tag_id = 1
                    elif state.step is Step.INBOUND:
                        BoxR_turn2(3)
                        state.step = Step.OUTBOUND
                        state.tag_id = 1
                    state.phase = Phase.START_MOVING
            else:
                markers = tag_reader.get_nearest_marker()
                if len(markers) < 4:
                    # 综合调试：显示无 Tag 状态
                    show_combined_debug(None)

                    # ── 无 Tag → 备用盲目前进 / 超时转向找 Tag ──
                    no_tag_count += 1
                    rospy.loginfo_throttle(2, f"无Tag (连续{no_tag_count}次)，盲目前进, ID={state.tag_id}, phase={state.phase.value}, step={state.step.value}")
                    if no_tag_count > NO_TAG_TURN_THRESH:
                        rospy.logwarn("连续无Tag次数过多，执行转向寻找 Tag")
                        if state.step is Step.OUTBOUND:
                            BoxL_turn1(1)
                        else:
                            L_turn1(1)
                        no_tag_count = 0
                        rate.sleep()
                        continue

                    # 备用动作（统一前进，避免卡死）
                    if state.tag_id == 1 and state.phase is Phase.END_BOX:
                        BoxL_move1(1)
                    elif state.tag_id == 1 and state.phase is Phase.START_MOVING:
                        box_go(1)
                    elif state.tag_id == 2:
                        box_go(1)
                    elif state.tag_id == 3:
                        box_go(1)
                    elif state.tag_id == 4:
                        box_go2(1)
                    elif state.tag_id == 5 and state.step is Step.OUTBOUND:
                        box_go(1)
                    elif state.tag_id == 5 and state.step is Step.INBOUND and state.phase is Phase.START_MOVING:
                        BoxL_move1(1)
                    elif state.tag_id == 6:
                        go_fast(1)
                    elif state.tag_id == 7:
                        go_fast(1)
                    elif state.tag_id == 5 and state.step is Step.INBOUND and state.phase is Phase.REVERSE_MOVING:
                        box_go(1)
                    else:
                        box_go(1)
                else:
                    no_tag_count = 0
                    tag_id = markers[0]
                    tag_x = markers[1]
                    tag_y = markers[2]
                    tag_yaw = markers[3] + TAG_YAW_OFFSET_DEG

                    # —— 综合调试：显示 Tag 状态 ——
                    show_combined_debug(markers)

                    # ---------- 通用跳转逻辑 ----------
                    # 允许检测到后续 Tag 时直接跳转，但跳过放箱关键阶段
                    if state.tag_id != 0:
                        if state.step is Step.OUTBOUND:
                            # 去程只允许跳到 1~5（6/7 无去程 handler，跳过去会死锁）
                            can_skip = state.tag_id < tag_id <= 5
                        else:
                            can_skip = tag_id > state.tag_id
                        if can_skip:
                            rospy.logwarn(f"检测到后续 Tag {tag_id}（当前 ID={state.tag_id}），直接跳转")
                            state.tag_id = tag_id
                    # ---------------------------------

                    if state.step is Step.OUTBOUND:
                        flow = _handle_outbound_tag(tag_id, tag_x, tag_y, tag_yaw)
                    else:
                        flow = _handle_inbound_tag(tag_id, tag_x, tag_y, tag_yaw)

                    if flow is Flow.CONTINUE:
                        continue
            rate.sleep()
        except rospy.ROSInterruptException:
            break
        except Exception as e:
            rospy.logerr(f"主循环异常: {e}")
            time.sleep(0.5)

if __name__ == '__main__':
    main()