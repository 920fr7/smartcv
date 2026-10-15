import time
import cv2
import threading
import numpy as np
import rospy
import math
import sys
import subprocess
from enum import Enum, auto

# 用户自定义模块
from image_Tag_converter import ImgConverter, TagConverter

# ROS 图像消息（用于头部摄像头 topic 订阅）
from sensor_msgs.msg import Image as RosImage

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
# 白线前进过程中"Tag 已大致对齐即中断"的宽松阈值倍数
TAG_INTERRUPT_THRESH_MULT = 2.5
TAG_INTERRUPT_THETA_MULT = 2.0

# 动作执行默认等待时间（秒）
ACTION_WAIT = 0.3
ACTION_TIMEOUT = 5.0

# 连续无Tag次数阈值，触发转向
NO_TAG_TURN_THRESH = 10

# ==================== 白线检测与跟踪参数 ====================
WHITE_LOWER = np.array([0, 0, 100])        # 白线 HSV 下限（低饱和度+中高亮度=通用白色）
WHITE_UPPER = np.array([180, 60, 255])    # 白线 HSV 上限（容忍光照变化导致色相/饱和度漂移）
LINE_ROI_BOTTOM = 0.35                     # ROI 取底部 35%
LINE_MIN_AREA = 30                         # 白线最小轮廓面积（过滤像素级噪声）
LINE_OFFSET_DEADZONE = 10                  # 横向偏移死区（像素）
LINE_OFFSET_CRITICAL = 50                 # ⚠️ 白线严重偏移阈值（像素）：超过此值则 Tag 优先暂时让给白线纠偏，防止跑飞
LINE_CORRECTION_MAX_STEPS = 2              # 单次纠偏最大横移步数
LINE_ANGLE_DEADZONE = 8                   # 白线角度死区（度）：小于此值不转向，防止小幅抖动
ANGLE_TURN_THRESH = 15                    # 显著弯道角度（度）：达到此值执行大幅度转向
CORNER_ANGLE_THRESH = 35                 # ⛰️ 直角拐弯检测阈值（度）：白线角度超过此值说明正在经过正方形那样的直角弯
CORNER_TURN_STEPS = 2                     # 直角拐弯的转向步数（每步 ~1s Box 或 ~0.5s 空载）
                                          # 数值越大转得越狠：1=小转弯 / 2=标准直角弯 / 3=大角度急弯
                                          # 抱箱模式下用 BoxR_turn2/BoxL_turn2，空载用 R_turn2/L_turn2
                                          # 具体走几步会在 handle_corner() 里引用这个值

# ==================== 预判控制参数 ====================
PREDICTIVE_GAIN = 0.4          # 预判强度：0=纯反应式，0.3=温和预判，0.7=激进预判
PREDICTIVE_LOOKAHEAD = 35      # 预判距离（像素）：往前看多远来推算白线趋势

# ==================== 显示优化参数（减轻画面卡顿） ====================
DISPLAY_SCALE = 0.6            # 显示缩放比例：1.0=原尺寸，0.6=缩到60%（像素量减64%，大幅降CPU）
DISPLAY_DEBUG_INTERVAL = 3     # 调试帧绘制间隔：1=每帧都画（最卡），3=每3帧画一次
DETECT_DEBUG_INTERVAL = 2      # detect()调试绘制间隔：1=每次都画，2=每2次画一次

# ==================== 双摄像头配置 ====================
HEAD_CAM_ENABLED = True        # 是否启用头部摄像头（用于白线检测）
HEAD_CAM_DISPLAY_SCALE = 0.35  # 头部摄像头 PIP 画中画缩放比例
HEAD_CAPTURE_FPS = 10          # 头部摄像头采集帧率
HEAD_CAM_ROS_TOPIC = None      # 🎯 头部摄像头 ROS topic 名称（None=自动检测）
                               # 机器人上 ImgConverter 占用 GStreamer 导致 OpenCV 无法打开头部
                               # 改用 ROS topic 订阅方式获取头部画面

# ==================== Tag 优先级策略（重要！） ====================
# 🏷️ 双摄像头始终同时工作 + Tag 优先级（白线严重偏移时例外）
#   - 胸部摄像头 → Tag 识别（每帧），通过 ImgConverter.chest_image()
#   - 头部摄像头 → 白线检测（每帧，无论有无 Tag）
#     ⚡ 优先使用 ROS topic 订阅获取头部画面，线程内调用 spin_once()
#     自动检测可用 Image topic，排除胸部 topic
#     如果 ROS 订阅失败，回退到 OpenCV（CAP_V4L2 后端）
#     如果 OpenCV 也失败，回退到胸部摄像头
#   检测到 Tag + 白线正常（≤CRITICAL）→ 🏷️ 听 Tag 指令
#   检测到 Tag + 白线严重偏移（>CRITICAL）→ ⚠️ 先纠偏白线再处理 Tag
#   无 Tag      → 用头部摄像头白线循迹控制
#   白线也没有 → 备用盲目前进 + 超时转向
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

    @property
    def step_mode(self):
        """抱箱/空载动作模式，跟随真实负载 has_box（而非按 Tag 号推断）。"""
        return 'box' if self.has_box else 'empty'

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
head_img_buffer = ImageBuffer()     # 头部相机图像缓存（用于白线检测）
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


class LineFollower:
    """白线跟踪器：检测画面底部的白色引导线，计算横向偏移，执行纠偏。"""
    def __init__(self):
        self.last_offset = 0.0
        self.offset_smooth = 0.0    # 低通滤波后的偏移量（防抖动→防S型路径）
        self.last_angle = None      # 白线方向角度（弧度），用于弯道检测
        self.angle_smooth = 0.0     # 低通滤波后的角度偏差（度），正=右弯 负=左弯
        self.angle_dev = 0.0        # 当前角度偏差（度），供纠偏使用
        self._debug = True          # ← 默认开启调试，让你能看见白线状态
        self._debug_dir = None      # 设为路径可保存调试帧到文件
        self._frame_count = 0
        self._debug_counter = 0     # 调试绘制帧计数器（控制绘制频率，减少CPU占用）
        self._last_log = ""         # 防重复日志
        self._last_debug_img = None # 最新调试帧（供综合窗口叠加 Tag 信息）
        self.corner_state = 0          # ⛰️ 直角拐角状态: 0=无, 1=右拐, -1=左拐
        self.corner_handled = False    # 拐角是否已执行转向（防重复动作）
        self._tag_interrupted = False  # ⚡ Tag 中断标记：当 move_forward 被 Tag 打断时为 True
        self.predicted_offset = 0.0    # 🔮 预判偏移量（在 detect() 中计算，供 correct() 使用）

    def detect(self, img):
        """
        检测白线，返回横向偏移量（像素），正=偏右，负=偏左，None=未检测到。
        偏移量绝对值越大说明偏离越严重。
        """
        if img is None:
            return None
        h, w = img.shape[:2]
        roi_y = int(h * (1 - LINE_ROI_BOTTOM))
        roi = img[roi_y:, :]

        # HSV 阈值提取白色区域
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, WHITE_LOWER, WHITE_UPPER)

        # 形态学去噪
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # 找最大轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            if self._debug:
                msg = f"[白线] ⚠️ 无白色轮廓  画面={w}x{h}  ROI底部={roi_y}:{h}"
                if msg != self._last_log:
                    rospy.logwarn(msg)
                    self._last_log = msg
                # 存储无白线的基础帧
                base = img.copy()
                cv2.rectangle(base, (0, roi_y), (w, h), (255, 0, 0), 2)
                cv2.putText(base, "白线: 未检测到", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
                self._last_debug_img = base
            return None

        # —— 多白线处理：过滤有效轮廓，选最靠近画面中心的那个 ——
        # 先过滤掉面积太小的噪声
        valid = [(c, cv2.contourArea(c)) for c in contours]
        valid = [(c, a) for c, a in valid if a >= LINE_MIN_AREA]

        if not valid:
            if self._debug:
                msg = f"[白线] ⚠️ 所有轮廓面积均 < {LINE_MIN_AREA}（共 {len(contours)} 个）"
                if msg != self._last_log:
                    rospy.logwarn(msg)
                    self._last_log = msg
                base = img.copy()
                cv2.rectangle(base, (0, roi_y), (w, h), (255, 0, 0), 2)
                cv2.putText(base, f"白线: 面积太小（{len(contours)}个噪声）", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
                self._last_debug_img = base
            return None

        if len(valid) > 1:
            # 有多个有效白线 → 选最靠近画面中心的那个
            # 避免多条白线（场地标记、反光）导致最大轮廓来回跳 → 左右徘徊
            cx_list = []
            for c, a in valid:
                M = cv2.moments(c)
                cx = M['m10'] / M['m00'] if M['m00'] != 0 else w
                cx_list.append(cx)
            # 选质心离画面中心最近的轮廓
            w_center = w / 2.0
            best_idx = min(range(len(valid)), key=lambda i: abs(cx_list[i] - w_center))
            target, area = valid[best_idx]
            cx = cx_list[best_idx]
            rospy.loginfo(f"[白线] 🚦 检测到 {len(valid)} 个白线轮廓，选中距中心最近的（cx={cx:.0f}）")
            M = cv2.moments(target)
        else:
            # 只有一个有效轮廓 → 用最大的那个（跟之前一样）
            target, area = valid[0]
            M = cv2.moments(target)
            if M['m00'] == 0:
                return None
            cx = M['m10'] / M['m00']

        # 相对于图像中心的偏移量（像素）
        self.last_offset = cx - w / 2.0

        # —— 一阶低通滤波：防止偏移量帧间抖动导致S型路径 ——
        # 原始偏移量里面如果有帧间噪声（1~5px的跳动），直接用会→纠偏抖动→S型
        # 平滑后只反应"真正偏离中心"的趋势，不反应噪声
        alpha = 0.35  # 新测量权重（越小越平滑）
        self.offset_smooth = alpha * self.last_offset + (1 - alpha) * self.offset_smooth
        # 极小偏移归零，防止积分残余
        if abs(self.offset_smooth) < 3:
            self.offset_smooth = 0.0

        # —— 白线角度检测：计算白线在 ROI 中的方向 ——
        # 当白线是倾斜的（不是垂直向上），说明赛道在转弯
        # 角度偏差 = 弯道方向 → 需要转向来跟随弯道
        if len(target) >= 20:  # 至少 20 个轮廓点才能拟合出可靠的角度
            vx, vy, _, _ = cv2.fitLine(target, cv2.DIST_L2, 0, 0.01, 0.01)
            # ⚡ 角度偏差计算沿用 fitLine 原始方向，PID 控制依赖此符号
            # 方向模糊问题仅在箭头显示处归一化处理（只影响画面显示）
            raw_angle = math.atan2(vy, vx)  # 弧度，垂直向上 ≈ -90°
            raw_dev = math.degrees(raw_angle - (-math.pi / 2))  # 相对垂直的偏差（度）
            # 一阶低通滤波
            alpha_angle = 0.30
            if self.last_angle is None:
                self.angle_smooth = raw_dev
            else:
                self.angle_smooth = alpha_angle * raw_dev + (1 - alpha_angle) * self.angle_smooth
            self.last_angle = raw_angle
            # 小角度归零防抖
            self.angle_dev = self.angle_smooth if abs(self.angle_smooth) >= 3 else 0.0

            # 🔮 预判偏移量：结合当前偏移 + 角度趋势，推算"再往前走一段白线会在哪"
            # 预测公式：预判偏移 = 当前平滑偏移 + tan(角度°) × 预判距离
            # 角度越陡→预判偏移越大→提前响应弯道（不等真正偏出去再纠偏）
            if abs(self.angle_dev) >= 3:
                pred_delta = math.tan(math.radians(self.angle_dev)) * PREDICTIVE_LOOKAHEAD
                pred_delta = max(-120, min(120, pred_delta))  # 钳位防暴走
                self.predicted_offset = self.offset_smooth + pred_delta
            else:
                self.predicted_offset = self.offset_smooth
        else:
            self.last_angle = None
            self.angle_dev = 0.0
            self.predicted_offset = self.offset_smooth  # 无角度时退化为当前偏移

        # —— ⛰️ 直角拐弯检测 ——
        # 赛道是"正方形的角"那样的直角弯，不是平滑 S 弯
        # 当白线角度偏差超过阈值 → 正在过直角弯
        if abs(self.angle_dev) >= CORNER_ANGLE_THRESH:
            new_state = 1 if self.angle_dev > 0 else -1
            if self.corner_state != new_state:
                rospy.loginfo(f"[白线] ⛰️ 检测到直角弯！方向={'右↻' if new_state > 0 else '左↺'}  角度={self.angle_dev:+.0f}°")
            self.corner_state = new_state
        # 拐角结束：角度回落到死区以下 → 重置状态
        # 不需要判断 corner_handled：如果拐弯后角度卡在 8°~35° 之间，
        # 不加这个条件会导致 corner_state 僵死→correct_angle()渐进转向被跳过
        if abs(self.angle_dev) < LINE_ANGLE_DEADZONE and self.corner_state != 0:
            if self.corner_handled:
                rospy.loginfo(f"[白线] ✓ 直角弯结束（角度={self.angle_dev:+.0f}°）")
            self.corner_state = 0
            self.corner_handled = False

        # —— 调试可视化：存储白线轮廓帧，供综合窗口使用 ——
        self._debug_counter += 1
        should_draw_debug = self._debug and (self._debug_counter % DETECT_DEBUG_INTERVAL == 0)
        if should_draw_debug:
            debug_img = img.copy()
            cv2.rectangle(debug_img, (0, roi_y), (w, h), (255, 0, 0), 2)    # ROI 蓝框
         
            for c, a in valid:
                alpha = 0.3 if (c is not target) else 1.0
                color = (0, int(255 * alpha), 0)
                cv2.drawContours(debug_img[roi_y:], [c], -1, color, 2)
            # 选中的轮廓用亮绿色加粗
            cv2.drawContours(debug_img[roi_y:], [target], -1, (0, 255, 0), 3)
            cv2.circle(debug_img[roi_y:], (int(cx), int(M['m01'] / M['m00'])), 5, (0, 0, 255), -1)  # 质心红点
            cv2.line(debug_img[roi_y:], (w // 2, 0), (w // 2, h - roi_y), (255, 255, 0), 1)  # 画面中线
            cv2.putText(debug_img, f"偏移: {self.last_offset:+.0f}px", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(debug_img, f"白线: {len(valid)}条 | 选中面积: {area:.0f}px", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
            # 显示角度/拐角信息（如果有）
            if self.last_angle is not None and abs(self.angle_dev) >= 3:
                y_info = 90
                if self.corner_state != 0:
                    dir_str = "右↻" if self.corner_state > 0 else "左↺"
                    status_str = "✓已转" if self.corner_handled else "⚡检测到!"
                    cv2.putText(debug_img, f"⛰️ 直角弯 {dir_str} {status_str} 角度={self.angle_dev:+.0f}°",
                                (10, y_info), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                else:
                    turn_dir = "右弯↻" if self.angle_dev > 0 else "左弯↺"
                    cv2.putText(debug_img, f"角度: {self.angle_dev:+.0f}° {turn_dir}", (10, y_info),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 150, 0), 1)
                # 🔮 预判偏移显示（当预判与当前差异明显时）
                if abs(self.predicted_offset - self.offset_smooth) > 5:
                    cv2.putText(debug_img, f"🔮 预判偏移: {self.predicted_offset:+.0f}px (增益={PREDICTIVE_GAIN})",
                                (10, y_info + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 100, 255), 1)
                # 画方向线：用 fitLine 的方向向量画一条紫色箭头
                # ⚠️ fitLine 返回的方向是"线延伸方向"，不区分前后（方向模糊）
                # 如果不归一化，箭头可能指向上方（正确=前进方向）或下方（错误=倒退方向）
                # 修复：归一化为始终指向上方（画面 y 负方向=机器人前进方向）
                if len(target) >= 20:
                    # 复用上方已计算的 fitLine 方向向量（无需重复拟合）
                    # 如果向量指向下方（vy > 0），取反使箭头始终指向上方
                    if vy > 0:
                        vx, vy = -vx, -vy
                    cy_roi = int(M['m01'] / M['m00'])
                    cx_line = int(cx)
                    cy_line = cy_roi
                    arrow_len = 60
                    p1 = (int(cx_line - vx * arrow_len), int(roi_y + cy_line - vy * arrow_len))
                    p2 = (int(cx_line + vx * arrow_len), int(roi_y + cy_line + vy * arrow_len))
                    cv2.arrowedLine(debug_img, p1, p2, (255, 0, 255), 2, tipLength=0.3)
            self._last_debug_img = debug_img  # 存下来让综合窗口使用

        # —— 非调试模式也有控制台输出 ——
        direction = "右偏" if self.last_offset > LINE_OFFSET_DEADZONE else ("左偏" if self.last_offset < -LINE_OFFSET_DEADZONE else "居中")
        contour_info = f"  [{len(valid)}条白线]" if len(valid) > 1 else ""
        msg = f"[白线] 检测到！偏移={self.last_offset:+.0f}px  {direction}  面积={area:.0f}px{contour_info}"
        if msg != self._last_log:
            rospy.loginfo(msg)
            self._last_log = msg

        return self.last_offset

    def correct(self, step_mode='box', max_steps=None):
        """根据当前白线偏移执行横向纠偏，返回实际移动步数。

        🔮 预判模式：结合角度趋势推算「再往前走一段白线会在哪」，
        提前响应弯道，减少 S 型抖动。
        PREDICTIVE_GAIN=0 → 纯反应式（只看当前偏移）
        PREDICTIVE_GAIN=1 → 激进预判（完全用推算偏移）
        """
        # 预判偏移 = 当前偏移和预判偏移的加权组合
        offset = (1 - PREDICTIVE_GAIN) * self.offset_smooth + PREDICTIVE_GAIN * self.predicted_offset
        if abs(offset) < LINE_OFFSET_DEADZONE:
            return 0

        if max_steps is None:
            max_steps = LINE_CORRECTION_MAX_STEPS

        # 偏移量越大，移动步数越多（除数 /40 比之前 /25 更精细，避免一步过冲）
        steps = min(int(abs(offset) / 40) + 1, max_steps)
        if offset > 0:  # 白线偏右 → 机器人需要左移
            rospy.loginfo(f"[白线] 𝄜 偏右 {offset:.0f}px → 左移 {steps} 步 ({step_mode})")
            if step_mode == 'box':
                BoxL_move1(steps)
            else:
                L_move1(steps)
        else:  # 白线偏左 → 机器人需要右移
            rospy.loginfo(f"[白线] 𝄜 偏左 {offset:.0f}px → 右移 {steps} 步 ({step_mode})")
            if step_mode == 'box':
                BoxR_move1(steps)
            else:
                R_move1(steps)
        return steps

    def correct_angle(self, step_mode='box'):
        """根据白线角度偏差执行转向纠偏，返回实际转向步数。

        当白线在 ROI 中倾斜时，说明赛道在转弯。
        角度偏差越大 → 弯越急 → 需要更多转向步数来跟随弯道。
        与 correct()（横向纠偏）配合使用：前者保持居中，后者保持方向。
        """
        dev = self.angle_dev
        if abs(dev) < LINE_ANGLE_DEADZONE:
            return 0

        # 角度越大，转向越大幅度
        if abs(dev) < ANGLE_TURN_THRESH:
            steps = 1  # 小弯：微调
        else:
            steps = min(int(abs(dev) / ANGLE_TURN_THRESH), 2)  # 大弯：多转几步

        if dev > 0:  # 白线偏右 → 右弯 → 右转跟随
            rospy.loginfo(f"[白线] ↻ 右弯 {dev:.0f}° → 右转 {steps} 步 ({step_mode})")
            if step_mode == 'box':
                BoxR_turn1(steps)
            else:
                R_turn1(steps)
        else:  # 白线偏左 → 左弯 → 左转跟随
            rospy.loginfo(f"[白线] ↺ 左弯 {abs(dev):.0f}° → 左转 {steps} 步 ({step_mode})")
            if step_mode == 'box':
                BoxL_turn1(steps)
            else:
                L_turn1(steps)
        return steps

    def handle_corner(self, step_mode='box'):
        """⛰️ 直角拐弯处理：检测到正方形那样的直角弯时，执行硬转向而不是渐进纠偏。

        与 correct_angle()（平滑弯道渐进转向）不同，直角弯需要一次性的硬转向，
        否则机器人会沿切线方向冲出赛道。

        返回:
            True  = 已执行拐弯
            False = 无需处理（无拐角或已处理）
        """
        if self.corner_state == 0 or self.corner_handled:
            return False

        steps = CORNER_TURN_STEPS  # 引用参数定义的转向步数
        rospy.loginfo(f"[白线] ⛰️ 直角拐弯！方向={'右↻' if self.corner_state > 0 else '左↺'}  步数={steps}  ({step_mode})")

        if self.corner_state > 0:  # 右直角弯
            if step_mode == 'box':
                BoxR_turn2(steps)
            else:
                R_turn2(steps)
        else:  # 左直角弯
            if step_mode == 'box':
                BoxL_turn2(steps)
            else:
                L_turn2(steps)

        self.corner_handled = True
        return True

    def priority_align(self, step_mode='box', max_iterations=5, timeout=3.0):
        """
        🚦 白线最高优先级对齐模式
        ===================================
        只要检测到白线且偏移超过死区，就持续纠偏直到居中。
        在 Tag 导航前调用此方法，确保白线优先对齐。

        ⚡ 如果 _tag_interrupted 为 True（来自 move_forward 的 Tag 中断），
        说明机器人已经大致对齐，跳过此步骤避免重复动作。

        参数:
            step_mode: 'box' 或 'empty'
            max_iterations: 最大纠偏次数
            timeout: 超时时间（秒）

        返回:
            True  = 已对齐（或未检测到白线，不阻塞）
            False = 超时或异常
        """
        # ⚡ Tag 中断跳过：机器人已经在白线跟随中大致对齐了
        if self._tag_interrupted:
            rospy.loginfo(f"[白线] ⏭ 跳过 priority_align（Tag 中断标记，已大致对齐）")
            self._tag_interrupted = False
            return True
        rospy.loginfo(f"[白线] 🚦 白线优先对齐启动 ({step_mode}模式)")

        # 第一步：先采样一次看看有没有白线（用头部摄像头）
        img = get_latest_head_image()
        if img is None:
            rospy.logwarn("[白线] 优先对齐：无法获取图像")
            return False

        first_offset = self.detect(img)
        if first_offset is None:
            rospy.loginfo("[白线] 优先对齐：视野内无白线，跳过对齐")
            return True   # 没白线就不阻塞

        # 有白线 → 持续纠偏到居中
        start_time = time.time()
        aligned = False

        for i in range(max_iterations):
            # 超时检查
            if time.time() - start_time > timeout:
                rospy.logwarn(f"[白线] ⏰ 优先对齐超时 ({timeout}s)")
                break

            # 采图检测（用头部摄像头）
            img = get_latest_head_image()
            if img is None:
                time.sleep(0.1)
                continue

            offset = self.detect(img)
            if offset is None:
                # 走着走着白线没了 → 说明可能已经走到位了
                rospy.loginfo(f"[白线] 第{i+1}次：白线消失（可能已到达）")
                aligned = True
                break

            if abs(offset) < LINE_OFFSET_DEADZONE:
                rospy.loginfo(f"[白线] ✓ 已对齐！（偏移={offset:+.0f}px ≤ 死区{LINE_OFFSET_DEADZONE}px）🎯")
                aligned = True
                break

            # 有偏移 → 纠偏一步
            rospy.loginfo(f"[白线] 第{i+1}/{max_iterations}次纠偏：偏移={offset:+.0f}px")
            self.correct(step_mode)
            time.sleep(0.15)

        if not aligned:
            final_offset = getattr(self, 'last_offset', 0)
            rospy.logwarn(f"[白线] 优先对齐结束：未完全居中（偏移={final_offset:+.0f}px）")

        return aligned

    def move_forward(self, total_steps, step_mode='box', check_tag_ids=None):
        """
        沿白线前进指定步数，每步都采样白线位置做横向纠偏。

        新增参数 check_tag_ids：指定一组 Tag ID。
        每步前进后会检查这些 Tag，如果检测到且大致对齐
        → 立即中断白线跟随，返回该 Tag ID。

        返回：
            None = 正常走完全部步数
            int  = 被中断的 Tag ID（主循环直接处理该 Tag）
        """
        rospy.loginfo(f"[白线] ▶ 开始白线引导前进 ({total_steps}步, {step_mode}模式)"
                      f"{'  🏷 监听Tag中断: '+str(check_tag_ids) if check_tag_ids else ''}")
        corrected_count = 0
        detected_count = 0
        for i in range(total_steps):
            rospy.loginfo(f"[白线] ── 第 {i+1}/{total_steps} 步 ──")

            # ===== ⭐ Tag 中断检查：在白线检测之前，先看 Tag =====
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

                        rospy.loginfo(f"[白线] 📍 检测到Tag #{tid}  "
                                      f"x_err={x_err:.3f} y_err={y_err:.3f} th_err={th_err:.1f}  "
                                      f"阈值: x<{x_th:.3f} y<{y_th:.3f} th<{th_th:.1f}")

                        if x_err <= x_th and y_err <= y_th and th_err <= th_th:
                            rospy.loginfo(f"[白线] ⏭  Tag #{tid} 已大致对齐，中断白线跟随！")
                            self._tag_interrupted = True  # 标记中断，跳过后续 priority_align
                            return tid  # ← 直接把 Tag ID 返回给主循环
                        else:
                            rospy.loginfo(f"[白线] 检测到Tag #{tid} 但未对齐，继续白线跟随")

            # 第一步：前进
            if step_mode == 'box':
                box_go(1)
            else:
                go_fast(1)

            # 第二步：采样白线（用头部摄像头）
            time.sleep(0.12)
            img = get_latest_head_image()
            if img is not None:
                offset = self.detect(img)
                if offset is not None:
                    detected_count += 1

                    # —— ⛰️ 直角拐弯优先处理 ——
                    if self.corner_state != 0 and not self.corner_handled:
                        self.handle_corner(step_mode)
                        corrected_count += 1
                        continue

                    # —— 横向纠偏（保持居中） ——
                    if abs(offset) >= LINE_OFFSET_DEADZONE * 1.5:
                        self.correct(step_mode)
                        corrected_count += 1
                    elif abs(offset) >= LINE_OFFSET_DEADZONE:
                        rospy.loginfo(f"[白线] ∿ 小偏移 {offset:+.0f}px，容忍不纠偏（防S型）")
                    else:
                        rospy.loginfo(f"[白线] ✓ 偏移 {offset:+.0f}px 在死区内，不需纠偏")

                    # —— 角度纠偏（平滑弯道） ——
                    if abs(self.angle_dev) >= LINE_ANGLE_DEADZONE and self.corner_state == 0:
                        self.correct_angle(step_mode)
                        corrected_count += 1
                    elif abs(self.angle_dev) >= 3 and self.corner_state == 0:
                        rospy.loginfo(f"[白线] ∿ 小角度 {self.angle_dev:+.0f}°，容忍不转向")
                else:
                    rospy.logwarn(f"[白线] ✗ 第 {i+1} 步未检测到白线！可能：①光照变化 ②HSV参数不匹配 ③白线不在视野内")
            else:
                rospy.logwarn(f"[白线] ✗ 第 {i+1} 步无法获取图像")

            time.sleep(0.08)

        rospy.loginfo(f"[白线] ✓ 前进完成 ({total_steps}步, 检测到{detected_count}次, 纠偏{corrected_count}次)"
                      f"{'  ⏭ 未被打断' if check_tag_ids else ''}")
        return None


# ======================== 综合调试：白线 + Tag 可视化 ========================
_combined_debug_window_name = "[综合调试] 白线 + Tag 状态"


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


def show_combined_debug(img, tag_data=None, nav_info=None):
    """更新综合调试状态，实际渲染由独立显示线程完成"""
    if tag_data is not None:
        _combined_debug_state.update_tag(tag_data)
    _combined_debug_state.update_nav(nav_id=state.tag_id, nav_step=state.step.value,
                                     nav_level=state.phase.value)


def display_thread_func():
    """
    独立显示线程：双摄像头综合调试画面。

    🖥️ 布局说明（双摄像头）：
      主画面 = 胸部相机（Tag 识别 + 状态栏 + 白线状态文字）
      PIP    = 头部相机（白线检测图像 + 轮廓绘制，右下角画中画）

    🚀 性能优化：
    1. 显示缩放到 DISPLAY_SCALE（默认60%）→ 像素量仅为 36%
    2. 胸部相机为底图，Tag 信息叠加其上
    3. 头部相机画面以 PIP 形式缩小叠加在右下角
    """
    win = _combined_debug_window_name
    cv2.namedWindow(win)
    rospy.loginfo(f"[显示] 双摄像头显示线程启动（缩放={DISPLAY_SCALE:.0%}，PIP={HEAD_CAM_DISPLAY_SCALE:.0%}）")

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
        if HEAD_CAM_ENABLED:
            label += " | 📷 双摄"
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

        # ── 3. 白线状态（左上角） ──
        if line_follower is not None:
            smooth_off = line_follower.offset_smooth
            angle_dev = line_follower.angle_dev
            corner_state = line_follower.corner_state

            if abs(smooth_off) > 0.1:
                direction = "右偏" if smooth_off > LINE_OFFSET_DEADZONE else (
                    "左偏" if smooth_off < -LINE_OFFSET_DEADZONE else "居中")
                clr = (0, 255, 0) if abs(smooth_off) < LINE_OFFSET_DEADZONE else (0, 0, 255)
                cam_label = "头部" if HEAD_CAM_ENABLED else ""
                cv2.putText(debug, f"白线: {smooth_off:+.0f}px {direction} [{cam_label}]", (8, int(48 * fs)),
                            cv2.FONT_HERSHEY_SIMPLEX, fs * 0.5, clr, 2)

                yl = int(64 * fs)
                if corner_state != 0:
                    d = "右↻" if corner_state > 0 else "左↺"
                    s = "已转" if line_follower.corner_handled else "检测到!"
                    cv2.putText(debug, f"直角弯 {d} ({s})", (8, yl),
                                cv2.FONT_HERSHEY_SIMPLEX, fs * 0.5, (0, 0, 255), 2)
                elif abs(angle_dev) >= LINE_ANGLE_DEADZONE:
                    td = "右弯↻" if angle_dev > 0 else "左弯↺"
                    cv2.putText(debug, f"弯道: {angle_dev:+.0f}° {td}", (8, yl),
                                cv2.FONT_HERSHEY_SIMPLEX, fs * 0.5, (255, 150, 0), 2)

                # 头部相机健康状态（左上角更下面一行）
                if HEAD_CAM_ENABLED:
                    head_age = head_img_buffer.get_age()
                    if head_age is not None:
                        head_status = f"头部: {head_age:.1f}s前" if head_age < 3 else f"⚠️ 头部: {head_age:.1f}s前"
                        head_clr = (0, 255, 0) if head_age < 1 else (0, 165, 255) if head_age < 3 else (0, 0, 255)
                        cv2.putText(debug, head_status, (8, yl + 18),
                                    cv2.FONT_HERSHEY_SIMPLEX, fs * 0.4, head_clr, 1)
            else:
                cv2.putText(debug, "白线: 未检测", (8, int(48 * fs)),
                            cv2.FONT_HERSHEY_SIMPLEX, fs * 0.5, (100, 100, 100), 1)

        # ── 4. ROI 框 + 中线（轻量绘制在主画面上，仅供参考位置） ──
        roi_y = int(h * (1 - LINE_ROI_BOTTOM))
        cv2.rectangle(debug, (0, roi_y), (w, h), (255, 0, 0), 1)
        cv2.line(debug, (w // 2, roi_y), (w // 2, h), (255, 255, 0), 1)

        # ══════ 📷 头部摄像头 PIP 画中画 ══════
        if HEAD_CAM_ENABLED and line_follower is not None:
            # 优先用白线检测的调试帧（有轮廓绘制），没有的话用原始头部画面
            head_debug_img = getattr(line_follower, '_last_debug_img', None)
            if head_debug_img is None:
                # 尝试从头部缓冲区取原始画面
                raw_head = head_img_buffer.get()
                if raw_head is not None:
                    # 缩小 + 打个标题，当作简易 PIP
                    pip_scale = HEAD_CAM_DISPLAY_SCALE
                    pip_w = int(raw_head.shape[1] * pip_scale)
                    pip_h = int(raw_head.shape[0] * pip_scale)
                    pip = cv2.resize(raw_head, (pip_w, pip_h), interpolation=cv2.INTER_NEAREST)
                    cv2.putText(pip, "头部相机(白线检测未就绪)", (4, int(14 * pip_scale)),
                                cv2.FONT_HERSHEY_SIMPLEX, pip_scale * 0.4, (100, 100, 100), 1)
                else:
                    pip = None
            else:
                pip_scale = HEAD_CAM_DISPLAY_SCALE
                pip_w = int(head_debug_img.shape[1] * pip_scale)
                pip_h = int(head_debug_img.shape[0] * pip_scale)
                pip = cv2.resize(head_debug_img, (pip_w, pip_h), interpolation=cv2.INTER_NEAREST)

            if pip is not None:
                # 放在右下角
                margin = 5
                x_off = debug.shape[1] - pip_w - margin
                y_off = debug.shape[0] - pip_h - margin

                # PIP 边框 + 阴影效果
                shaded = cv2.addWeighted(
                    debug[y_off - 1:y_off + pip_h + 1, x_off - 1:x_off + pip_w + 1],
                    0.4, np.zeros_like(debug[y_off - 1:y_off + pip_h + 1, x_off - 1:x_off + pip_w + 1]),
                    0.6, 0
                )
                debug[y_off - 1:y_off + pip_h + 1, x_off - 1:x_off + pip_w + 1] = shaded

                # 叠加头部相机画面
                overlay = debug[y_off:y_off + pip_h, x_off:x_off + pip_w]
                pip_alpha = 0.85
                blended = cv2.addWeighted(overlay, 1 - pip_alpha, pip, pip_alpha, 0)
                debug[y_off:y_off + pip_h, x_off:x_off + pip_w] = blended

                # 白色边框
                cv2.rectangle(debug, (x_off - 1, y_off - 1),
                              (x_off + pip_w + 1, y_off + pip_h + 1), (200, 200, 255), 1)
                # 标签 - 加上头部相机新鲜度标记
                pip_label = "📷 头部相机（白线）"
                head_age = head_img_buffer.get_age()
                if head_age is not None and head_age > 2.0:
                    pip_label += " ⚠️ 回退中"
                cv2.putText(debug, pip_label,
                            (x_off + 3, y_off + int(14 * pip_scale)),
                            cv2.FONT_HERSHEY_SIMPLEX, pip_scale * 0.5, (200, 200, 255), 1)

        # ── 5. 底部图例（每 5 帧画一次） ──
        if _frame % 5 == 1:
            if HEAD_CAM_ENABLED:
                cv2.putText(debug, "主:胸部相机(Tag) | PIP:头部相机(白线) | Q=退出",
                            (8, h - 4), cv2.FONT_HERSHEY_SIMPLEX, fs * 0.35, (170, 170, 170), 1)
            else:
                cv2.putText(debug, "蓝框=ROI | 绿=Tag | 红=未识别 | Q=退出",
                            (8, h - 4), cv2.FONT_HERSHEY_SIMPLEX, fs * 0.35, (170, 170, 170), 1)

        cv2.imshow(win, debug)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            rospy.loginfo("[显示] 用户按 Q 退出程序")
            _combined_debug_state.stop()
            rospy.signal_shutdown("用户按 Q 退出")
            break

    cv2.destroyWindow(win)
    rospy.loginfo("[显示] 独立显示线程退出")


def close_combined_debug():
    """关闭综合调试窗口"""
    _combined_debug_state.stop()


line_follower = LineFollower()


def find_box(img, color_name='green', debug=False):
    """返回 (x, y) 坐标，无面积"""
    if img is None:
        return None, None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blurred = cv2.GaussianBlur(hsv, (3, 3), 0)
    mask = cv2.inRange(blurred, COLOR_RANGE[color_name][0], COLOR_RANGE[color_name][1])
    opened = cv2.dilate(mask, np.ones((4, 4), np.uint8), iterations=2)
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
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

def get_latest_head_image():
    """获取最新的头部摄像头图像，用于白线检测。

    ⚡ 判断流程：
      1. 头部缓存有图且 2 秒内更新过 → 从头部取（重试最多 0.5 秒）
      2. 头部缓存无图或超过 2 秒没更新 → 立即回退到胸部摄像头（不等了）
      3. HEAD_CAM_ENABLED=False → 直接用胸部摄像头
    """
    if not HEAD_CAM_ENABLED:
        return get_latest_image()  # 未启用头部摄像头时回退到胸部摄像头

    # 快速检查：头部缓存最近有无新帧？
    head_age = head_img_buffer.get_age()
    if head_age is None or head_age > 2.0:
        # 头部摄像头可能挂了或还没准备好 → 快速回退到胸部
        rospy.logwarn_throttle(3, f"[头部相机] ⏰ 缓存已{'无图' if head_age is None else f'{head_age:.1f}s无更新'}，快速回退到胸部")
        return get_latest_image()

    # 头部缓存有正常帧 → 走正常取图流程（最多等 0.5 秒）
    for _ in range(5):
        img = head_img_buffer.get()
        if img is not None:
            return img
        time.sleep(0.1)

    # 理论上不会走到这里（age 检查应该确保有图），但以防万一
    rospy.logwarn("[头部相机] ⚠️ 头部缓存有 age 但取不到帧，回退到胸部")
    return get_latest_image()

# ========================== 抱箱函数（前进三次后强制抱箱） ==========================
def approach_and_pick():
    """
    调整阶段，每前进一次计数一次，累计前进3次后立即抱箱。
    """
    rospy.loginfo("开始抱箱流程（前进三次强制抱箱）")
    tracker.reset()

    forward_count = 0

    for adjust_cnt in range(30):
        if forward_count >= MAX_FORWARD_BEFORE_PICK:
            rospy.loginfo(f"已前进 {forward_count} 次，强制进入抱箱")
            break

        img = get_latest_image()
        if img is None:
            rospy.logwarn("获取图像失败")
            break
        x, y = find_box(img, 'green')
        if x is None:
            rospy.logwarn("未检测到箱子，等待图像更新")
            time.sleep(0.3)
            continue

        x_in_deadzone = (X_TARGET_LO - X_DEAD_ZONE <= x <= X_TARGET_HI + X_DEAD_ZONE)
        y_in_target = (Y_TARGET_LO <= y <= Y_TARGET_HI)

        if y_in_target and x_in_deadzone:
            rospy.loginfo(f"坐标达标: x={x:.1f}, y={y:.1f}")
            break

        if y < Y_TARGET_LO:
            rospy.loginfo(f"前进一步 y={y:.1f} (计数 {forward_count+1}/{MAX_FORWARD_BEFORE_PICK})")
            go_fast(1)
            forward_count += 1
        elif x < X_TARGET_LO - X_DEAD_ZONE:
            rospy.loginfo(f"左移一步 x={x:.1f}")
            L_move1(1)
        elif x > X_TARGET_HI + X_DEAD_ZONE:
            rospy.loginfo(f"右移一步 x={x:.1f}")
            R_move1(1)
        elif y > Y_TARGET_HI:
            rospy.loginfo(f"后退一步 y={y:.1f}")
            Back1(1)
        else:
            rospy.loginfo("已达到合理区域，退出调整")
            break

        time.sleep(0.2)
    else:
        rospy.logwarn("调整次数超限，仍尝试抱箱")

    rospy.loginfo("执行抱箱动作")
    ActionExecutor.execute("forward_small", 1, wait=0.3)
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
            img_buffer.update(frame)
        else:
            rospy.logwarn_throttle(5, "等待胸部相机图像")
        time.sleep(0.1)  # ~10fps，机器人动作步长 0.3-1s，完全够用



# ========================== ROS 头部摄像头订阅 ==========================
_head_ros_sub = None      # ROS 头部相机订阅句柄（None=未订阅）
_head_ros_ready = False   # 头部相机 ROS 订阅是否已收到过至少一帧数据


def _decode_ros_image(buf, h, w, channels, step):
    """将 ROS 图像字节流转成 np 数组，兼容 step != w*channels 的行填充。

    step 以 buf 元素个数为单位（uint8 时即字节数）。
    数据不足时抛 ValueError，由调用方兜底丢弃该帧。
    """
    row = w * channels
    if step is None or step == row:
        if buf.size < h * row:
            raise ValueError(f"图像数据不足: {buf.size} < {h * row}")
        flat = buf[:h * row]
        return flat.reshape(h, w, channels) if channels > 1 else flat.reshape(h, w)
    if buf.size < (h - 1) * step + row:
        raise ValueError(f"图像数据不足(带行填充): {buf.size} < {(h - 1) * step + row}")
    rows = np.stack([buf[i * step:i * step + row] for i in range(h)])
    return rows.reshape(h, w, channels) if channels > 1 else rows


def _head_ros_callback(msg):
    """ROS 头部相机 subscriber 回调：将 ROS Image 消息转为 OpenCV 格式存入缓冲区"""
    global _head_ros_ready
    try:
        h, w = msg.height, msg.width
        # msg.step 为每行字节数；uint8 编码时与元素数一致
        step = msg.step if msg.step and msg.step > 0 else None

        if msg.encoding == 'bgr8':
            img = _decode_ros_image(np.frombuffer(msg.data, dtype=np.uint8), h, w, 3, step)
        elif msg.encoding == 'rgb8':
            img = _decode_ros_image(np.frombuffer(msg.data, dtype=np.uint8), h, w, 3, step)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif msg.encoding in ('mono8', '8UC1', 'gray'):
            img = _decode_ros_image(np.frombuffer(msg.data, dtype=np.uint8), h, w, 1, step)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif msg.encoding == 'yuv422':
            img = _decode_ros_image(np.frombuffer(msg.data, dtype=np.uint8), h, w, 2, step)
            img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR_YUYV)
        elif msg.encoding == '16UC1':
            # 16 位深度，step 以字节计，转成 uint16 元素单位
            img_u16 = _decode_ros_image(
                np.frombuffer(msg.data, dtype=np.uint16), h, w, 1,
                step // 2 if step else None
            )
            img = cv2.cvtColor((img_u16 >> 8).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        else:
            # 未知编码，尝试当 bgr8 处理
            rospy.logwarn_throttle(10, f"[头部相机] ⚠ 未知编码 {msg.encoding}，尝试作为 bgr8")
            img = _decode_ros_image(np.frombuffer(msg.data, dtype=np.uint8), h, w, 3, step)
        head_img_buffer.update(img)
        _head_ros_ready = True
    except Exception as e:
        rospy.logwarn_throttle(10, f"[头部相机] ⚠ ROS回调转换异常: {e}")


def find_head_camera_topic():
    """自动检测头部摄像头的 ROS topic

    遍历所有已发布 topic，找到 sensor_msgs/Image 类型的话题。
    如果有多个，排除包含 'chest' 关键词的，返回第一个非胸部 topic。
    如果只有一个，直接使用。
    """
    try:
        topics = rospy.get_published_topics()
        img_topics = [(n, t) for n, t in topics if 'sensor_msgs/Image' in t or '/Image' in t]

        if not img_topics:
            rospy.logwarn("[头部相机] 未发现任何 Image topic")
            return None

        rospy.loginfo(f"[头部相机] 可用 Image topics: {[n for n, t in img_topics]}")

        if len(img_topics) >= 2:
            # 排除胸部 topic
            non_chest = [n for n, t in img_topics if 'chest' not in n.lower()]
            if non_chest:
                return non_chest[0]
            # 没有明确标记 chest/head → 用第二个
            return img_topics[1][0]

        # 只有一个 topic → 试试看
        return img_topics[0][0]
    except Exception as e:
        rospy.logwarn(f"[头部相机] Topic 检测异常: {e}")
        return None


# ========================== 头部图像采集线程 ==========================

def head_image_capture_thread():
    """头部摄像头采集线程

    🎯 优先级策略：
      1. ROS topic 订阅（自动检测 topic 名称）
      2. OpenCV 回退（强制 V4L2 后端）
      3. 线程永不崩溃——找不到就定期重试
    """
    global _head_ros_ready, _head_ros_sub

    if not HEAD_CAM_ENABLED:
        rospy.loginfo("[头部相机] ⏭ 未启用（HEAD_CAM_ENABLED=False），跳过启动")
        return

    # ────────────── Phase 1: ROS topic 订阅 ──────────────
    topic = HEAD_CAM_ROS_TOPIC if HEAD_CAM_ROS_TOPIC else find_head_camera_topic()
    ros_sub_active = False

    if topic:
        _head_ros_ready = False
        try:
            _head_ros_sub = rospy.Subscriber(topic, RosImage, _head_ros_callback, queue_size=1)
            rospy.loginfo(f"[头部相机] 📡 已注册 ROS 订阅: {topic}")

            # 等待最多 3 秒，看是否有数据到达
            wait_start = time.time()
            while not _head_ros_ready and time.time() - wait_start < 3.0:
                rospy.spin_once()
                time.sleep(0.1)

            if _head_ros_ready:
                ros_sub_active = True
                rospy.loginfo(f"[头部相机] ✓ ROS topic {topic} 接收正常")
            else:
                rospy.logwarn(f"[头部相机] ⏳ ROS topic {topic} 无数据（3秒超时），尝试 OpenCV 回退")
                _head_ros_sub.unregister()
                _head_ros_sub = None
        except Exception as e:
            rospy.logwarn(f"[头部相机] ⚠ ROS 订阅异常: {e}")
            if _head_ros_sub is not None:
                _head_ros_sub.unregister()
                _head_ros_sub = None

    if ros_sub_active:
        # ── ROS 模式：用 spin_once() 接收回调 ──
        ros_spin_count = 0
        while not rospy.is_shutdown():
            rospy.spin_once()
            ros_spin_count += 1

            # 画面失联检测：topic 停止发布时 _head_ros_ready 不会自动复位，
            # 这里根据缓存年龄主动回退 OpenCV，避免头部相机永久失效。
            age = head_img_buffer.get_age()
            if age is not None and age > 2.0:
                rospy.logwarn(f"[头部相机] ⚠ ROS topic 数据中断（{age:.1f}s 无新帧），回退 OpenCV")
                _head_ros_ready = False
                if _head_ros_sub is not None:
                    _head_ros_sub.unregister()
                    _head_ros_sub = None
                break

            if ros_spin_count % 30 == 0:
                age_str = f"{age:.1f}s前" if age is not None else "无数据"
                rospy.loginfo(f"[头部相机] ✓ ROS 采集正常（{age_str}）")
            time.sleep(1.0 / HEAD_CAPTURE_FPS)
        # ROS 订阅断了的时候，fall through 到下面重试
        ros_sub_active = False

    # ────────────── Phase 2: OpenCV 回退 ──────────────
    head_cap = None
    opened_index = -1

    # 用 V4L2 后端重试（不用 GStreamer，避免设备占用冲突）
    for idx in list(range(6)) + list(range(10, 20)):  # 尝试 0~5 + 10~19
        try:
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)  # ← 强制 V4L2 后端
            if cap.isOpened():
                ret, test = cap.read()
                if ret and test is not None:
                    head_cap = cap
                    opened_index = idx
                    head_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    head_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    rospy.loginfo(f"[头部相机] ✓ 通过 OpenCV(CAP_V4L2) 打开设备 {idx}")
                    break
                else:
                    cap.release()
            else:
                cap.release()
        except Exception as e:
            rospy.logwarn_throttle(30, f"[头部相机] ⚠ 打开设备 {idx} 异常: {e}")

    if head_cap is None:
        rospy.logwarn("[头部相机] ⏳ OpenCV 未找到头部摄像头，将定期重试")

    _capture_count = 0
    _retry_counter = 0
    while not rospy.is_shutdown():
        frame = None

        if head_cap is not None:
            try:
                ret, frame = head_cap.read()
                if not ret or frame is None:
                    rospy.logwarn_throttle(5, "[头部相机] ⏳ OpenCV 读取失败")
                    frame = None
            except Exception as e:
                rospy.logwarn_throttle(5, f"[头部相机] ⚠ OpenCV 读取异常: {e}")
                frame = None

        if frame is not None:
            head_img_buffer.update(frame)
            _capture_count += 1
            _retry_counter = 0
            if _capture_count % 30 == 0:
                cam_label = f"OpenCV(设备{opened_index})" if opened_index >= 0 else "未知"
                rospy.loginfo(f"[头部相机] ✓ 已采集 {_capture_count} 帧（{cam_label}）")
        else:
            _retry_counter += 1
            if head_cap is None and _retry_counter >= 30:
                _retry_counter = 0
                rospy.loginfo("[头部相机] ♻ 重试打开摄像头...")
                for idx in list(range(6)) + list(range(10, 20)):
                    try:
                        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
                        if cap.isOpened():
                            ret, test = cap.read()
                            if ret and test is not None:
                                head_cap = cap
                                opened_index = idx
                                head_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                                head_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                                rospy.loginfo(f"[头部相机] ✓ 重试成功，打开设备 {idx}")
                                break
                            cap.release()
                    except Exception as e:
                        rospy.logwarn_throttle(30, f"[头部相机] ⚠ 重试打开设备 {idx} 异常: {e}")
                if head_cap is None:
                    rospy.logwarn("[头部相机] ♻ 重试仍未找到头部摄像头")

        time.sleep(1.0 / HEAD_CAPTURE_FPS)

    if head_cap is not None:
        head_cap.release()
        rospy.loginfo("[头部相机] 🔒 已释放 OpenCV 摄像头")
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
                rospy.loginfo("一号 Tag 对正完成，沿白线前进至二号")
                state.tag_id = 2
                intr = line_follower.move_forward(5, 'box', check_tag_ids=[2, 3, 4, 5])
                if intr is not None:
                    rospy.loginfo(f"[Tag] ⏭ 白线被 Tag #{intr} 中断，直接跳转")
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
                rospy.loginfo("二号 Tag 对正完成，白线引导前进至三号")
                state.tag_id = 3
                intr = line_follower.move_forward(3, 'box', check_tag_ids=[3, 4, 5])
                if intr is not None:
                    rospy.loginfo(f"[Tag] ⏭ 白线被 Tag #{intr} 中断")
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
                intr = line_follower.move_forward(12, 'box', check_tag_ids=[4, 5])
                if intr is not None:
                    rospy.loginfo(f"[Tag] ⏭ 白线被 Tag #{intr} 中断，跳过后续动作")
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
                rospy.loginfo("五号 Tag 对正完成，沿白线前进放箱")
                # ⚡ 对正后不要再监听 Tag #5（否则刚对正又看到 Tag #5 → 无限循环），
                # 只监听后续 Tag #6（还没到放箱位就看到 #6 → 原地放箱再继续）
                intr = line_follower.move_forward(3, 'box', check_tag_ids=[6])
                if intr is not None:
                    rospy.loginfo(f"[Tag] ⏭ 白线被 Tag #{intr} 中断，原地放箱")
                    Box_Down(1)
                    state.has_box = False
                    state.step = Step.INBOUND
                    state.tag_id = intr
                    return Flow.CONTINUE
                Box_Down(1)
                state.has_box = False
                rospy.loginfo("放下箱子，空载沿白线前进并转向")
                intr = line_follower.move_forward(3, 'empty', check_tag_ids=[6])
                if intr is not None:
                    rospy.loginfo(f"[Tag] ⏭ 空载白线被 Tag #{intr} 中断")
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
            rospy.loginfo("五号反向对正完成，沿白线移至六号")
            state.tag_id = 6
            intr = line_follower.move_forward(3, 'empty', check_tag_ids=[6])
            if intr is not None:
                rospy.loginfo(f"[Tag] ⏭ 白线被 Tag #{intr} 中断")
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
                rospy.loginfo("六号对正完成，左移至七号（白线引导前进）")
                state.tag_id = 7
                L_move2(6)
                L_turn2(1)
                intr = line_follower.move_forward(5, 'empty', check_tag_ids=[7])
                if intr is not None:
                    rospy.loginfo(f"[Tag] ⏭ 白线被 Tag #{intr} 中断")
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
            rospy.loginfo("一号反向对正完成，沿白线前进抓块")
            line_follower.move_forward(4, 'empty')
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

    # ── 启动头部摄像头采集线程（双摄像头模式） ──
    if HEAD_CAM_ENABLED:
        head_th = threading.Thread(target=head_image_capture_thread, daemon=True)
        head_th.start()
        rospy.loginfo(f"[头部相机] 📷 头部相机采集线程已启动（{HEAD_CAPTURE_FPS}fps）")
    else:
        rospy.loginfo("[头部相机] ⏭ 未启用，白线检测将使用胸部相机")

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
    try:
        result = subprocess.run(
            ['v4l2-ctl', '--list-devices'],
            capture_output=True, text=True, timeout=3
        )
        out = result.stdout.strip()
        if out:
            rospy.loginfo(f"[诊断] V4L2 设备:\n{out}")
        else:
            rospy.loginfo("[诊断] V4L2 设备: (无)")
    except Exception as e:
        rospy.loginfo(f"[诊断] v4l2-ctl 不可用: {e}")
    try:
        result2 = subprocess.run(
            ['ls', '-la', '/dev/video*'],
            capture_output=True, text=True, timeout=3
        )
        out2 = result2.stdout.strip()
        if out2:
            rospy.loginfo(f"[诊断] /dev/video* 设备列表:\n{out2}")
    except Exception as e:
        pass
    rospy.loginfo("=" * 45)

    # 等待胸部相机图像就绪（带超时，避免相机异常时无限刷屏）
    chest_wait_start = time.time()
    while img_buffer.get() is None:
        if time.time() - chest_wait_start > 10.0:
            rospy.logwarn("[启动] ⏰ 等待胸部相机超时(10s)，继续启动")
            break
        rospy.loginfo_throttle(2, "等待胸部相机图像数据...")
        time.sleep(0.2)

    # 等待头部相机图像就绪（如果启用）
    if HEAD_CAM_ENABLED:
        wait_start = time.time()
        head_ready = False
        while head_img_buffer.get() is None:
            if time.time() - wait_start > 5.0:
                rospy.logwarn("[头部相机] ⏰ 等待头部相机超时，继续启动…")
                break
            rospy.loginfo_throttle(2, "等待头部相机图像数据...")
            time.sleep(0.2)
        else:
            head_ready = True
        if head_ready:
            rospy.loginfo("[头部相机] ✓ 头部相机图像已就绪")
        else:
            rospy.logwarn("[头部相机] ⚠️ 头部相机未就绪，将回退到胸部相机进行白线检测")

    # ── 启动独立显示线程：画面从此稳定刷新 ──
    display_th = threading.Thread(target=display_thread_func, daemon=True)
    display_th.start()
    rospy.loginfo("[显示] 独立显示线程已启动（双摄像头模式）")

    state.reset()

    rospy.loginfo("启动完成，前进两步")
    go_fast2(2)

    rate = rospy.Rate(10)
    no_tag_count = 0

    while not rospy.is_shutdown():
        try:
            img = img_buffer.get()
            if img is None:
                rate.sleep()
                continue

            if state.tag_id == 0:
                # 综合调试：装箱阶段无 Tag
                show_combined_debug(img, None)

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
                # ── 📷 始终运行头部摄像头白线检测 ──
                # 不管有没有 Tag，两个摄像头始终同时工作。白线数据每帧刷新，随时可用
                head_img = get_latest_head_image()
                white_line_ok = False
                if head_img is not None:
                    offset = line_follower.detect(head_img)
                    if offset is not None:
                        white_line_ok = True

                markers = tag_reader.get_nearest_marker()
                if len(markers) < 4:
                    # 综合调试：显示无 Tag 状态
                    show_combined_debug(img, None)
                    line_follower._tag_interrupted = False  # 没有 Tag 时清空中断标记

                    # ── 无 Tag → 用白线循迹控制 ──
                    if white_line_ok:
                        no_tag_count = 0  # 白线在工作，不算"完全丢失"
                        rospy.loginfo_throttle(2, f"[优先级] 无Tag→白线引导 (ID={state.tag_id}, step={state.step.value}, phase={state.phase.value})")

                        step_mode = state.step_mode

                        # ⛰️ 直角拐弯优先
                        if line_follower.corner_state != 0 and not line_follower.corner_handled:
                            line_follower.handle_corner(step_mode)
                        else:
                            # 横向纠偏（保持居中）
                            line_follower.correct(step_mode)
                            # 角度纠偏（平滑弯道）
                            if abs(line_follower.angle_dev) >= LINE_ANGLE_DEADZONE and line_follower.corner_state == 0:
                                line_follower.correct_angle(step_mode)

                        # 前进一步
                        if step_mode == 'box':
                            box_go(1)
                        else:
                            go_fast(1)

                        rate.sleep()
                        continue

                    # ── 白线也没有 → 原备用逻辑（盲目前进/转向） ──
                    no_tag_count += 1
                    rospy.loginfo_throttle(2, f"无Tag+无白线 (连续{no_tag_count}次)，盲目前进, ID={state.tag_id}, phase={state.phase.value}, step={state.step.value}")
                    if no_tag_count > NO_TAG_TURN_THRESH:
                        rospy.logwarn("连续无Tag无白线次数过多，执行转向")
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

                    # —— 综合调试：每帧显示 Tag + 白线状态 ——
                    show_combined_debug(img, markers)

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

                    # ⚡ 白线严重偏移 → 先纠偏再处理 Tag
                    # 即使检测到 Tag，如果白线偏移超过 CRITICAL 阈值，
                    # 说明机器人严重偏离轨迹（白线在画面边缘），
                    # 先拉回白线再处理 Tag，防止跑到一半偏出赛道
                    if white_line_ok:
                        critical_offset = abs(line_follower.offset_smooth)
                        if critical_offset >= LINE_OFFSET_CRITICAL:
                            _sm = state.step_mode
                            rospy.logwarn(
                                f"[优先级] ⚠️ 有Tag#{tag_id} 但白线严重偏移 "
                                f"({line_follower.offset_smooth:+.0f}px ≥ {LINE_OFFSET_CRITICAL}) → 先纠偏再处理Tag"
                            )
                            line_follower.correct(_sm)
                            if (abs(line_follower.angle_dev) >= LINE_ANGLE_DEADZONE
                                    and line_follower.corner_state == 0):
                                line_follower.correct_angle(_sm)
                            # ⏭ 纠偏后继续处理 Tag（Tag 可能还在视野内）
                            # 即使 Tag 因为车身转动暂时丢失，下个主循环会重新检测
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
