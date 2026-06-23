import cv2
import serial
import numpy as np
import lgpio
import sys, os
import time
import threading
import re
from collections import deque

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from ultralytics import YOLO

# -----------------------------
# THERMAL SENSOR IMPORTS
# -----------------------------
try:
    import board
    import busio
    import adafruit_mlx90640
    THERMAL_AVAILABLE = True
    print("✅ Thermal library loaded")
except ImportError:
    THERMAL_AVAILABLE = False
    print("⚠️ Thermal library not available")

# -----------------------------
# CONFIG
# -----------------------------
BRIGHTNESS_TH        = 40
CONTRAST_TH          = 15
MAX_RADAR_POINTS     = 50

# RADAR ON UART0
RADAR_PORT           = '/dev/ttyAMA0'
BAUD_RATE            = 115200
RADAR_MIN_DIST       = 10
RADAR_MAX_DIST       = 600
RADAR_DANGER_DIST    = 150

# BUZZER PINS
BUZZER_PIN           = 18   # GPIO18 (Pin 12)
OPTIONAL_PIN         = 17   # GPIO17 (Pin 11)

# LCD PINS (Parallel LCD)
LCD_RS               = 5    # GPIO5 (Pin 29)
LCD_E                = 6    # GPIO6 (Pin 31)
LCD_D4               = 13   # GPIO13 (Pin 33)
LCD_D5               = 19   # GPIO19 (Pin 35)
LCD_D6               = 26   # GPIO26 (Pin 37)
LCD_D7               = 20   # GPIO20 (Pin 38)
LCD_WIDTH            = 16

# LCD Commands
LCD_LINE_1           = 0x80
LCD_LINE_2           = 0xC0

# THERMAL CONFIG
TEMP_WARNING         = 33.0
TEMP_DANGER          = 38.0
TEMP_ALERT_THRESHOLD = 36.0
THERMAL_REFRESH_HZ   = 4

# FUSION CONFIG
THREAT_THRESHOLD     = 0.45
WEIGHT_SMOOTH        = 0.25
CAM_WEIGHT_MIN       = 0.15
CAM_WEIGHT_MAX       = 0.85

# -----------------------------
# GPIO SETUP
# -----------------------------
chip = lgpio.gpiochip_open(0)

# Buzzer pins
lgpio.gpio_claim_output(chip, BUZZER_PIN)
lgpio.gpio_claim_output(chip, OPTIONAL_PIN)
lgpio.gpio_write(chip, BUZZER_PIN, 0)
lgpio.gpio_write(chip, OPTIONAL_PIN, 0)
print(f"✅ GPIO initialized - Buzzer on GPIO{BUZZER_PIN}")

# LCD pins
lgpio.gpio_claim_output(chip, LCD_RS)
lgpio.gpio_claim_output(chip, LCD_E)
lgpio.gpio_claim_output(chip, LCD_D4)
lgpio.gpio_claim_output(chip, LCD_D5)
lgpio.gpio_claim_output(chip, LCD_D6)
lgpio.gpio_claim_output(chip, LCD_D7)

# Initialize LCD pins to LOW
for pin in [LCD_RS, LCD_E, LCD_D4, LCD_D5, LCD_D6, LCD_D7]:
    lgpio.gpio_write(chip, pin, 0)

# -----------------------------
# LCD FUNCTIONS
# -----------------------------

def lcd_byte(bits, mode):
    """Send byte to LCD"""
    lgpio.gpio_write(chip, LCD_RS, 1 if mode else 0)
    
    # High nibble
    lgpio.gpio_write(chip, LCD_D4, 1 if (bits >> 4) & 1 else 0)
    lgpio.gpio_write(chip, LCD_D5, 1 if (bits >> 5) & 1 else 0)
    lgpio.gpio_write(chip, LCD_D6, 1 if (bits >> 6) & 1 else 0)
    lgpio.gpio_write(chip, LCD_D7, 1 if (bits >> 7) & 1 else 0)
    lgpio.gpio_write(chip, LCD_E, 1)
    time.sleep(0.0005)
    lgpio.gpio_write(chip, LCD_E, 0)
    time.sleep(0.0005)
    
    # Low nibble
    lgpio.gpio_write(chip, LCD_D4, 1 if bits & 1 else 0)
    lgpio.gpio_write(chip, LCD_D5, 1 if (bits >> 1) & 1 else 0)
    lgpio.gpio_write(chip, LCD_D6, 1 if (bits >> 2) & 1 else 0)
    lgpio.gpio_write(chip, LCD_D7, 1 if (bits >> 3) & 1 else 0)
    lgpio.gpio_write(chip, LCD_E, 1)
    time.sleep(0.0005)
    lgpio.gpio_write(chip, LCD_E, 0)
    time.sleep(0.0005)

def lcd_init():
    """Initialize LCD"""
    time.sleep(0.1)
    lcd_byte(0x33, False)
    lcd_byte(0x32, False)
    lcd_byte(0x28, False)  # 4-bit mode, 2 lines
    lcd_byte(0x0C, False)  # Display on, cursor off
    lcd_byte(0x06, False)  # Entry mode
    lcd_byte(0x01, False)  # Clear display
    time.sleep(0.1)

def lcd_display(line1="", line2=""):
    """Display two lines on LCD"""
    lcd_byte(LCD_LINE_1, False)
    for char in line1[:LCD_WIDTH]:
        lcd_byte(ord(char), True)
    
    if line2:
        lcd_byte(LCD_LINE_2, False)
        for char in line2[:LCD_WIDTH]:
            lcd_byte(ord(char), True)

def lcd_clear():
    """Clear LCD"""
    lcd_byte(0x01, False)
    time.sleep(0.1)

# Initialize LCD
lcd_init()
lcd_display("SafeVision", "Initializing...")
print("✅ LCD initialized")

# -----------------------------
# LOAD YOLO MODEL
# -----------------------------
try:
    model = YOLO(os.path.join(BASE_DIR, "best1.pt"))
    print("✅ YOLO model loaded")
except:
    print("⚠️ YOLO model not found")
    model = None

# -----------------------------
# CAMERA INIT
# -----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Camera not initialized")
    lgpio.gpiochip_close(chip)
    exit(1)

# -----------------------------
# RADAR INIT
# -----------------------------
ser = None
try:
    ser = serial.Serial(RADAR_PORT, BAUD_RATE, timeout=0.5)
    print(f"✅ Radar connected on {RADAR_PORT}")
except serial.SerialException as e:
    print(f"⚠️ Radar not found on {RADAR_PORT}: {e}")

if ser is None:
    print("WARNING: No radar connected")

radar_data_deque = deque(maxlen=MAX_RADAR_POINTS)

# Store last values for display
last_radar_distance = None
last_thermal_temp = 25.0
lcd_update_counter = 0

# =====================================================================
# THERMAL SENSOR CLASS
# =====================================================================

class ThermalSensor:
    def __init__(self):
        self.mlx = None
        self.latest_frame = None
        self.latest_stats = {
            'min': 25.0,
            'max': 25.0,
            'avg': 25.0,
            'threat': 0.0
        }
        self.frame_lock = threading.Lock()
        self.running = False
        self.thread = None
        self.available = False
        
    def start(self):
        if not THERMAL_AVAILABLE:
            print("⚠️ Thermal library not available")
            return False
            
        try:
            i2c = busio.I2C(board.SCL, board.SDA)
            self.mlx = adafruit_mlx90640.MLX90640(i2c)
            
            refresh_map = {
                2: adafruit_mlx90640.RefreshRate.REFRESH_2_HZ,
                4: adafruit_mlx90640.RefreshRate.REFRESH_4_HZ,
                8: adafruit_mlx90640.RefreshRate.REFRESH_8_HZ,
                16: adafruit_mlx90640.RefreshRate.REFRESH_16_HZ
            }
            self.mlx.refresh_rate = refresh_map.get(THERMAL_REFRESH_HZ, adafruit_mlx90640.RefreshRate.REFRESH_4_HZ)
            
            print(f"✅ Thermal sensor ready")
            self.available = True
            self.running = True
            self.thread = threading.Thread(target=self._update_loop, daemon=True)
            self.thread.start()
            return True
        except Exception as e:
            print(f"⚠️ Thermal sensor failed: {e}")
            return False
    
    def _update_loop(self):
        frame = [0] * 768
        while self.running:
            try:
                self.mlx.getFrame(frame)
                temps = np.array(frame).reshape(24, 32)
                temps = np.clip(temps, -20, 100)
                
                with self.frame_lock:
                    self.latest_frame = temps.copy()
                    valid = temps.flatten()
                    max_temp = np.max(valid)
                    self.latest_stats = {
                        'min': float(np.min(valid)),
                        'max': float(max_temp),
                        'avg': float(np.mean(valid)),
                        'threat': min(1.0, max(0, (max_temp - TEMP_WARNING) / (TEMP_DANGER - TEMP_WARNING)))
                    }
            except Exception:
                pass
            time.sleep(0.05)
    
    def get_frame(self):
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None
    
    def get_stats(self):
        with self.frame_lock:
            return self.latest_stats.copy()
    
    def get_threat(self):
        return self.get_stats()['threat']
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)

# =====================================================================
# THERMAL VISUALIZATION
# =====================================================================

def temperature_to_color(temp_celsius):
    temp_min = 20.0
    temp_max = 35.0
    
    t = max(temp_min, min(temp_max, temp_celsius))
    normalized = (t - temp_min) / (temp_max - temp_min)
    
    if normalized < 0.25:
        return (0, int(255 * (normalized * 4)), 255)
    elif normalized < 0.5:
        return (0, 255, int(255 * (1 - (normalized - 0.25) * 4)))
    elif normalized < 0.75:
        return (int(255 * ((normalized - 0.5) * 4)), 255, 0)
    else:
        return (255, int(255 * (1 - (normalized - 0.75) * 4)), 0)

def create_thermal_overlay(frame, thermal_frame, alpha=0.3):
    if thermal_frame is None:
        return frame
    
    h, w = frame.shape[:2]
    thermal_resized = cv2.resize(thermal_frame, (w, h), interpolation=cv2.INTER_NEAREST)
    temp_min, temp_max = 20.0, 35.0
    normalized = np.clip((thermal_resized - temp_min) / (temp_max - temp_min), 0, 1)
    thermal_uint8 = (normalized * 255).astype(np.uint8)
    thermal_color = cv2.applyColorMap(thermal_uint8, cv2.COLORMAP_JET)
    return cv2.addWeighted(frame, 1 - alpha, thermal_color, alpha, 0)

def draw_mini_thermal(frame, thermal_frame, x, y):
    if thermal_frame is None:
        return frame
    
    h, w = thermal_frame.shape
    cell_w, cell_h = 12, 10
    rows, cols = min(6, h), min(10, w)
    step_r, step_c = h // rows, w // cols
    
    for i in range(rows):
        for j in range(cols):
            temp = thermal_frame[i * step_r, j * step_c]
            color = temperature_to_color(temp)
            cv2.rectangle(frame, (x + j * cell_w, y + i * cell_h), 
                         (x + (j+1) * cell_w, y + (i+1) * cell_h), color, -1)
    
    cv2.rectangle(frame, (x, y), (x + cols * cell_w, y + rows * cell_h), (255, 255, 255), 1)
    cv2.putText(frame, "THERMAL", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return frame

# =====================================================================
# DARK CHANNEL PRIOR DEHAZING
# =====================================================================

def dark_channel(img, patch_size=15):
    min_channel = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    return cv2.erode(min_channel, kernel)

def estimate_atmosphere(img, dark):
    h, w = dark.shape
    n_search = max(1, int(h * w * 0.001))
    flat_dark = dark.flatten()
    flat_img = img.reshape(h * w, 3)
    indices = np.argsort(flat_dark)[-n_search:]
    A = np.max(flat_img[indices], axis=0)
    return A

def dehaze_frame(frame, omega=0.95, patch_size=15):
    try:
        img = frame.astype(np.float32) / 255.0
        dark = dark_channel(img, patch_size)
        A = estimate_atmosphere(img, dark)
        norm = img / (A + 1e-6)
        dc = dark_channel(norm, patch_size)
        t = 1.0 - omega * dc
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        try:
            t = cv2.ximgproc.guidedFilter(guide=gray, src=t, radius=40, eps=1e-3)
        except AttributeError:
            t = cv2.bilateralFilter(t.astype(np.float32), 9, 75, 75)
        t = np.maximum(t, 0.1)[:, :, np.newaxis]
        J = (img - A) / t + A
        return (np.clip(J, 0, 1) * 255).astype(np.uint8)
    except Exception as e:
        print(f"Dehaze error: {e}")
        return frame

# =====================================================================
# FOG SEVERITY
# =====================================================================

def fog_severity(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    contrast_score = max(0.0, 1.0 - (gray.std() / (CONTRAST_TH * 2.5)))
    brightness_score = max(0.0, min(1.0, (gray.mean() - 150) / 80.0))
    return float(np.clip(contrast_score * 0.65 + brightness_score * 0.35, 0.0, 1.0))

# =====================================================================
# CONFIDENCE SCORES
# =====================================================================

def camera_confidence(frame, detections, fog_sev):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness_ok = min(1.0, gray.mean() / BRIGHTNESS_TH)
    contrast_ok = min(1.0, gray.std() / CONTRAST_TH)
    image_quality = (brightness_ok + contrast_ok) / 2.0
    fog_penalty = 1.0 - (fog_sev * 0.75)

    if detections is not None and len(detections) > 0:
        det_quality = float(np.mean(detections.conf.cpu().numpy()))
        score = image_quality * 0.30 + det_quality * 0.45 + fog_penalty * 0.25
    else:
        score = image_quality * 0.55 + fog_penalty * 0.45
    return float(np.clip(score, 0.0, 1.0))

def radar_confidence(valid_radar):
    if not valid_radar:
        return 0.0
    count_score = min(1.0, len(valid_radar) / 3.0)
    proximity_score = max(0.0, 1.0 - (min(valid_radar) / RADAR_MAX_DIST))
    return float(np.clip(count_score * 0.4 + proximity_score * 0.6, 0.0, 1.0))

def thermal_confidence(thermal_stats):
    if thermal_stats is None:
        return 0.0
    if 15 < thermal_stats['max'] < 50:
        return 0.8
    return 0.3

# =====================================================================
# THREAT SCORES
# =====================================================================

def radar_threat(valid_radar):
    if not valid_radar:
        return 0.0
    closest = min(valid_radar)
    if closest <= RADAR_DANGER_DIST:
        return float(np.clip(1.0 - (closest / RADAR_DANGER_DIST), 0.0, 1.0))
    return 0.0

def camera_threat(detections):
    if detections is None or len(detections) == 0:
        return 0.0
    confs = detections.conf.cpu().numpy()
    return float(np.clip(np.mean(confs) * 0.7 + min(1.0, len(confs) / 5.0) * 0.3, 0.0, 1.0))

def thermal_threat(thermal_stats):
    if thermal_stats is None:
        return 0.0
    return thermal_stats['threat']

# =====================================================================
# DYNAMIC WEIGHT CALCULATOR
# =====================================================================

def compute_dynamic_weights_3sensors(cam_conf, radar_conf, thermal_conf, fog_sev, prev_weights):
    total = cam_conf + radar_conf + thermal_conf
    if total < 1e-6:
        raw = [0.34, 0.33, 0.33]
    else:
        raw = [cam_conf / total, radar_conf / total, thermal_conf / total]
    
    raw[0] -= fog_sev * (raw[0] - CAM_WEIGHT_MIN)
    raw[0] = float(np.clip(raw[0], 0.1, 0.7))
    raw[1] = float(np.clip(raw[1], 0.1, 0.6))
    raw[2] = float(np.clip(raw[2], 0.1, 0.5))
    
    total_raw = sum(raw)
    if total_raw > 0:
        raw = [w / total_raw for w in raw]
    
    smooth = [
        WEIGHT_SMOOTH * prev_weights[0] + (1 - WEIGHT_SMOOTH) * raw[0],
        WEIGHT_SMOOTH * prev_weights[1] + (1 - WEIGHT_SMOOTH) * raw[1],
        WEIGHT_SMOOTH * prev_weights[2] + (1 - WEIGHT_SMOOTH) * raw[2]
    ]
    
    total_smooth = sum(smooth)
    if total_smooth > 0:
        smooth = [w / total_smooth for w in smooth]
    
    return smooth[0], smooth[1], smooth[2]

# =====================================================================
# STATE
# =====================================================================
prev_weights = [0.5, 0.3, 0.2]
dehaze_enabled = True
show_thermal_overlay = True
show_thermal_grid = True

# Initialize thermal sensor
thermal_sensor = ThermalSensor()
thermal_available = thermal_sensor.start()

# =====================================================================
# MAIN LOOP
# =====================================================================
print("\n" + "="*50)
print("🚀 SafeVision Fusion Started (LCD + Buzzer)")
print("="*50)
print(f"📡 Radar Port: {RADAR_PORT}")
print(f"🌡️ Thermal: {'Available' if thermal_available else 'Not available'}")
print(f"🔔 Buzzer on GPIO{BUZZER_PIN}")
print(f"🖥️ LCD: Connected")
print("\n⚠️ Alert Conditions:")
print("   - Radar distance < 150cm → Buzzer ON + LCD Alert")
print("   - Thermal temperature > 32°C → Buzzer ON + LCD Alert")
print("   - Fused threat > 45% → Buzzer ON")
print("\nControls:")
print("  'q' - Quit")
print("  'h' - Toggle dehaze")
print("  't' - Toggle thermal overlay")
print("  'g' - Toggle thermal grid")
print("="*50 + "\n")

try:
    while True:
        ret, frame = cap.read()

        # Camera pipeline
        if not ret:
            frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
            results = None
            detections = None
            fog_sev_v = 1.0
            cam_conf = 0.0
        else:
            fog_sev_v = fog_severity(frame)
            if dehaze_enabled and fog_sev_v > 0.2:
                frame = dehaze_frame(frame)
                fog_sev_v = fog_severity(frame) * 0.5
            
            if model:
                results = model(frame, imgsz=256, conf=0.3, verbose=False)
                detections = results[0].boxes if results else None
            cam_conf = camera_confidence(frame, detections, fog_sev_v)

        # ── RADAR PIPELINE ─────────────────────────────────────────
        radar_values = []
        if ser:
            try:
                if ser.in_waiting:
                    line = ser.readline().decode(errors="ignore").strip()
                    if line:
                        numbers = re.findall(r'\d+\.?\d*', line)
                        for num in numbers:
                            try:
                                dist = float(num)
                                if RADAR_MIN_DIST <= dist <= RADAR_MAX_DIST:
                                    radar_values.append(dist)
                            except:
                                pass
                        
                        if radar_values:
                            radar_data_deque.append(radar_values)
                            last_radar_distance = radar_values[0]
            except Exception as e:
                pass

        valid_radar = [d for d in radar_values if RADAR_MIN_DIST <= d <= RADAR_MAX_DIST]
        radar_conf = radar_confidence(valid_radar)
        
        # Use last known distance for display if no new data
        display_distance = valid_radar[0] if valid_radar else last_radar_distance

        # Thermal pipeline
        thermal_stats = thermal_sensor.get_stats() if thermal_available else None
        thermal_conf = thermal_confidence(thermal_stats) if thermal_available else 0.0
        
        if thermal_stats:
            last_thermal_temp = thermal_stats['max']

        # Dynamic weighting
        cam_w, radar_w, thermal_w = compute_dynamic_weights_3sensors(
            cam_conf, radar_conf, thermal_conf, fog_sev_v, prev_weights
        )
        prev_weights = [cam_w, radar_w, thermal_w]

        # Threat scores
        r_threat = radar_threat(valid_radar)
        c_threat = camera_threat(detections) if detections is not None else 0.0
        t_threat = thermal_threat(thermal_stats)

        fused_threat = float(np.clip(
            c_threat * cam_w + r_threat * radar_w + t_threat * thermal_w,
            0.0, 1.0
        ))

        # ========== BUZZER + LCD ALERT CONTROL ==========
        current_distance = min(valid_radar) if valid_radar else (last_radar_distance if last_radar_distance else 999)
        current_temp = thermal_stats['max'] if thermal_stats else last_thermal_temp
        
        buzzer_on = False
        alert_type = "NONE"
        alert_message = ""
        
        # Check radar alert (distance < 150cm)
        if current_distance < RADAR_DANGER_DIST:
            buzzer_on = True
            alert_type = "RADAR"
            alert_message = f"RADAR: {current_distance:.0f}cm"
            print(f"🔔 RADAR ALERT: {current_distance:.0f}cm!")
        
        # Check thermal alert (temperature > 32°C)
        elif current_temp > TEMP_ALERT_THRESHOLD:
            buzzer_on = True
            alert_type = "THERMAL"
            alert_message = f"TEMP: {current_temp:.1f}C"
            print(f"🔔 THERMAL ALERT: {current_temp:.1f}C!")
        
        # Check fused threat
        elif fused_threat >= THREAT_THRESHOLD:
            buzzer_on = True
            alert_type = "THREAT"
            alert_message = f"THREAT: {fused_threat*100:.0f}%"
            print(f"⚠️ THREAT ALERT: {fused_threat*100:.0f}%")
        
        else:
            buzzer_on = False
        
        # Apply buzzer state
        lgpio.gpio_write(chip, BUZZER_PIN, 1 if buzzer_on else 0)
        
        # Update LCD Display (every few frames to reduce flicker)
        lcd_update_counter += 1
        if lcd_update_counter >= 5:
            lcd_update_counter = 0
            
            if alert_type == "RADAR":
                # Display Radar Alert on LCD
                lcd_clear()
                lcd_display("⚠️ RADAR ALERT!", f"Dist: {current_distance:.0f}cm")
                
            elif alert_type == "THERMAL":
                # Display Thermal Alert on LCD
                lcd_clear()
                lcd_display("⚠️ BODY HEAT!", f"Temp: {current_temp:.1f}C")
                
            elif alert_type == "THREAT":
                # Display Threat Alert on LCD
                lcd_clear()
                lcd_display("⚠️ THREAT!", f"{fused_threat*100:.0f}%")
                
            else:
                # Normal display - show sensor data
                # Line 1: Radar distance
                if display_distance:
                    radar_text = f"R:{display_distance:.0f}cm"
                else:
                    radar_text = "R:--cm"
                
                # Line 2: Thermal temperature
                temp_text = f"T:{current_temp:.1f}C"
                
                # Color code on display
                if current_distance < RADAR_DANGER_DIST:
                    radar_text = f"!{radar_text}!"
                if current_temp > TEMP_ALERT_THRESHOLD:
                    temp_text = f"!{temp_text}!"
                
                lcd_clear()
                lcd_display(radar_text, temp_text)
        # ================================================

        # Thermal visualization
        if thermal_available:
            thermal_frame = thermal_sensor.get_frame()
            if show_thermal_overlay and thermal_frame is not None:
                frame = create_thermal_overlay(frame, thermal_frame, alpha=0.3)
            if show_thermal_grid and thermal_frame is not None:
                frame = draw_mini_thermal(frame, thermal_frame, x=frame.shape[1]-130, y=50)

        # Display
        if results is not None and results[0] is not None:
            annotated = results[0].plot()
        else:
            annotated = frame.copy()
            
        h, w = annotated.shape[:2]

        # Threat bar
        BX, BY, BW, BH = 20, h - 45, 300, 22
        if fused_threat < 0.3:
            bar_color = (0, 255, 0)
        elif fused_threat < 0.6:
            bar_color = (0, 165, 255)
        else:
            bar_color = (0, 0, 255)
            
        cv2.rectangle(annotated, (BX, BY), (BX+BW, BY+BH), (50,50,50), -1)
        cv2.rectangle(annotated, (BX, BY), (BX+int(BW*fused_threat), BY+BH), bar_color, -1)
        cv2.rectangle(annotated, (BX, BY), (BX+BW, BY+BH), (200,200,200), 1)
        cv2.putText(annotated, f"THREAT: {fused_threat*100:.0f}%",
                    (BX+BW+8, BY+16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bar_color, 2)

        # Sensor info
        if cam_w >= 0.45:
            active_sensor, sensor_color = "CAMERA DOMINANT", (0, 255, 0)
        elif radar_w >= 0.35:
            active_sensor, sensor_color = "RADAR DOMINANT", (0, 165, 255)
        else:
            active_sensor, sensor_color = "THERMAL DOMINANT", (255, 0, 0)
        
        cv2.putText(annotated, active_sensor, (20,35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, sensor_color, 2)
        cv2.putText(annotated, f"CAM {cam_w*100:.0f}% | RADAR {radar_w*100:.0f}% | THERMAL {thermal_w*100:.0f}%",
                    (20,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # Thermal temp display
        if current_temp >= TEMP_ALERT_THRESHOLD:
            temp_color = (0, 0, 255)
            temp_status = "⚠️ BODY HEAT!"
        elif current_temp > TEMP_WARNING:
            temp_color = (0, 165, 255)
            temp_status = "⚠️ Warm"
        else:
            temp_color = (0, 255, 0)
            temp_status = "Normal"
        
        cv2.putText(annotated, f"🌡️ THERMAL: {current_temp:.1f}C ({temp_status})", 
                   (20,85), cv2.FONT_HERSHEY_SIMPLEX, 0.45, temp_color, 1)

        # Radar distance display
        y_offset = 115
        if display_distance:
            if display_distance < RADAR_DANGER_DIST:
                dist_color = (0, 0, 255)
                status = " ⚠️ DANGER!"
            elif display_distance < RADAR_DANGER_DIST + 50:
                dist_color = (0, 165, 255)
                status = " ⚠️ Warning"
            else:
                dist_color = (0, 255, 0)
                status = " ✓ Safe"
            
            cv2.putText(annotated, f"📡 RADAR: {display_distance:.0f} cm{status}", 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, dist_color, 2)
        else:
            cv2.putText(annotated, "📡 RADAR: Waiting...", 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100,100,100), 1)

        # Alert banner on screen
        if buzzer_on:
            ov = annotated.copy()
            cv2.rectangle(ov, (0,0), (w,8), (0,0,255), -1)
            cv2.rectangle(ov, (0,h-8), (w,h), (0,0,255), -1)
            cv2.addWeighted(ov, 0.6, annotated, 0.4, 0, annotated)
            
            cv2.putText(annotated, f"⚠️ {alert_message} ⚠️", (w//2 - 120, h//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("SafeVision - Tri-Sensor Fusion", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('h'):
            dehaze_enabled = not dehaze_enabled
            print(f"Dehaze toggled: {'ON' if dehaze_enabled else 'OFF'}")
        elif key == ord('t'):
            show_thermal_overlay = not show_thermal_overlay
            print(f"Thermal overlay: {'ON' if show_thermal_overlay else 'OFF'}")
        elif key == ord('g'):
            show_thermal_grid = not show_thermal_grid
            print(f"Thermal grid: {'ON' if show_thermal_grid else 'OFF'}")

finally:
    print("\n🛑 Shutting down SafeVision Fusion...")
    lcd_clear()
    lcd_display("SafeVision", "Shutdown")
    time.sleep(1)
    lcd_clear()
    lgpio.gpio_write(chip, BUZZER_PIN, 0)
    thermal_sensor.stop()
    cap.release()
    if ser:
        ser.close()
    lgpio.gpiochip_close(chip)
    cv2.destroyAllWindows()
