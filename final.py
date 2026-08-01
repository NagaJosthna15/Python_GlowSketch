import cv2
import numpy as np
import random
import math
import pygame

CANVAS_W, CANVAS_H = 1280, 720
FPS = 60

def load_and_fit(image_path, canvas_w, canvas_h):
    img = cv2.imread(image_path)
    if img is None:
        return None
    ih, iw = img.shape[:2]
    scale = min(canvas_w / iw, canvas_h / ih)
    new_w, new_h = int(iw * scale), int(ih * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    x_off = (canvas_w - new_w) // 2
    y_off = (canvas_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas

def make_neon_edge_layer(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    edges = cv2.Canny(gray, 60, 150)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)

    mask = edges.astype(bool)
    color_layer = np.zeros_like(img)
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.7, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * 1.5 + 60, 0, 255)
    boosted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    color_layer[mask] = boosted[mask]

    inner_glow = cv2.GaussianBlur(color_layer, (0, 0), sigmaX=4, sigmaY=4)
    outer_glow = cv2.GaussianBlur(color_layer, (0, 0), sigmaX=14, sigmaY=14)

    combined = (
        color_layer.astype(np.float32) * 1.4
        + inner_glow.astype(np.float32) * 1.1
        + outer_glow.astype(np.float32) * 0.7
    ) * 1.6
    return np.clip(combined, 0, 255).astype(np.uint8)

def add_reflection(frame, ratio=0.28):
    h, w = frame.shape[:2]
    refl_h = int(h * ratio)
    reflection = cv2.flip(frame[h - refl_h:h, :], 0)
    fade = np.linspace(0.35, 0.0, refl_h).reshape(refl_h, 1, 1)
    reflection = (reflection.astype(np.float32) * fade).astype(np.uint8)
    
    out = frame.copy()
    start_y = h - refl_h
    blended = cv2.addWeighted(out[start_y:h], 0.4, reflection, 0.9, 0)
    out[start_y:h] = np.maximum(out[start_y:h], blended)
    return out

def apply_zoom(frame, zoom_factor):
    h, w = frame.shape[:2]
    new_h, new_w = int(h / zoom_factor), int(w / zoom_factor)
    top = (h - new_h) // 2
    left = (w - new_w) // 2
    cropped = frame[top:top + new_h, left:left + new_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

class FloatingParticles:
    def __init__(self, count=40, width=CANVAS_W, height=CANVAS_H):
        self.w = width
        self.h = height
        self.particles = []
        for _ in range(count):
            x = random.randint(0, width)
            y = random.randint(0, height)
            speed = random.uniform(0.5, 1.5)
            radius = random.randint(1, 3)
            color = (random.randint(150, 255), random.randint(150, 255), 255)
            self.particles.append([x, y, speed, radius, color])

    def draw_and_update(self, frame):
        out = frame.copy()
        for p in self.particles:
            p[1] -= p[2]
            if p[1] < 0:
                p[1] = self.h
                p[0] = random.randint(0, self.w)
            cv2.circle(out, (int(p[0]), int(p[1])), p[3], p[4], -1, lineType=cv2.LINE_AA)
        return out

def get_blocks(img, block_size=10):
    h, w = img.shape[:2]
    blocks = []
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            sub = img[y:y+block_size, x:x+block_size]
            if sub.max() > 18:
                center_color = img[min(y + block_size//2, h-1), min(x + block_size//2, w-1)]
                blocks.append((x, y, block_size, tuple(int(c) for c in center_color)))
    return blocks

if __name__ == "__main__":
    image_name = "shiva.png"
    audio_name = "song.mp3"
    
    base_img = load_and_fit(image_name, CANVAS_W, CANVAS_H)
    neon_img = make_neon_edge_layer(base_img)
    
    pygame.mixer.init()
    try:
        pygame.mixer.music.load(audio_name)
        pygame.mixer.music.play()
    except Exception as e:
        print(f"Audio error: {e}")

    blocks = get_blocks(neon_img, block_size=10)
    random.seed(42)
    random.shuffle(blocks)
    
    particles = FloatingParticles(count=50)
    delay_ms = max(1, int(1000 / FPS))
    total_frames = int(35.0 * FPS)
    
    canvas = np.zeros_like(neon_img)
    total_blocks = len(blocks)
    dot_frames = int(8.0 * FPS)
    hold_frames = int(6.0 * FPS)
    cross_frames = int(4.0 * FPS)
    
    revealed = 0

    for frame_idx in range(total_frames):
        zoom = 1.0 + (0.12 * (frame_idx / total_frames))
        pulse = 1.0 + 0.15 * math.sin(frame_idx * 0.1)

        if frame_idx < dot_frames:
            target = int(total_blocks * (frame_idx + 1) / dot_frames)
            for b in blocks[revealed:target]:
                bx, by, bsize, color = b
                center = (bx + bsize // 2, by + bsize // 2)
                cv2.circle(canvas, center, bsize // 2, color, -1, lineType=cv2.LINE_AA)
            revealed = target
            current_frame = canvas.copy()
        
        elif frame_idx < (dot_frames + hold_frames):
            pulsed_neon = cv2.convertScaleAbs(neon_img, alpha=pulse, beta=0)
            current_frame = pulsed_neon
            
        elif frame_idx < (dot_frames + hold_frames + cross_frames):
            t = (frame_idx - dot_frames - hold_frames) / cross_frames
            current_frame = cv2.addWeighted(neon_img, 1 - t, base_img, t, 0)
            
        else:
            current_frame = base_img

        current_frame = add_reflection(current_frame)
        
        current_frame = particles.draw_and_update(current_frame)
        
        final_frame = apply_zoom(current_frame, zoom)

        cv2.imshow("Cinematic Neon Reveal", final_frame)
        
        key = cv2.waitKey(delay_ms) & 0xFF
        if key == ord('q') or key == 27:
            break
        if cv2.getWindowProperty("Cinematic Neon Reveal", cv2.WND_PROP_VISIBLE) < 1:
            break

    pygame.mixer.music.stop()
    cv2.destroyAllWindows()