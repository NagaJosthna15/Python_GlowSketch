import cv2
import numpy as np
import random
from step1_load import load_and_fit, CANVAS_W, CANVAS_H
from step2_neon import make_neon_edge_layer

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
def get_blocks(img, block_size=10):
    h, w = img.shape[:2]
    blocks = []
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            sub = img[y:y+block_size, x:x+block_size]
            max_val = sub.max()
            if max_val > 18: # Dark pixels ni skip chestham
                center_color = img[min(y + block_size//2, h-1), min(x + block_size//2, w-1)]
                blocks.append((x, y, block_size, tuple(int(c) for c in center_color)))
    return blocks
if __name__ == "__main__":
    image_name = "shiva.png"
    
    base_img = load_and_fit(image_name, CANVAS_W, CANVAS_H)
    neon_img = make_neon_edge_layer(base_img)
    blocks = get_blocks(neon_img, block_size=10)
    random.seed(42)
    random.shuffle(blocks)
    
    canvas = np.zeros_like(neon_img)
    total_blocks = len(blocks)
    frames_count = 120 
    
    print("Animation start avthundhi... Close cheyadaniki 'q' or 'Esc' nokandi.")
    
    revealed = 0
    for i in range(frames_count):
        target = int(total_blocks * (i + 1) / frames_count)
        for b in blocks[revealed:target]:
            bx, by, bsize, color = b
            center = (bx + bsize // 2, by + bsize // 2)
            cv2.circle(canvas, center, bsize // 2, color, -1, lineType=cv2.LINE_AA)
        
        revealed = target
        final_frame = add_reflection(canvas)
        cv2.imshow("Step 3 - Dots Reveal & Reflection", final_frame)
        
        key = cv2.waitKey(16) & 0xFF
        if key == ord('q') or key == 27:
            break
    final_sharp = add_reflection(neon_img)
    cv2.imshow("Step 3 - Dots Reveal & Reflection", final_sharp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()