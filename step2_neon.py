import cv2
import numpy as np
from step1_load import load_and_fit, CANVAS_W, CANVAS_H

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

    combined = np.clip(combined, 0, 255).astype(np.uint8)
    return combined

if __name__ == "__main__":
    image_name = "shiva.png"  
    
    base_img = load_and_fit(image_name, CANVAS_W, CANVAS_H)
    
    if base_img is not None:
        neon_img = make_neon_edge_layer(base_img)
        
        cv2.imshow("Step 1 - Original Fitted", base_img)
        cv2.imshow("Step 2 - Neon Sketch Glow", neon_img)
        
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()