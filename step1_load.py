import cv2
import numpy as np

CANVAS_W, CANVAS_H = 1280, 720

def load_and_fit(image_path, canvas_w, canvas_h):
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Could not find image '{image_path}'! Please check the file path.")
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


if __name__ == "__main__":
  
    image_name = "shiva.png" 
    
    fitted_image = load_and_fit(image_name, CANVAS_W, CANVAS_H)
    
    if fitted_image is not None:
    
        cv2.imshow("Step 1 - Letterboxed Image", fitted_image)
        
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()