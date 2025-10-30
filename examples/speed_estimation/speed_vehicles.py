import cv2
import torch
import numpy as np
import argparse, os
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        type=str,
        nargs="?",
        default="content/car_org.mp4",
        help="Path to input video"
    )
    parser.add_argument(
        "--output",
        type=str,
        nargs="?",
        help="path to output video",
        default="content/output.mp4"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.50,
        help="confidence threshold",
    )
    parser.add_argument(
        "--blur_id",
        type=int,
        default=None,
        help="class ID to apply Gaussian Blur",
    )
    parser.add_argument(
        "--class_id",
        type=int,
        default=None,
        help="class ID to track",
    )
    # Label rendering options
    parser.add_argument(
        "--label_mode",
        type=str,
        choices=["outline", "box"],
        default="outline",
        help="Speed label style: 'outline' (clear, minimal occlusion) or 'box' (background box)",
    )
    parser.add_argument(
        "--label_alpha",
        type=float,
        default=0.4,
        help="Background alpha if label_mode='box' (0..1)",
    )
    parser.add_argument(
        "--label_scale",
        type=float,
        default=1.0,
        help="Scale multiplier for label font size (baseline auto-scales by frame height)",
    )
    opt = parser.parse_args()
    return opt



def draw_corner_rect(img, bbox, line_length=30, line_thickness=5, rect_thickness=1,
                     rect_color=(255, 0, 255), line_color=(0, 255, 0)):
    x, y, w, h = bbox
    x1, y1 = x + w, y + h

    if rect_thickness != 0:
        cv2.rectangle(img, bbox, rect_color, rect_thickness)

    # Top Left  x, y
    cv2.line(img, (x, y), (x + line_length, y), line_color, line_thickness)
    cv2.line(img, (x, y), (x, y + line_length), line_color, line_thickness)

    # Top Right  x1, y
    cv2.line(img, (x1, y), (x1 - line_length, y), line_color, line_thickness)
    cv2.line(img, (x1, y), (x1, y + line_length), line_color, line_thickness)

    # Bottom Left  x, y1
    cv2.line(img, (x, y1), (x + line_length, y1), line_color, line_thickness)
    cv2.line(img, (x, y1), (x, y1 - line_length), line_color, line_thickness)

    # Bottom Right  x1, y1
    cv2.line(img, (x1, y1), (x1 - line_length, y1), line_color, line_thickness)
    cv2.line(img, (x1, y1), (x1, y1 - line_length), line_color, line_thickness)

    return img  

def calculate_speed(distance, fps):
    return (distance *fps)*3.6


def calculate_distance(p1, p2):
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


def read_frames(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame 


def main(_argv):
    FRAME_WIDTH=25
    FRAME_HEIGHT=70

    SOURCE_POLYGONE = np.array([
    [165, 877],   # bottom-left
    [1580, 880],  # bottom-right
    [1107, 534],  # top-right
    [749, 538]    # top-left
    ], dtype=np.float32)

    BIRD_EYE_VIEW = np.array([[0, 0], [FRAME_WIDTH, 0], [FRAME_WIDTH, FRAME_HEIGHT],[0, FRAME_HEIGHT]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(SOURCE_POLYGONE, BIRD_EYE_VIEW)


    # Initialize the video capture
    video_input = opt.video

    cap = cv2.VideoCapture(video_input)
    if not cap.isOpened():
        print('Error: Unable to open video source.')
        return
    
  
    frame_generator = read_frames(cap)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    pts = SOURCE_POLYGONE.astype(np.int32) 
    pts = pts.reshape((-1, 1, 2))

    polygon_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    cv2.fillPoly(polygon_mask, [pts], 255)
    # video writer objects
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(opt.output, fourcc, fps, (frame_width, frame_height))

    # Initialize the DeepSort tracker
    tracker = DeepSort(max_age=50)
    # Load YOLO model
    model = YOLO("yolov10n.pt")
    # Use model's built-in class names to avoid external files
    try:
        class_names = model.names
        if isinstance(class_names, dict):
            # Ultralytics exposes a dict idx->name
            class_names = [class_names[i] for i in range(len(class_names))]
    except Exception:
        class_names = None

    np.random.seed(42)
    n_classes = len(class_names) if isinstance(class_names, (list, tuple)) else 80
    colors = np.random.randint(0, 255, size=(n_classes, 3)) 
    # FPS calculation variables
    frame_count = 0
    start_time = time.time()
    prev_positions={}
    speed_accumulator={}

    def draw_speed_label(img, bbox, text, mode="outline", alpha=0.4, scale_mult=1.0):
        x1, y1, x2, y2 = bbox
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Base scale follows resolution; multiplied by user factor
        base_scale = max(0.7, img.shape[0] / 1080.0 * 0.9) * float(scale_mult)
        # Prefer to place above; if not enough room, place below
        (tw, th), baseline = cv2.getTextSize(text, font, base_scale, 2)
        pad = 6
        # Try above
        bx1 = x1
        by2 = y1 - 4
        by1 = by2 - th - baseline - 2 * pad
        # If above goes out of frame, place below
        if by1 < 0:
            by1 = y2 + 4
            by2 = min(img.shape[0] - 1, by1 + th + baseline + 2 * pad)
        # Keep inside right boundary
        bx2 = min(img.shape[1] - 1, bx1 + tw + 2 * pad)
        # If still outside width, shift left
        if bx2 >= img.shape[1]:
            bx1 = max(0, img.shape[1] - (tw + 2 * pad) - 1)
            bx2 = img.shape[1] - 1

        tx = bx1 + pad
        ty = by2 - pad - baseline

        if mode == "box":
            # Semi-transparent box
            overlay = img.copy()
            cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (0, 0, 0), -1)
            cv2.addWeighted(overlay, float(alpha), img, 1 - float(alpha), 0, img)
            # White text with AA
            cv2.putText(img, text, (tx, ty), font, base_scale, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            # Outline text: draw black thick, then white thinner (minimal occlusion)
            cv2.putText(img, text, (tx, ty), font, base_scale, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, text, (tx, ty), font, base_scale, (255, 255, 255), 2, cv2.LINE_AA)
        return img
    
    while True:
        try:
            frame = next(frame_generator)
        except StopIteration:
            break
        # Run model on each frame
        with torch.no_grad():
            results = model(frame)
        detect = []
        for pred in results:
            for box in pred.boxes:    
                x1, y1, x2, y2 = map(int, box.xyxy[0] )
                confidence = box.conf[0]     
                label = box.cls[0]

                # Filter out weak detections by confidence threshold and class_id
                if opt.class_id is None:
                    if confidence < opt.conf:
                        continue
                else:
                    # Use label (class index) for filtering
                    if int(label) != opt.class_id or confidence < opt.conf:
                        continue            
                    
                if polygon_mask[(y1 + y2) // 2, (x1 + x2) // 2] == 255:
                    detect.append([[x1, y1, x2 - x1, y2 - y1], confidence, int(label)])            
        tracks = tracker.update_tracks(detect, frame=frame)
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id    
            ltrb = track.to_ltrb()
            class_id = track.get_det_class()
            x1, y1, x2, y2 = map(int, ltrb)
            if polygon_mask[(y1+y2)//2,(x1+x2)//2] == 0:
                tracks.remove(track)
            # Fallback: if class_names is None, ensure colors length is safe
            color = colors[class_id % len(colors)]
            B, G, R = map(int, color)
            text = f"{track_id} - {class_names[class_id]}"
            center_pt = np.array([[(x1+x2)//2, (y1+y2)//2]], dtype=np.float32)
            transformed_pt = cv2.perspectiveTransform(center_pt[None, :, :], M)
            if track_id in prev_positions:
                prev_position = prev_positions[track_id]
                distance = calculate_distance(prev_position, transformed_pt[0][0])
                speed = calculate_speed(distance, fps)
                if track_id in speed_accumulator:
                    speed_accumulator[track_id].append(speed)
                    if len(speed_accumulator[track_id]) > 100:
                        speed_accumulator[track_id].pop(0)
                else:
                    speed_accumulator[track_id] = []
                    speed_accumulator[track_id].append(speed)
            prev_positions[track_id] = transformed_pt[0][0]
            # Draw bounding box and text
            frame = draw_corner_rect(frame, (x1, y1, x2 - x1, y2 - y1), line_length=15, line_thickness=3, rect_thickness=1, rect_color=(B, G, R), line_color=(R, G, B))
            #cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
            # cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(text) * 10, y1), (B, G, R), -1)
            # cv2.putText(frame, text, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            if track_id in speed_accumulator:
                avg_speed = sum(speed_accumulator[track_id]) / len(speed_accumulator[track_id])
                label = f"{avg_speed:.0f} km/h"
                frame = draw_speed_label(
                    frame,
                    (x1, y1, x2, y2),
                    label,
                    mode=opt.label_mode,
                    alpha=opt.label_alpha,
                    scale_mult=opt.label_scale,
                )
            # Apply Gaussian Blur
            if opt.blur_id is not None and class_id == opt.blur_id:
                print("true")
                if 0 <= x1 < x2 <= frame.shape[1] and 0 <= y1 < y2 <= frame.shape[0]:
                    frame[y1:y2, x1:x2] = cv2.GaussianBlur(frame[y1:y2, x1:x2], (99, 99), 3)

        # cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
        # cv2.putText(frame, f"Height: {FRAME_HEIGHT}", (1500, 900), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        # cv2.putText(frame, f"Width: {FRAME_WIDTH}", (1530, 930), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow('speed_estimation', frame)
        writer.write(frame)
        frame_count += 1
        if frame_count % 10 == 0:
            elapsed_time = time.time() - start_time
            fps_calc = frame_count / elapsed_time
            print(f"FPS: {fps_calc:.2f}")
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
