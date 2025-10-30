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
    parser.add_argument(
        "--vehicle_only",
        action="store_true",
        help="Track only vehicle classes (car, motorcycle, bus, truck)",
    )
    parser.add_argument(
        "--min_hits",
        type=int,
        default=3,
        help="Minimum hits before showing a track",
    )
    parser.add_argument(
        "--hide_when_predict",
        action="store_true",
        help="Hide label when track not updated by detection this frame",
    )
    parser.add_argument(
        "--bbox_max_grow",
        type=float,
        default=1.3,
        help="Per-frame max growth factor for bbox size",
    )
    parser.add_argument(
        "--bbox_max_shrink",
        type=float,
        default=0.7,
        help="Per-frame min shrink factor for bbox size",
    )
    parser.add_argument(
        "--bbox_max_center_shift",
        type=float,
        default=0.25,
        help="Max fraction of previous box size center can shift per frame",
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
        default=0.5,
        help="Scale multiplier for label font size (baseline auto-scales by frame height)",
    )
    parser.add_argument(
        "--label_text_bgr",
        type=str,
        default="230,230,230",
        help="Text color as B,G,R (e.g., '230,230,230' for light gray)",
    )
    parser.add_argument(
        "--label_bg_bgr",
        type=str,
        default="0,0,0",
        help="Background color (box mode) as B,G,R (e.g., '0,0,0' for black)",
    )
    opt = parser.parse_args()
    return opt



def draw_corner_rect(img, bbox, line_length=30, line_thickness=5, rect_thickness=1,
                     rect_color=(255, 0, 255), line_color=(0, 255, 0)):
    x, y, w, h = bbox
    ih, iw = img.shape[:2]
    # Clamp to image bounds
    x = max(0, min(int(x), iw - 1))
    y = max(0, min(int(y), ih - 1))
    w = max(1, int(w))
    h = max(1, int(h))
    x1, y1 = x + w, y + h
    x1 = max(0, min(x1, iw - 1))
    y1 = max(0, min(y1, ih - 1))

    if rect_thickness != 0:
        cv2.rectangle(img, (x, y), (x1, y1), rect_color, rect_thickness)

    # Top Left  x, y
    cv2.line(img, (x, y), (min(x + line_length, iw - 1), y), line_color, line_thickness)
    cv2.line(img, (x, y), (x, min(y + line_length, ih - 1)), line_color, line_thickness)

    # Top Right  x1, y
    cv2.line(img, (x1, y), (max(x1 - line_length, 0), y), line_color, line_thickness)
    cv2.line(img, (x1, y), (x1, min(y + line_length, ih - 1)), line_color, line_thickness)

    # Bottom Left  x, y1
    cv2.line(img, (x, y1), (min(x + line_length, iw - 1), y1), line_color, line_thickness)
    cv2.line(img, (x, y1), (x, max(y1 - line_length, 0)), line_color, line_thickness)

    # Bottom Right  x1, y1
    cv2.line(img, (x1, y1), (max(x1 - line_length, 0), y1), line_color, line_thickness)
    cv2.line(img, (x1, y1), (x1, max(y1 - line_length, 0)), line_color, line_thickness)

    return img

def calculate_speed(distance, fps):
    return (distance *fps)*3.6


def calculate_distance(p1, p2):
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


def read_frames(cap):
    # COCO vehicle classes (car, motorcycle, bus, truck)
    VEHICLE_CLASSES = {2, 3, 5, 7}

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

    # Initialize the DeepSort tracker with stricter init
    tracker = DeepSort(max_age=50, n_init=opt.min_hits)
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
    smooth_bboxes = {}
    # Smoothed bounding boxes to reduce jitter and avoid sudden oversized boxes
    smooth_bboxes = {}

    def _parse_bgr(s: str):
        try:
            b, g, r = [int(v.strip()) for v in s.split(",")]
            return (max(0, min(b, 255)), max(0, min(g, 255)), max(0, min(r, 255)))
        except Exception:
            return (230, 230, 230)

    text_color = _parse_bgr(opt.label_text_bgr)
    bg_color = _parse_bgr(opt.label_bg_bgr)

    def draw_speed_label(img, bbox, text, mode="outline", alpha=0.4, scale_mult=1.0,
                         text_bgr=(230, 230, 230), bg_bgr=(0, 0, 0)):
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
            cv2.rectangle(overlay, (bx1, by1), (bx2, by2), bg_bgr, -1)
            cv2.addWeighted(overlay, float(alpha), img, 1 - float(alpha), 0, img)
            # White text with AA
            cv2.putText(img, text, (tx, ty), font, base_scale, text_bgr, 2, cv2.LINE_AA)
        else:
            # Outline text: use slimmer strokes to keep text slender
            cv2.putText(img, text, (tx, ty), font, base_scale, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, text, (tx, ty), font, base_scale, text_bgr, 1, cv2.LINE_AA)
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
                # Optional: filter to vehicles
                if opt.vehicle_only and int(label) not in VEHICLE_CLASSES:
                    continue
                    
                if polygon_mask[(y1 + y2) // 2, (x1 + x2) // 2] == 255:
                    detect.append([[x1, y1, x2 - x1, y2 - y1], confidence, int(label)])            
        tracks = tracker.update_tracks(detect, frame=frame)
        for track in tracks:
            if not track.is_confirmed():
                continue
            if hasattr(track, "hits") and track.hits < opt.min_hits:
                continue
            if opt.hide_when_predict and hasattr(track, "time_since_update") and track.time_since_update > 0:
                continue
            track_id = track.track_id    
            ltrb = track.to_ltrb()
            class_id = track.get_det_class()
            x1, y1, x2, y2 = map(int, ltrb)
            # Clamp coordinates to frame bounds to avoid overflow for large objects
            x1 = max(0, min(x1, frame.shape[1] - 1))
            x2 = max(0, min(x2, frame.shape[1] - 1))
            y1 = max(0, min(y1, frame.shape[0] - 1))
            y2 = max(0, min(y2, frame.shape[0] - 1))
            if x2 <= x1:
                x2 = min(frame.shape[1] - 1, x1 + 1)
            if y2 <= y1:
                y2 = min(frame.shape[0] - 1, y1 + 1)
            # Smooth bbox with explicit growth/shift clamps to avoid oversize on trucks
            cur_box = np.array([x1, y1, x2, y2], dtype=float)
            if track_id in smooth_bboxes:
                prev_box = smooth_bboxes[track_id]
                pw = max(1.0, prev_box[2] - prev_box[0])
                ph = max(1.0, prev_box[3] - prev_box[1])
                cw = max(1.0, cur_box[2] - cur_box[0])
                ch = max(1.0, cur_box[3] - cur_box[1])
                pcx = (prev_box[0] + prev_box[2]) / 2.0
                pcy = (prev_box[1] + prev_box[3]) / 2.0
                ccx = (cur_box[0] + cur_box[2]) / 2.0
                ccy = (cur_box[1] + cur_box[3]) / 2.0
                scale_w = np.clip(cw / pw, opt.bbox_max_shrink, opt.bbox_max_grow)
                scale_h = np.clip(ch / ph, opt.bbox_max_shrink, opt.bbox_max_grow)
                nw = pw * scale_w
                nh = ph * scale_h
                max_shift_x = pw * opt.bbox_max_center_shift
                max_shift_y = ph * opt.bbox_max_center_shift
                dcx = np.clip(ccx - pcx, -max_shift_x, max_shift_x)
                dcy = np.clip(ccy - pcy, -max_shift_y, max_shift_y)
                ncx = pcx + dcx
                ncy = pcy + dcy
                sx1 = int(max(0, min(frame.shape[1] - 1, ncx - nw / 2.0)))
                sy1 = int(max(0, min(frame.shape[0] - 1, ncy - nh / 2.0)))
                sx2 = int(max(0, min(frame.shape[1] - 1, ncx + nw / 2.0)))
                sy2 = int(max(0, min(frame.shape[0] - 1, ncy + nh / 2.0)))
                smoothed = np.array([sx1, sy1, sx2, sy2], dtype=float)
            else:
                smoothed = cur_box
            smooth_bboxes[track_id] = smoothed
            sx1, sy1, sx2, sy2 = [int(v) for v in smoothed]
            if polygon_mask[(sy1+sy2)//2,(sx1+sx2)//2] == 0:
                # Skip drawing/tracking outside polygon instead of mutating list in place
                continue
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
            # Draw bounding box (adaptive, slimmer) and text
            bw = max(1, sx2 - sx1)
            bh = max(1, sy2 - sy1)
            line_len = max(8, int(min(bw, bh) * 0.12))
            line_thk = max(1, int(round(min(bw, bh) / 400)))
            frame = draw_corner_rect(
                frame,
                (sx1, sy1, bw, bh),
                line_length=line_len,
                line_thickness=line_thk,
                rect_thickness=1,
                rect_color=(B, G, R),
                line_color=(R, G, B),
            )
            #cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
            # cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(text) * 10, y1), (B, G, R), -1)
            # cv2.putText(frame, text, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            if track_id in speed_accumulator:
                avg_speed = sum(speed_accumulator[track_id]) / len(speed_accumulator[track_id])
                label = f"{avg_speed:.0f} km/h"
                frame = draw_speed_label(
                    frame,
                    (sx1, sy1, sx2, sy2),
                    label,
                    mode=opt.label_mode,
                    alpha=opt.label_alpha,
                    scale_mult=opt.label_scale,
                    text_bgr=text_color,
                    bg_bgr=bg_color,
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
