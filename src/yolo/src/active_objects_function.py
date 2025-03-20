from src.iou_function import iou    

def active_objects_retrieval(results, iou_threshold=0.1):
    
    if not results or len(results) == 0:
        return None
    
    boxes = results[0].boxes.xyxy.cpu().tolist()
    scores = results[0].boxes.conf.cpu().tolist()
    obj_ids = results[0].boxes.cls.cpu().tolist()
    
    predictions_list = [
        {"box": box, "score": score, "obj_id": int(obj_id)}
        for box, score, obj_id in zip(boxes, scores, obj_ids)
    ]
    
    hands = [p for p in predictions_list if p["obj_id"] == 25]  # Hand class id is 25
    objects = [p for p in predictions_list if p["obj_id"] != 25]  # Other objects
    
    if not hands or not objects:
        return None
    
    # Sort hands by confidence and take top 2
    hands_sorted = sorted(hands, key=lambda x: x["score"], reverse=True)[:2]
    
    # Find active objects based on iou
    best_iou = -1
    best_obj = None
    
    for hand in hands_sorted:
        hand_box = hand["box"]
        for obj in objects:
            obj_box = obj["box"]
            current_iou = iou(hand_box, obj_box)
            if current_iou >= iou_threshold and current_iou > best_iou:
                best_iou = current_iou
                best_obj = obj
    
    return best_obj
