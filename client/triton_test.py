import cv2
import numpy as np
import time

from shapely.geometry import box as shapely_box

from GroundingDINO_Triton import GroundingDINO
from data_loader import CustomImageDataset


def convert_boxes(boxes):
    """ Convert bounding boxes to shapely format """
    new_boxes = []
    for box in boxes:
        new_boxes.append(shapely_box(int((box[0] * 256) - ((box[2] * 256) / 2)),
                                     int((box[1] * 256) - ((box[3] * 256) / 2)),
                                     int((box[0] * 256) + ((box[2] * 256) / 2)),
                                     int((box[1] * 256) + ((box[3] * 256) / 2))))
        
    return new_boxes

def get_intersections(boxes) -> list:
    """ If head intersects with an animal we count it as a real head """
    intersection_list = []

    for idx, i_rect in enumerate(boxes):
      all_intersections = boxes
      rect1 = all_intersections[idx]
      intersection_counter = 0
      for j_rect in all_intersections[1:]:
        if not (rect1.intersection(j_rect).is_empty):
          intersection_counter += 1
          rect1 = rect1.intersection(j_rect)
      if intersection_counter > 0:
        intersection_list.append(rect1)

    return intersection_list

GD = GroundingDINO("localhost:8001")
dataset = CustomImageDataset("./data/labels", "./data/images")

hit_counter = 0
index = 0
mean_dist = 0
start_time = time.time()
for image_object in dataset:
    index += 1
    image = image_object[0]
    image_source = image_object[1]
    image_cv = image_object[2]
    label = image_object[3][0]
    points = image_object[3][1][0]    

    image_cv = cv2.rectangle(image_cv, (int(points[0]), int(points[1])), (int(points[0] + 2), int(points[1] + 2)), (255, 0, 0) , 2)
    
    boxes_animal, logits_animal = GD.inference("animal. pet ", 0.15, 0.15, image)
    boxes_animal = convert_boxes(boxes_animal)
    animal = boxes_animal[0]

    heads_boxes, heads_logits = GD.inference("head.", 0.2, 0.25, image)
    heads_boxes = convert_boxes(heads_boxes)


    intersections = []
    center = [[0], [0]]
    if animal != [0, 0, 0, 0]:
        for box in heads_boxes:
            if not (box.intersection(animal)).is_empty:
                intersections.append(box)
    else:
        intersections = heads_boxes
    if len(intersections) > 2:
        intersections = get_intersections(intersections)
        
    if len(intersections) > 0:
        center = intersections[len(intersections) - 1].centroid.coords.xy
        image_cv = cv2.rectangle(image_cv, (int(center[0][0]), int(center[1][0])), (int(center[0][0] + 2), int(center[1][0] + 2)), (0, 0, 255) , 2)
    
    point_1 = np.array((center[0][0], center[1][0]))
    point_2 = np.array(points)
    square = np.square(point_1 - point_2)
    sum_square = np.sum(square)
    dist = np.sqrt(sum_square)
    mean_dist += dist
    if (dist / 256) < 0.1:
        hit_counter += 1
       
print("Средняя точность: ", hit_counter / index)
print("Среднее расстояние: ", (mean_dist / index) / 256)
print("Время работы: ", (time.time() - start_time))