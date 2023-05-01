import cv2
import json
import numpy as np
import os
from tqdm import tqdm
# Load image and bbox
image_path = 'whore_images'
image_list = os.listdir(image_path)

for image_name in tqdm(image_list):
    image_full_path = os.path.join(image_path, image_name)
    image = cv2.imread(image_full_path)
    h,w,c = image.shape
    with open('assets/colab/bbox_sportsfan.json') as jsonfile:
        bbox = json.load(jsonfile)['bbox']

    """### Inference and visualization:"""

    from spiga.inference.config import ModelConfig
    from spiga.inference.framework import SPIGAFramework

    # Process image
    dataset = 'wflw'
    processor = SPIGAFramework(ModelConfig(dataset))
    features = processor.inference(image, [[0,0,w,h]])

    import copy
    from spiga.demo.visualize.plotter import Plotter

    # Prepare variables
    # x0,y0,w,h = bbox
    canvas = copy.deepcopy(image)
    landmarks = np.array(features['landmarks'][0])
    headpose = np.array(features['headpose'][0])

    # Plot features
    plotter = Plotter()
    canvas = plotter.landmarks.draw_landmarks(canvas, landmarks, thick=10)
    # canvas = plotter.hpose.draw_headpose(canvas, [0,0,0,0], headpose[:3], headpose[3:], euler=True)

    # Show image results
    (h, w) = canvas.shape[:2]
    canvas = cv2.resize(canvas, (512, int(h*512/w)))
    cv2.imwrite('outputs/'+image_name, canvas)
