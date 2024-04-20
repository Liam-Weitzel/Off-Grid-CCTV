import os
import xml.etree.ElementTree as ET

# Path configurations
label_dir = './data/toggenburg_alpaca_ranch/valid/labels/'
image_dir = './data/toggenburg_alpaca_ranch/valid/images/'
output_dir = './data/toggenburg_alpaca_ranch/valid/xml/'

# Dictionary to map class indices to names
class_names = {0: 'car', 1: 'sheep'}

def convert_to_xml(label_file, image_file, output_file):
    tree = ET.Element('annotation')
    ET.SubElement(tree, 'folder').text = 'images'
    ET.SubElement(tree, 'filename').text = os.path.basename(image_file)

    width, height, depth = 480, 270, 3
    size = ET.SubElement(tree, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    ET.SubElement(size, 'depth').text = str(depth)
    
    with open(label_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center, y_center = float(parts[1]) * width, float(parts[2]) * height
            box_width, box_height = float(parts[3]) * width, float(parts[4]) * height
            xmin, ymin = int(x_center - box_width / 2), int(y_center - box_height / 2)
            xmax, ymax = int(x_center + box_width / 2), int(y_center + box_height / 2)

            obj = ET.SubElement(tree, 'object')
            ET.SubElement(obj, 'name').text = class_names[class_id]
            ET.SubElement(obj, 'pose').text = 'Unspecified'
            ET.SubElement(obj, 'truncated').text = '0'
            ET.SubElement(obj, 'occluded').text = '0'
            ET.SubElement(obj, 'difficult').text = '0'
            bndbox = ET.SubElement(obj, 'bndbox')
            ET.SubElement(bndbox, 'xmin').text = str(xmin)
            ET.SubElement(bndbox, 'ymin').text = str(ymin)
            ET.SubElement(bndbox, 'xmax').text = str(xmax)
            ET.SubElement(bndbox, 'ymax').text = str(ymax)

    tree = ET.ElementTree(tree)
    tree.write(output_file)

# Convert all label files
for filename in os.listdir(label_dir):
    if filename.endswith('.txt'):
        label_file = os.path.join(label_dir, filename)
        image_name = filename.split('.')[0] + '.png'  # Assuming image extension is png
        image_file = os.path.join(image_dir, image_name)
        output_file = os.path.join(output_dir, image_name.replace('.png', '.xml'))
        convert_to_xml(label_file, image_file, output_file)

print("Conversion completed!")
