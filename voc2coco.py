#!/usr/bin/env python3
"""
Convert PASCAL VOC XML → COCO JSON format
For finetuning Deformable-DETR with --dataset_file coco

Usage:
    python voc2coco.py \
        --voc_path /data02/yolov3_voc/VOCdevkit/VOC2007 \
        --output_path ./data/voc_coco \
        --train_split trainval \
        --val_split test
"""

import os, sys, json, shutil
from pathlib import Path
import xml.etree.ElementTree as ET
from collections import OrderedDict

VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
    'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

def parse_voc_xml(xml_path):
    """Parse VOC XML and return image info + list of annotations."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find('filename').text
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    objs = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        difficult = int(obj.find('difficult').text) if obj.find('difficult') is not None else 0
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)

        # Skip difficult objects (optional, same as COCO practice)
        # if difficult == 1:
        #     continue

        objs.append({
            'name': name,
            'bbox': [xmin, ymin, xmax - xmin, ymax - ymin],  # xywh
            'area': (xmax - xmin) * (ymax - ymin),
            'difficult': difficult,
        })

    return filename, width, height, objs


def load_image_ids(split_file):
    """Load image IDs from VOC split txt file."""
    ids = []
    with open(split_file) as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(line.split()[0])  # Some files have -1/+1 flags
    return ids


def convert(voc_path, output_path, train_split='trainval', val_split='test', copy_images=True):
    voc_path = Path(voc_path)
    output_path = Path(output_path)

    ann_dir = voc_path / 'Annotations'
    img_dir = voc_path / 'JPEGImages'
    split_dir = voc_path / 'ImageSets' / 'Main'

    # Category mapping
    categories = [
        {'id': i + 1, 'name': name, 'supercategory': 'none'}
        for i, name in enumerate(VOC_CLASSES)
    ]

    splits = {
        'train': train_split,
        'val': val_split,
    }

    for split_name, split_file in splits.items():
        print(f"\n=== Processing {split_name} (split: {split_file}) ===")

        split_path = split_dir / f'{split_file}.txt'
        if not split_path.exists():
            print(f"  WARNING: {split_path} not found, skipping")
            continue

        image_ids = load_image_ids(split_path)
        print(f"  Found {len(image_ids)} images")

        out_img_dir = output_path / split_name / 'images'
        out_img_dir.mkdir(parents=True, exist_ok=True)

        images = []
        annotations = []
        ann_id = 1

        for idx, img_id in enumerate(image_ids):
            if (idx + 1) % 500 == 0:
                print(f"  Progress: {idx + 1}/{len(image_ids)}")

            xml_path = ann_dir / f'{img_id}.xml'
            img_file = img_dir / f'{img_id}.jpg'

            if not xml_path.exists() or not img_file.exists():
                continue

            filename, width, height, objs = parse_voc_xml(xml_path)

            images.append({
                'id': idx + 1,
                'file_name': f'{img_id}.jpg',
                'width': width,
                'height': height,
            })

            for obj in objs:
                cat_id = VOC_CLASSES.index(obj['name']) + 1
                annotations.append({
                    'id': ann_id,
                    'image_id': idx + 1,
                    'category_id': cat_id,
                    'bbox': [round(v, 2) for v in obj['bbox']],
                    'area': round(obj['area'], 2),
                    'iscrowd': 0,
                    'segmentation': [],
                })
                ann_id += 1

            # Copy image
            if copy_images:
                dst = out_img_dir / f'{img_id}.jpg'
                if not dst.exists():
                    shutil.copy2(img_file, dst)

        # Write COCO JSON
        coco_json = {
            'images': images,
            'annotations': annotations,
            'categories': categories,
        }

        json_path = output_path / split_name / f'{split_name}.json'
        with open(json_path, 'w') as f:
            json.dump(coco_json, f)

        print(f"  Images: {len(images)}, Annotations: {len(annotations)}")
        print(f"  Saved: {json_path}")

    print("\n✅ Done! Use with:")
    print(f"    --coco_path {output_path} --num_classes {len(VOC_CLASSES) + 1}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--voc_path', required=True, help='Path to VOC2007 directory')
    parser.add_argument('--output_path', required=True, help='Output COCO-format directory')
    parser.add_argument('--train_split', default='trainval', help='Split file for training (default: trainval)')
    parser.add_argument('--val_split', default='test', help='Split file for validation (default: test)')
    parser.add_argument('--no-copy', action='store_true', help='Do not copy images (just generate JSON)')
    args = parser.parse_args()

    convert(args.voc_path, args.output_path, args.train_split, args.val_split, copy_images=not args.no_copy)