# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert raw PASCAL dataset to TFRecord for object_detection.

Example usage:
    python -m object_detection.dataset_tools.create_pascal_tf_record \
        --data_dir=/app/model_training/tutorial \
        --year=cifar_train \
        --output_path=/app/model_training/tutorial/records/cifar_train.record \
        --label_map_path=/app/model_training/tutorial/cifar10.pbtxt \
        --annotations_dir=/app/model_training/tutorial/cifar_train/Annotations \
        --images_dir=/app/model_training/tutorial/cifar_train/JPEGImages \

    python -m object_detection.dataset_tools.create_pascal_tf_record \
        --data_dir=/app/model_training/tutorial \
        --year=cifar_train \
        --output_path=/app/model_training/tutorial/records/cifar_train.record \
        --label_map_path=/app/model_training/tutorial/cifar10.pbtxt \
        --annotations_dir=/app/model_training/tutorial/cifar_train/Annotations \
        --images_dir=/app/model_training/tutorial/cifar_train/JPEGImages \
        --skip_category=$skip_category \
        --custom_label_map="$custom_label_map" \
        --keep_empty_image=$keep_empty_image

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os

from lxml import etree
import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '', 'Root directory to raw PASCAL VOC dataset.')
flags.DEFINE_string('set', 'all', 'Convert training set, validation set or '
                                    'merged set.')
flags.DEFINE_string('annotations_dir', 'Annotations',
                    '(Relative) path to annotations directory.')
flags.DEFINE_string('images_dir', 'JPEGImages',
                    '(Relative) path to images directory.')
flags.DEFINE_string('year', '', 'Desired challenge year.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', 'data/pascal_label_map.pbtxt',
                    'Path to label map proto')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore '
                                                          'difficult instances')
flags.DEFINE_string('annotation_image_dirs_file', '', 'Multiple annotation_dir & image_dir pairs text file.')
flags.DEFINE_string('skip_category', '', 'comma separated')
flags.DEFINE_string('custom_label_map', '', 'comma separated')
flags.DEFINE_boolean('keep_empty_image', False, 'Keep the image even the image doesnot contains any labels')


SETS = ['train', 'val', 'trainval', 'test', 'all']
YEARS = ['VOC2007', 'VOC2012', 'merged']

label_count = dict(total=0)
custom_label_map = dict()


def dict_to_tf_example(data,
                       images_dir,
                       label_map_dict,
                       ignore_difficult_instances=False):
    """Convert XML derived dict to tf.Example proto.

    Notice that this function normalizes the bounding box coordinates provided
    by the raw data.

    Args:
      data: dict holding PASCAL XML fields for a single image (obtained by
        running dataset_util.recursive_parse_xml_to_dict)
      dataset_directory: Path to root directory holding PASCAL dataset
      label_map_dict: A map from string label names to integers ids.
      ignore_difficult_instances: Whether to skip difficult instances in the
        dataset  (default: False).
      image_subdirectory: String specifying subdirectory within the
        PASCAL dataset directory holding the actual image data.

    Returns:
      example: The converted tf.Example.

    Raises:
      ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """
    full_path = os.path.join(images_dir, data['filename'])
    with tf.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format == 'PNG' or image.size[0] > 1920:
        image.thumbnail((1920, 1920), PIL.Image.ANTIALIAS)
        temp_file = io.BytesIO()
        image.save(temp_file, format="jpeg")

        temp_file.seek(0)
        encoded_jpg = temp_file.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    width = int(data['size']['width'])
    height = int(data['size']['height'])

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []

    if 'object' in data:
        for obj in data['object']:
            difficult = bool(int(obj['difficult']))
            if ignore_difficult_instances and difficult:
                continue

            if obj['name'] in custom_label_map:
                obj['name'] = custom_label_map.get(obj['name'])

            if not obj['name']:
                continue
            elif FLAGS.skip_category and obj['name'] in set(FLAGS.skip_category.split(',')):
                continue
            difficult_obj.append(int(difficult))
            obj['name'] = obj['name'].lower()
            xmin.append(float(obj['bndbox']['xmin']) / width)
            ymin.append(float(obj['bndbox']['ymin']) / height)
            xmax.append(float(obj['bndbox']['xmax']) / width)
            ymax.append(float(obj['bndbox']['ymax']) / height)
            classes_text.append(obj['name'].encode('utf8'))
            classes.append(label_map_dict[obj['name']])
            truncated.append(int(obj['truncated']))
            poses.append(obj['pose'].encode('utf8'))
            label_count[obj['name']] = label_count.get(obj['name'], 0) + 1
        if len(data['object']) > 0:
            label_count['total'] += 1
    if len(classes) == 0 and not FLAGS.keep_empty_image:
        return
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
        'image/object/view': dataset_util.bytes_list_feature(poses),
    }))
    return example


def main(_):
    print('FLAGS.custom_label_map =', FLAGS.custom_label_map)
    print('FLAGS.skip_category =', FLAGS.skip_category)
    print('FLAGS.custom_label_map =', custom_label_map)
    if FLAGS.custom_label_map:
        custom_label_map.update(dict(label_map.split(':') for label_map in FLAGS.custom_label_map.split(',')))

    # if FLAGS.set not in SETS:
    #   raise ValueError('set must be in : {}'.format(SETS))
    # if FLAGS.year not in YEARS:
    #   raise ValueError('year must be in : {}'.format(YEARS))

    data_dir = FLAGS.data_dir
    years = ['VOC2007', 'VOC2012']
    if FLAGS.year != 'merged':
        years = [FLAGS.year]

    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

    anno_image_dir_pairs = []
    if FLAGS.annotation_image_dirs_file:
        with open(FLAGS.annotation_image_dirs_file) as f:
            for line in f:
                anno_image_dir_pairs.append(line.split() + ['year'])
    else:
        for year in years:
            annotations_dir = os.path.join(data_dir, year, FLAGS.annotations_dir)
            images_dir = os.path.join(data_dir, year, FLAGS.images_dir)
            anno_image_dir_pairs.append((annotations_dir, images_dir, year))

    for annotations_dir, images_dir, year in anno_image_dir_pairs:
        if FLAGS.set == 'all':
            annotation_set = [f[:-4] for f in os.listdir(annotations_dir) if f.endswith('.xml')]
            image_set = [f[:-4] for f in os.listdir(images_dir) if f.endswith('.jpg') or f.endswith('.png')]
            miss_images = set(annotation_set) - set(image_set)
            print('miss images:', len(miss_images), miss_images)
            examples_list = sorted(list(set(annotation_set) & set(image_set)))
        else:
            examples_path = os.path.join(data_dir, year, 'ImageSets', 'Main',
                                         'aeroplane_' + FLAGS.set + '.txt')
            examples_list = dataset_util.read_examples_list(examples_path)
        for idx, example in enumerate(examples_list):
            if idx % 100 == 0:
                logging.info('On image %d of %d', idx, len(examples_list))
            path = os.path.join(annotations_dir, example + '.xml')
            with tf.gfile.GFile(path, 'r') as fid:
                xml_str = fid.read()
            xml = etree.fromstring(xml_str)
            data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
            tf_example = dict_to_tf_example(data, images_dir, label_map_dict,
                                            FLAGS.ignore_difficult_instances)
            if tf_example is not None:
                writer.write(tf_example.SerializeToString())
    writer.close()
    print('\t'.join(sorted(label_count.keys())))
    print('\t'.join(map(str, (label_count[k] for k in sorted(label_count.keys())))))

if __name__ == '__main__':
    tf.app.run()
