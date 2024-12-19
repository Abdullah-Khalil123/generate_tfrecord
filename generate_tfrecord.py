import os
import glob
import pandas as pd
import io
import xml.etree.ElementTree as ET
import argparse
from object_detection.utils import dataset_util

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow logging (1)
import tensorflow.compat.v1 as tf
from PIL import Image
from collections import namedtuple

# Initiate argument parser
parser = argparse.ArgumentParser(
    description="Sample TensorFlow XML-to-TFRecord converter"
)
parser.add_argument(
    "-x",
    "--xml_dir",
    help="Path to the folder where the input .xml files are stored.",
    type=str,
)
parser.add_argument(
    "-l", "--labels_path", help="Path to the labels (.pbtxt) file.", type=str
)
parser.add_argument(
    "-o", "--output_path", help="Path of output TFRecord (.record) file.", type=str
)
parser.add_argument(
    "-i",
    "--image_dir",
    help="Path to the folder where the input image files are stored. "
    "Defaults to the same directory as XML_DIR.",
    type=str,
    default=None,
)
parser.add_argument(
    "-c",
    "--csv_path",
    help="Path of output .csv file. If none provided, then no file will be " "written.",
    type=str,
    default=None,
)

args = parser.parse_args()

if args.image_dir is None:
    args.image_dir = args.xml_dir

# Manually define the class-to-integer mapping here
# Load the class-to-ID mapping from the .pbtxt file

def load_label_map(pbtxt_path):
    """Parses a .pbtxt file to create a class-to-ID mapping."""
    class_name_to_id = {}
    with open(pbtxt_path, "r") as file:
        lines = file.readlines()
        current_id = None
        current_name = None
        for line in lines:
            line = line.strip()
            if line.startswith("id:"):
                # Remove commas and convert to integer
                current_id = int(line.split(":")[1].strip().replace(",", ""))
            elif line.startswith("name:"):
                # Strip quotes from the name
                current_name = line.split(":")[1].strip().strip('"')
            if current_id is not None and current_name is not None:
                class_name_to_id[current_name] = current_id
                current_id = None
                current_name = None
    return class_name_to_id


class_name_to_id = load_label_map(args.labels_path)



def xml_to_csv(path):
    """Parses XML files to extract object annotation data and convert it into a Pandas DataFrame."""
    xml_list = []
    for xml_file in glob.glob(path + "/*.xml"):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall("object"):
            try:
                bndbox = member.find("bndbox")
                value = (
                    root.find("filename").text,                  # Image filename
                    int(root.find("size/width").text),           # Image width
                    int(root.find("size/height").text),          # Image height
                    member.find("name").text,                    # Object class name
                    int(bndbox.find("xmin").text),               # Bounding box xmin
                    int(bndbox.find("ymin").text),               # Bounding box ymin
                    int(bndbox.find("xmax").text),               # Bounding box xmax
                    int(bndbox.find("ymax").text),               # Bounding box ymax
                )
                xml_list.append(value)
            except AttributeError as e:
                print(f"Error parsing file {xml_file}: {e}")
    column_name = [
        "filename",
        "width",
        "height",
        "class",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
    ]
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df



def class_text_to_int(row_label):
    """Manually converts class name to class integer based on the defined class_name_to_id mapping."""
    return class_name_to_id.get(row_label, -1)  # Returns -1 if class not found


def split(df, group):
    data = namedtuple("data", ["filename", "object"])
    gb = df.groupby(group)
    return [
        data(filename, gb.get_group(x))
        for filename, x in zip(gb.groups.keys(), gb.groups)
    ]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, "{}".format(group.filename)), "rb") as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode("utf8")
    image_format = b"jpg"
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row["xmin"] / width)
        xmaxs.append(row["xmax"] / width)
        ymins.append(row["ymin"] / height)
        ymaxs.append(row["ymax"] / height)
        classes_text.append(row["class"].encode("utf8"))
        classes.append(class_text_to_int(row["class"]))

    tf_example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "image/height": dataset_util.int64_feature(height),
                "image/width": dataset_util.int64_feature(width),
                "image/filename": dataset_util.bytes_feature(filename),
                "image/source_id": dataset_util.bytes_feature(filename),
                "image/encoded": dataset_util.bytes_feature(encoded_jpg),
                "image/format": dataset_util.bytes_feature(image_format),
                "image/object/bbox/xmin": dataset_util.float_list_feature(xmins),
                "image/object/bbox/xmax": dataset_util.float_list_feature(xmaxs),
                "image/object/bbox/ymin": dataset_util.float_list_feature(ymins),
                "image/object/bbox/ymax": dataset_util.float_list_feature(ymaxs),
                "image/object/class/text": dataset_util.bytes_list_feature(
                    classes_text
                ),
                "image/object/class/label": dataset_util.int64_list_feature(classes),
            }
        )
    )
    return tf_example


def main(_):

    writer = tf.python_io.TFRecordWriter(args.output_path)
    path = os.path.join(args.image_dir)
    examples = xml_to_csv(args.xml_dir)
    grouped = split(examples, "filename")
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())
    writer.close()
    print("Successfully created the TFRecord file: {}".format(args.output_path))
    if args.csv_path is not None:
        examples.to_csv(args.csv_path, index=None)
        print("Successfully created the CSV file: {}".format(args.csv_path))


if __name__ == "__main__":
    tf.app.run()
