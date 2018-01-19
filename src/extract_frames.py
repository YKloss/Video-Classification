import os
import re
import glob
from subprocess import call
import csv
import pandas as pd

DATA_DIR = os.path.join("data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "hmdb")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
SPLITS_DIR = os.path.join(DATA_DIR, "testTrainMulti_7030_splits")
DATA_FILE = os.path.join(DATA_DIR, "data_files.csv")


def extract_frames():

    splits = parse_split_files()
    data_file = []

    for class_name in splits:

        # Create directorys for the class, if they don't already exist
        if not os.path.exists(os.path.join(TRAIN_DIR, class_name)):
            os.makedirs(os.path.join(TRAIN_DIR, class_name))

        if not os.path.exists(os.path.join(TEST_DIR, class_name)):
            os.makedirs(os.path.join(TEST_DIR, class_name))

        for train_file in splits[class_name]["train"]:
            src = os.path.join(RAW_DATA_DIR, class_name, train_file + ".avi")
            dest = os.path.join(TRAIN_DIR, class_name, train_file + "-%04d.jpg")
            call(["ffmpeg", "-i", src, dest])
            frame_count = get_number_of_frames(os.path.join(TRAIN_DIR, class_name), train_file)
            data_file.append(["train", class_name, train_file, frame_count])
            print("Extracted " + str(frame_count) + " frames from video " + str(train_file))

        for test_file in splits[class_name]["test"]:
            src = os.path.join(RAW_DATA_DIR, class_name, test_file + ".avi")
            dest = os.path.join(TEST_DIR, class_name, test_file + "-%04d.jpg")
            call(["ffmpeg", "-i", src, dest])
            frame_count = get_number_of_frames(os.path.join(TEST_DIR, class_name), test_file)
            data_file.append(["test", class_name, test_file, frame_count])
            print("Extracted " + str(frame_count) + " frames from video " + str(test_file))

    with open(DATA_FILE, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(data_file)

    # df = pd.read_csv(os.path.join(DATA_DIR, "data_files.csv"), skip_blank_lines=True)
    # df.to_csv(os.path.join(DATA_DIR, "data_files_new.csv"), index=False)


def get_number_of_frames(class_path, video_name):
    frames = glob.glob(os.path.join(class_path, video_name + '*.jpg'))
    return len(frames)


def parse_split_files(split_id=1):

    split_file_regex = re.compile('.*?(' + str(split_id) + ')\.(txt)')
    split_groups = {}

    for split_file in os.listdir(SPLITS_DIR):
        if not split_file_regex.match(split_file):
            continue

        classname = split_file.split("_test")[0]

        split_groups[classname] = {"train": [], "test": []}
        with open(os.path.join(SPLITS_DIR, split_file)) as fin:
            for line in fin:
                line_parts = line.split(".avi")
                filename = line_parts[0]

                if line_parts[1].strip() == "1":
                    split_groups[classname]["train"].append(filename)
                elif line_parts[1].strip() == "2":
                    split_groups[classname]["test"].append(filename)
                else:
                    pass

    return split_groups


def main():
    extract_frames()

if __name__ == '__main__':
    main()
