import json
import argparse
import os.path


def get_content(jsonfile_path="../data/mt_bench/model_answer/vicuna-7b-v1.3-pld-float32.jsonl",
                output_path="../data/mt_bench/model_answer/txt/vicuna-7b-v1.3-pld-float32.txt"):
    data = []
    with open(jsonfile_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)

    contents=[]
    for datapoint in data:
        turns=datapoint["choices"][0]['turns']
        contents.append(str(turns))

    with open(output_path, 'w') as file:
        for content in contents:
            file.write(content)
            file.write('\n')


def txt_compare(file_path1, file_path2):
    with open(file_path1, 'r', encoding='utf-8') as f1:
        lines1 = f1.readlines()
    with open(file_path2, 'r', encoding='utf-8') as f2:
        lines2 = f2.readlines()
    for l1, l2 in zip(lines1, lines2):
        if l1 != l2:
            return False
    return True


def run_compare(file_path, jsonfile1, jsonfile2):
    jsonfile_path1 = os.path.join(file_path, jsonfile1)
    jsonfile_path2 = os.path.join(file_path, jsonfile2)
    output_path1 = file_path + "txt/" + jsonfile1.replace("jsonl", "txt")
    output_path2 = file_path + "txt/" + jsonfile2.replace("jsonl", "txt")
    get_content(jsonfile_path1, output_path1)
    get_content(jsonfile_path2, output_path2)
    if txt_compare(output_path1, output_path2):
        print("Result totally Equal!")
    else:
        print("Not Equal!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--file-path",
        type=str,
        help="The file path of model answers.",
    )
    parser.add_argument(
        "--jsonfile1",
        type=str,
        help="The file name of the first evaluated method.",
    )
    parser.add_argument(
        "--jsonfile2",
        type=str,
        help="The file name of the second evaluated method.",
    )
    args = parser.parse_args()
    run_compare(args.file_path, args.jsonfile1, args.jsonfile2)