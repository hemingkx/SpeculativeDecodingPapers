import json


def get_content(jsonfile_path, output_path):
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


if __name__ == "__main__":
    jsonl_file = "../data/mt_bench/model_answer/vicuna-7b-v1.3-greedy.jsonl"
    output_file = "../data/mt_bench/model_answer/txt/vicuna-7b-v1.3-greedy.txt"
    get_content(jsonl_file, output_file)