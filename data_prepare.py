import json

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

# Example usage
file_path = '/data/zeju/LLMreasoning/output_results_data/results_part_8.json/training_dataset.jsonl'
output_file_path = '/data/zeju/LLMreasoning/output_results_data/results_part_8.json/training_dataset_new.jsonl'

data_all = read_jsonl(file_path)

# output_file_path = '/data/zeju/ReST-MCTS/test_data.json'
# with open(output_file_path, 'w', encoding='utf-8') as output_file:
#     json.dump(data[0], output_file, ensure_ascii=False, indent=4)

# import json

# 定义递归函数，遍历每一步并更新 text
# def update_text(node):
#     # 检查 mc_value 并修改 text
#     if node['mc_value'] > 0:
#         node['text'] += ' +'
#     else:
#         node['text'] += ' -'
    
#     # 递归处理子节点
#     if 'children' in node and node['children']:
#         for child in node['children']:
#             update_text(child)




# for i in range(len(data)):
    
# # 递归更新根节点以及所有子节点的 text
#     update_text(data[i]['reasoning_steps'])




# with open(output_file_path, 'w', encoding='utf-8') as output_file:
#     for entry in data:
#         output_file.write(json.dumps(entry, ensure_ascii=False) + '\n')

import json
result = []
# 输入字典
for data in data_all:
    

# 结果列表，用于存储拆解后的字典
    

    # 遍历 `step` 和 `label`，将其元素拆分并重组成新的字典
    for s, l in zip(data['process'], data['label']):
        result.append({
            'question': data['question'], 
            'question_id': data['question_id'],# 保持原来的 question
            'process': s,                      # 对应的 step
            'label': l                      # 对应的 label
        })

    # 将拆解后的数据写回到一个 JSONL 文件


with open(output_file_path, 'w') as f:
    for item in result:
        json.dump(item, f)  # 将每个字典转为 JSON 格式并写入文件
        f.write('\n')        # 每个字典写入一行

# print(f"Data has been written to {output_file_path}")

data_new = read_jsonl(output_file_path)
# print(data_new[0])


