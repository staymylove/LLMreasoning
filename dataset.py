import json

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data


def process_steps(node):
    # 如果节点包含 text 字段，则按 \n\n 分割文本
    if 'text' in node:
        steps = node['text'].split('\n\n')
        node['steps'] = steps  # 保存分割后的步骤
        
        # 根据 mc_value 添加标签
        label = [1 if node['mc_value'] > 0 else 0] * len(steps)
        node['label'] = label  # 添加 label 字段

    # 如果有子节点，递归处理
    if 'children' in node:
        for child in node['children']:
            process_steps(child)

# 组合每个路径的 steps 和 label
def combine_steps_and_labels(node):
    # 处理子节点的组合路径
    paths_steps = []
    paths_labels = []

    # 递归遍历节点，组合 steps 和 label
    def recursive_combine(n, current_steps, current_labels):
        # 将当前节点的 steps 和 label 加入路径
        if 'steps' in n:
            current_steps.extend(n['steps'])
            current_labels.extend(n['label'])

        # 如果有子节点，则继续递归
        if 'children' in n:
            if len(n['children']) == 0:
                # 没有子节点，当前路径完成，保存到路径列表
                paths_steps.append(current_steps)
                paths_labels.append(current_labels)
            else:
                # 遍历所有子节点
                for child in n['children']:
                    recursive_combine(child, current_steps[:], current_labels[:])

    recursive_combine(node, [], [])

    # 保存组合后的结果
    node['combined_steps'] = paths_steps
    node['combined_label'] = paths_labels

def merge_list_to_paragraph(reasoning_steps, labels, j):
    # 提取从第 j 个元素到最后一个元素
    sublist = reasoning_steps[j:]
    labels = labels[j:]
    # 合并成一个段落，元素之间加上 " + \n\n"
    result = " + \n\n".join(sublist)
    result += " + \n\n"
    return result, labels



# 调用处理函数
# process_steps(data['reasoning_steps'])
# combine_steps_and_labels(data['reasoning_steps'])


# result_all=[]
# new_labels=[]
# # # 打印处理后的 data 字典
# for i in range(len(data['reasoning_steps']['combined_steps'])):
#     for j in range(len(data['reasoning_steps']['combined_steps'][i])):
    
#         if '<Thought>' in data['reasoning_steps']['combined_steps'][i][j]:
#             result, labels = merge_list_to_paragraph(data['reasoning_steps']['combined_steps'][i], data['reasoning_steps']['combined_label'][i], j)
#             result_all.append(result)
#             new_labels.append(labels)
            

# data['training_steps'] = result_all
# data['training_labels'] = new_labels     



file_path = '/data/zeju/reasoning/output_results_data/results_part_8.json/math-aps-v2.jsonl'
output_file_path = '/data/zeju/reasoning/output_results_data/results_part_8.json/training_dataset.jsonl'
data_all = read_jsonl(file_path)

for data in data_all:
    process_steps(data['reasoning_steps'])
    combine_steps_and_labels(data['reasoning_steps'])
    result_all=[]
    new_labels=[]
    for i in range(len(data['reasoning_steps']['combined_steps'])):
        for j in range(len(data['reasoning_steps']['combined_steps'][i])):
            if '<Thought>' in data['reasoning_steps']['combined_steps'][i][j]:
                result, labels = merge_list_to_paragraph(data['reasoning_steps']['combined_steps'][i], data['reasoning_steps']['combined_label'][i], j)
                result_all.append(result)
                new_labels.append(labels)
    data['process'] = result_all
    data['label'] = new_labels

print(data['label'])

with open(output_file_path, 'w', encoding='utf-8') as output_file:
    for entry in data_all:
        output_file.write(json.dumps(entry, ensure_ascii=False) + '\n')

