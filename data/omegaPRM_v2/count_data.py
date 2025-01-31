import json
import os

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def count_metrics(node, paths=None, current_path=None):
    if paths is None:
        paths = set()
    if current_path is None:
        current_path = []
        
    current_path = current_path + [str(id(node))]
    
    children = node.get('children', [])
    paths.add(tuple(current_path))
    if not children:  # Leaf node
        # paths.add(tuple(current_path))
        mc_value = node.get('mc_value', 0)
        return 1, 1 if mc_value == 1.0 else 0, paths
        
    leaf_count = 0
    success_count = 0
    for child in children:
        child_leaf, child_success, _ = count_metrics(child, paths, current_path)
        leaf_count += child_leaf
        success_count += child_success
        
    return leaf_count, success_count, paths

def add_nodes_edges(dot, node, parent_id=None):
    node_id = str(id(node))
    text = node.get('text', '') + "'"  # Increased text length for better readability
    dot.node(node_id, label=text, shape='box', style='filled', fillcolor='lightblue', 
            fontname='helvetica', fontsize='10')
    
    if parent_id is not None:
        # Add mc_value as edge label in red
        mc_value = node.get('mc_value', 0)
        dot.edge(parent_id, node_id, label=f"{mc_value}", fontcolor="red", 
                fontname='helvetica', fontsize='20')
    
    for child in node.get('children', []):
        add_nodes_edges(dot, child, node_id)

def visualize_tree(json_data, name='mcts_tree', save_dir='mcts_trees'):
    os.makedirs(save_dir, exist_ok=True)
    dot = Digraph(comment='MCTS Tree', format='png')
    # Set default node attributes
    dot.attr('node', shape='box', style='filled', fillcolor='lightblue', 
            fontname='helvetica', fontsize='20')
    # Set default edge attributes
    dot.attr('edge', fontcolor='red', fontname='helvetica', fontsize='20')
    
    add_nodes_edges(dot, json_data)
    # Increase the spacing between nodes
    dot.attr(ranksep='1', nodesep='1')
    dot.render(os.path.join(save_dir, name), cleanup=True)
    
    # Count and print metrics
    leaf_count, success_count, paths = count_metrics(json_data)
    print(f"Number of leaf nodes: {leaf_count}")
    print(f"Number of successful leaf nodes (mc_value=1.0): {success_count}")
    print(f"Number of distinct paths: {len(paths)}")


from tqdm import tqdm

leaf_counts = []
success_counts = []
path_counts = []

failed_trees_count = 0

for j in range(11, 19):
    with open(f'/root/LLMreasoning/data/omegaPRM_v2/output_results_data/results_part_{j}.json/math-aps-v2.jsonl', 'r') as file:
        offset_idx = len(leaf_counts)
        lines = file.readlines()
        for i, line in enumerate(tqdm(lines, desc="Processing trees")):
            data = json.loads(line)
            # visualize_tree(data['reasoning_steps'], name=f'mcts_tree_{i+offset_idx}')
            leaf_count, success_count, paths = count_metrics(data['reasoning_steps'])

            if leaf_count == 1:
                failed_trees_count += 1
                continue
            
            leaf_counts.append(leaf_count)
            success_counts.append(success_count) 
            path_counts.append(len(paths))


print('number of terminating nodes', sum(leaf_counts))
print('number of unique paths', sum(path_counts))
print('number of failed trees', failed_trees_count)
success_rate = sum(1 for count in success_counts if count > 0) / (len(success_counts) + failed_trees_count)
print(f'Success rate: {success_rate:.2%}')