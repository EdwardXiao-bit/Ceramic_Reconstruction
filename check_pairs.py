import json

# 读取测试结果
with open('results/boundary_validation_test.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 获取当前测试对
current_pairs = set()
for pair in data['test_info']['test_pairs']:
    current_pairs.add(tuple(sorted(pair)))

print("当前测试对:")
for pair in sorted(current_pairs):
    print(f"  {pair[0]}-{pair[1]}")
print(f"总共: {len(current_pairs)} 对")

print("\n应该有的10对:")
all_pairs = set()
for i in range(5):
    for j in range(i+1, 5):
        all_pairs.add((i, j))
        print(f"  {i}-{j}")

print(f"\n缺失的对:")
missing = all_pairs - current_pairs
for pair in sorted(missing):
    print(f"  {pair[0]}-{pair[1]}")

print(f"\n重复的对:")
duplicates = [pair for pair in data['test_info']['test_pairs'] if data['test_info']['test_pairs'].count(pair) > 1]
for pair in duplicates:
    print(f"  {pair[0]}-{pair[1]}")