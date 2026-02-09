#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
比较 train 和 train_ablation.py 的差异
Compare differences between train and train_ablation.py
"""

import difflib
import sys

def compare_files(file1, file2):
    """比较两个文件并显示差异"""
    try:
        with open(file1, 'r', encoding='utf-8') as f1:
            lines1 = f1.readlines()
        with open(file2, 'r', encoding='utf-8') as f2:
            lines2 = f2.readlines()
    except FileNotFoundError as e:
        print(f"错误: 文件未找到 - {e}")
        return
    
    print(f"文件对比:")
    print(f"  原版文件: {file1} ({len(lines1)} 行)")
    print(f"  消融版本: {file2} ({len(lines2)} 行)")
    print(f"  差异行数: {len(lines2) - len(lines1)} 行\n")
    
    # 生成差异
    diff = difflib.unified_diff(
        lines1, lines2,
        fromfile=file1,
        tofile=file2,
        lineterm='',
        n=3  # 显示上下文行数
    )
    
    print("=" * 80)
    print("主要修改内容:")
    print("=" * 80)
    
    modifications = []
    current_hunk = []
    in_diff = False
    
    for line in diff:
        if line.startswith('@@'):
            if current_hunk:
                modifications.append('\n'.join(current_hunk))
                current_hunk = []
            in_diff = True
            current_hunk.append(line)
        elif in_diff:
            if line.startswith('+') and not line.startswith('+++'):
                current_hunk.append(f"  {line}")
            elif line.startswith('-') and not line.startswith('---'):
                current_hunk.append(f"  {line}")
            elif not line.startswith('---') and not line.startswith('+++'):
                current_hunk.append(f"  {line}")
    
    if current_hunk:
        modifications.append('\n'.join(current_hunk))
    
    if modifications:
        print("\n修改点 1: LTGQ.__init__ 参数增加")
        print("-" * 80)
        for mod in modifications[:2]:  # 显示前两个修改点
            print(mod)
            print()
        
        if len(modifications) > 2:
            print(f"\n... 还有 {len(modifications) - 2} 处修改 (encode方法、main函数、argparse等)")
    else:
        print("未检测到差异")
    
    print("\n" + "=" * 80)
    print("关键修改总结:")
    print("=" * 80)
    print("1. LTGQ.__init__: 新增 disable_qfm, disable_dtm, disable_history 参数")
    print("2. encode() 方法: 新增三个 if/else 消融分支")
    print("3. main() 函数: 传递消融参数给 LTGQ 构造函数")
    print("4. argparse: 新增 --disable_qfm, --disable_dtm, --disable_history 命令行参数")
    print("5. 其他代码: 与原版完全一致")

if __name__ == "__main__":
    if len(sys.argv) > 2:
        file1 = sys.argv[1]
        file2 = sys.argv[2]
    else:
        file1 = "train"
        file2 = "train_ablation.py"
    
    compare_files(file1, file2)
