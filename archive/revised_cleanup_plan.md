# 修订版清理计划

## 目标
1. 将所有需要清理的文件移动到archive目录（而不是backup）
2. 分析temp目录中的重要文件，将核心功能整合到对应包中
3. 清理密度修正专项（分析后决定是否保留）
4. 更新所有文档

## 分析结果

### 1. temp目录中的重要文件分析

**需要整合的核心功能：**
1. `osu_parser.py` - OSU文件解析器（重要，应整合到automakeosufile模块）
2. `evolutionary_optimizer.py` - 进化算法优化器（重要，应整合到automakeosufile模块）

**临时分析文件（可移动到archive）：**
1. `analyze_unused_files.py` - 文件使用分析
2. `cleanup_plan.md` - 清理计划
3. `project_analysis_report.md` - 项目分析报告
4. `test_cleaned_project.py` - 清理测试
5. `test_optimization.py` - 优化测试

### 2. 密度修正专项分析
需要检查每个文件的功能：
- `DensityFix.py` - 密度修正
- `FrequencyAnalyze.py` - 频率分析
- `main.py` - 主程序
- `OsuFile.py` - OSU文件处理
- `OsuFileVisualize.py` - 可视化
- `test_osufile.py` - 测试

### 3. 需要移动到archive目录的内容
1. 之前备份到backup的内容
2. 密度修正专项（如果确定不需要）
3. temp目录中的临时分析文件

## 执行步骤

### 步骤1: 恢复之前的备份到archive
```bash
# 将backup/20250305_cleanup/的内容移动到archive/
mv backup/20250305_cleanup/* archive/
rmdir backup/20250305_cleanup
```

### 步骤2: 分析并整合temp目录的核心文件
```bash
# 将osu_parser.py整合到automakeosufile模块
cp temp/osu_parser.py automakeosufile/osu_parser.py

# 将evolutionary_optimizer.py整合到automakeosufile模块
cp temp/evolutionary_optimizer.py automakeosufile/evolutionary_optimizer.py

# 更新automakeosufile/__init__.py添加新模块
```

### 步骤3: 分析密度修正专项
检查每个文件的功能，决定：
1. 如果功能重要且与主项目相关 → 整合到automakeosufile
2. 如果功能独立但有用 → 保留在密度修正专项
3. 如果功能重复或过时 → 移动到archive

### 步骤4: 清理temp目录
```bash
# 移动临时分析文件到archive
mv temp/analyze_unused_files.py archive/
mv temp/cleanup_plan.md archive/
mv temp/project_analysis_report.md archive/
mv temp/test_cleaned_project.py archive/
mv temp/test_optimization.py archive/

# 删除temp目录中的.pyc缓存文件
rm -rf temp/__pycache__/
```

### 步骤5: 更新文档
1. 更新README.md反映新的结构
2. 更新docs/中的文档
3. 在archive/中添加README说明所有归档内容

## 预期结构

清理后：
```
AutoMakeosuFile/
├── automakeosufile/          # 核心模块
│   ├── audio_processing.py
│   ├── feature_extraction.py
│   ├── beatmap_generator.py
│   ├── config.py
│   ├── main.py
│   ├── utils.py
│   ├── auto_optimization.py
│   ├── osu_parser.py          # 新增
│   └── evolutionary_optimizer.py # 新增
├── archive/                  # 所有归档文件
│   ├── algorithm_original/
│   ├── archive_original/
│   ├── fileprocess_original/
│   ├── main_old_version.py
│   ├── main_old_experimental.py
│   ├── 密度修正专项/          # 可能移动到这里
│   └── temp_analysis_files/  # 临时分析文件
├── temp/                     # 仅保留当前正在使用的临时文件
├── output/
├── picture/
├── audio/
├── docs/
├── algorithm/                # 空目录
├── fileprocess/              # 空目录
└── README.md
```

## 风险控制
1. 先备份再移动
2. 逐步执行，每步验证
3. 保持git历史可追溯
4. 测试整合后的功能