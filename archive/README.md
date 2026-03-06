# Archive 目录 - 归档文件存储

此目录包含所有已归档的旧代码、实验性代码和临时分析文件。

## 归档内容说明

### 1. 原始项目备份
- `algorithm_original/` - 原algorithm目录内容（分散的算法实现）
- `archive_original/` - 原archive目录内容（旧版本代码和实验代码）
- `fileprocess_original/` - 原fileprocess目录内容（旧版本文件处理功能）

### 2. 旧版本主程序
- `main_old_version.py` - 原main.py（旧版本主程序）
- `main_old_experimental.py` - 原main_old.py（实验性代码）

### 3. 密度修正专项实验
- `密度修正专项/` - 独立的密度修正实验模块
  - `DensityFix.py` - 密度修正类
  - `FrequencyAnalyze.py` - 频率分析
  - `main.py` - 主程序入口
  - `OsuFile.py` - OSU文件处理
  - `OsuFileVisualize.py` - 可视化
  - `test_osufile.py` - 测试

### 4. 临时分析文件
- `analyze_unused_files.py` - 文件使用分析脚本
- `cleanup_plan.md` - 初始清理计划
- `project_analysis_report.md` - 项目分析报告
- `revised_cleanup_plan.md` - 修订版清理计划
- `test_cleaned_project.py` - 清理后项目测试
- `test_optimization.py` - 优化测试

## 归档原则

1. **功能重复** - 与automakeosufile模块功能重复的代码
2. **实验性代码** - 独立的实验模块
3. **临时分析** - 项目分析过程中创建的临时文件
4. **旧版本** - 已被新版本替代的代码

## 当前项目状态

所有核心功能已整合到 `automakeosufile/` 模块：
- 音频处理：`audio_processing.py`
- 特征提取：`feature_extraction.py`
- 谱面生成：`beatmap_generator.py`
- 配置管理：`config.py`
- 自动优化：`auto_optimization.py`
- OSU文件解析：`osu_parser.py`（从temp整合）
- 进化算法优化：`evolutionary_optimizer.py`（从temp整合）

## 恢复说明

如需恢复任何归档内容，请：
1. 检查归档文件的功能
2. 评估是否与当前项目兼容
3. 谨慎集成到现有代码中

**注意**：归档文件仅供参考和历史记录，生产代码请使用 `automakeosufile/` 模块。
