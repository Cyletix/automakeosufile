# 项目清理计划

## 目标
1. 删除重复和无用文件
2. 保留有用的旧代码作为参考
3. 整理文件夹结构
4. 更新README.md

## 文件分析

### 1. 新模块 vs 旧模块对比

#### 音频处理
- ✅ **新**: `automakeosufile/audio_processing.py` - 完整实现（Mel频谱 + 自适应二值化）
- ❌ **旧**: `algorithm/binarize.py` - 简单二值化（固定阈值0.9）
- ❌ **旧**: `algorithm/custom_stft.py` - 简单STFT
- ❌ **旧**: `algorithm/dynamic_spectrum.py` - 实验代码

#### BPM计算
- ✅ **新**: `automakeosufile/feature_extraction.py` - 完整BPM检测
- ❌ **旧**: `algorithm/bpm_calculate.py` - 简单BPM计算

#### 谱面生成
- ✅ **新**: `automakeosufile/beatmap_generator.py` - 完整.osu生成
- ❌ **旧**: `fileprocess/osu_file_make.py` - 简单生成

#### 特征提取
- ✅ **新**: `automakeosufile/feature_extraction.py` - 完整特征提取
- ❌ **旧**: `algorithm/custom_onset_detect.py` - 简单onset检测
- ❌ **旧**: `algorithm/note_duration.py` - 未完成
- ❌ **旧**: `algorithm/windows_size.py` - 简单窗口计算

### 2. 需要保留的文件

#### 核心文件
- `automakeosufile/` - 新模块化包（保留）
- `main.py` - 旧的主程序（保留作为参考）
- `main_old.py` - 更旧的版本（保留作为历史）

#### 工具文件
- `fileprocess/mp3_to_wav.py` - MP3转WAV（有用）
- `fileprocess/osu_file_parse.py` - OSU文件解析（有用）
- `密度修正专项/` - 密度修正实验（有用）

#### 测试文件
- `test_new_algorithm.py` - 新算法测试（有用）
- `test/` - 测试文件（保留）

#### 文档
- `README.md` - 项目说明（需要更新）
- `docs/` - 设计文档（保留）
- `IMPROVEMENTS_SUMMARY.md` - 改进总结（保留）

### 3. 可以删除的文件

#### algorithm/目录
- `algorithm/边缘检测.py` - 实验代码
- `algorithm/chatgpt_generate_function.py` - ChatGPT生成
- `algorithm/PCA_test.py` - PCA测试
- `algorithm/svd.py` - SVD测试
- `algorithm/HandleData.java` - Java版本
- `algorithm/HandleData.py` - Python版本

#### fileprocess/目录
- `fileprocess/osu_file_make_test.py` - 测试文件
- `fileprocess/readmp3.py` - 冗余

#### plotfunction/目录
- `plotfunction/plot_animation.py` - 未使用
- `plotfunction/stft_plotly.py` - 未使用

#### 其他
- `librosa_example.py` - 示例
- `P_fullscore_calculater.py` - 未使用
- `test_import.py` - 测试
- `test_moviepy_import.py` - 测试
- `test_chroma_b.npy` - 数据文件
- `test_chroma.npy` - 数据文件

### 4. 清理步骤

#### 第一步：备份重要文件
1. 创建`archive/`目录
2. 将可以删除但可能有参考价值的文件移动到`archive/`

#### 第二步：删除无用文件
1. 删除algorithm/目录中的冗余文件
2. 删除fileprocess/目录中的冗余文件
3. 删除plotfunction/目录
4. 删除其他无用文件

#### 第三步：整理文件夹结构
```
AutoMakeosuFile/
├── automakeosufile/     # 新模块化包
├── archive/            # 归档的旧代码
├── audio/              # 音频文件
├── docs/               # 文档
├── output/             # 输出文件
├── picture/            # 图片
├── test/               # 测试文件
├── 密度修正专项/        # 密度修正实验
├── main.py             # 旧主程序（参考）
├── main_old.py         # 更旧版本（历史）
├── README.md           # 项目说明
└── requirements.txt    # 依赖
```

#### 第四步：更新README.md
1. 添加新模块的使用说明
2. 更新项目结构
3. 添加清理说明

### 5. 执行计划

1. **先移动，后删除** - 将文件移动到archive/再决定是否删除
2. **保留历史** - 重要的旧代码保留在archive/中
3. **更新文档** - 确保README反映最新状态
4. **测试功能** - 清理后测试核心功能是否正常