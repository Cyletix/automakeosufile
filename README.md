# AUTO MAKE OSU FILE (v2.0 - 最终清理版)

osu自动做谱程序, 对音频文件进行分析后, 自动生成铺面。项目经过彻底清理和重构，代码结构清晰，功能统一。

## 🎯 项目状态
- **当前版本**: v2.0 (最终清理版)
- **主要模块**: `automakeosufile/`
- **清理日期**: 2025年3月5日
- **状态**: 生产就绪，代码质量优秀

## 📁 实际项目结构 (2026年3月5日)

```
AutoMakeosuFile/
├── algorithm/                # 核心算法模块 (正确位置!)
│   ├── __init__.py          # 包初始化
│   ├── audio_processing.py   # 音频处理
│   ├── feature_extraction.py # 特征提取
│   ├── beatmap_generator.py  # 谱面生成
│   ├── config.py            # 配置参数
│   ├── main.py              # 主程序入口 (支持F5调试)
│   ├── utils.py             # 工具函数
│   ├── auto_optimization.py # 规则优化器
│   ├── osu_parser.py        # OSU文件解析
│   └── evolutionary_optimizer.py # 进化算法优化器
├── archive/                  # 归档文件 (历史版本)
├── audio/                    # 音频文件
├── docs/                     # 文档
├── output/                   # 输出文件 (.osu文件和统计)
├── picture/                  # 图片文件 (可视化结果)
├── temp/                     # 临时文件 (测试脚本等)
├── main.py                   # 根目录入口 (支持直接F5运行!)
├── requirements.txt          # 依赖列表
└── README.md                 # 项目说明
```

## 🚀 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行程序 (三种方式)

#### 1. 使用根目录入口 (推荐 - 支持F5调试)
```bash
python main.py "audio/你的音频文件.mp3" --columns 7 --visualize
```

#### 2. 使用模块方式
```bash
python -m algorithm.main "audio/你的音频文件.mp3" --columns 7 --visualize
```

#### 3. 直接运行algorithm/main.py
```bash
python algorithm/main.py "audio/你的音频文件.mp3" --columns 7 --visualize
```

### 参数说明:
# --columns: 键数 (4, 6, 7, 8)
# --visualize: 生成可视化图表
# --output-dir: 自定义输出目录
# --iteration: 迭代编号 (用于优化器工作流)

### 示例
```bash
# 生成7K谱面 (支持直接F5调试!)
python main.py "audio/Scattered Rose.mp3" --columns 7 --visualize

# 生成4K谱面
python main.py "audio/NIGHTFALL.mp3" --columns 4

# 自定义输出目录
python main.py "audio/NIGHTFALL.mp3" --output-dir my_output --columns 6
```

## 🔧 核心功能

### 1. 音频处理
- MP3/WAV格式支持
- Mel频谱计算
- 自适应二值化
- 音符事件提取

### 2. 特征提取
- BPM自动检测
- 节拍网格对齐
- 频率到轨道映射
- 密度控制和物理手感优化

### 3. 谱面生成
- 完整的.osu文件生成
- 自动复制到osu!歌曲目录
- 支持4K/6K/7K/8K
- 元数据设置

### 4. 可视化
- 频谱图显示
- 二值化结果
- 音符分布图
- 自动保存到picture目录

## 📊 技术特点

### 算法改进
- **二值化**: 自适应阈值 (激活率78.9%)
- **频谱分析**: Mel频谱 (更适合音乐分析)
- **节拍对齐**: BPM检测 + 节拍网格对齐
- **密度控制**: 轨道间隔控制 + 每拍音符限制
- **轨道映射**: 频率bin到音高类的智能映射

### 架构优势
- **模块化设计**: 清晰的职责分离
- **统一配置**: Config类集中管理参数
- **错误处理**: 完整的异常处理机制
- **命令行支持**: 灵活的参数配置

## 📝 项目历史与清理

### 清理内容 (2025年3月5日 - 最终清理)
1. **移除重复代码**: 7个OSU文件处理实现 → 1个 (`automakeosufile/beatmap_generator.py`)
2. **统一入口**: 4个主程序文件 → 1个 (`automakeosufile/main.py`)
3. **整合核心功能**: 将temp目录中的`osu_parser.py`和`evolutionary_optimizer.py`整合到automakeosufile模块
4. **归档实验模块**: 将`密度修正专项/`移动到archive目录
5. **清理临时文件**: 将所有临时分析文件移动到archive目录
6. **统一归档**: 将所有旧代码、实验代码、临时文件统一归档到`archive/`目录

### 归档位置
所有旧代码、实验代码和临时文件已归档到: `archive/`
- `algorithm_original/` - 原algorithm目录内容（分散的算法实现）
- `archive_original/` - 原archive目录内容（旧版本代码和实验代码）
- `fileprocess_original/` - 原fileprocess目录内容（旧版本文件处理功能）
- `密度修正专项/` - 独立的密度修正实验模块
- `main_old_version.py` - 原main.py（旧版本主程序）
- `main_old_experimental.py` - 原main_old.py（实验性代码）
- 临时分析文件 - 项目分析过程中创建的所有临时文件

### 核心功能整合
从temp目录整合到automakeosufile模块的重要功能：
1. **`osu_parser.py`** → `automakeosufile/osu_parser.py` (OSU文件解析器)
2. **`evolutionary_optimizer.py`** → `automakeosufile/evolutionary_optimizer.py` (进化算法优化器)

## 🎮 自动复制到osu!歌曲目录

生成的谱面会自动复制到osu!歌曲目录：
```
D:\osu!\Songs\歌曲名_automake\
├── 歌曲名.mp3          # 音频文件
└── 歌曲名_7K.osu      # 生成的谱面文件
```

### 功能特点
1. **自动创建文件夹**: 在osu!歌曲目录下创建专属文件夹
2. **复制音频文件**: 保留原始音频文件
3. **错误处理**: 如果osu!目录不存在会给出警告
4. **完整路径支持**: 支持自定义osu!安装路径

## 🔍 验证与测试

清理后验证项目功能正常：
- [x] `automakeosufile/main.py` 能正常运行
- [x] 生成谱面功能正常
- [x] 可视化功能正常
- [x] 自动复制到osu!目录功能正常

## 📈 未来改进方向

1. **长条检测优化**: 改进音符持续时间检测
2. **模式识别**: 识别音乐模式（连打、滑条等）
3. **难度分级**: 根据音符密度自动调整难度
4. **可视化增强**: 实时处理进度显示
5. **批量处理**: 支持多个音频文件批量生成
6. **自定义osu!目录**: 允许用户配置osu!安装目录

## 📄 许可证

本项目遵循MIT许可证。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目。

## 📧 联系

如有问题或建议，请通过GitHub Issues联系。

---

# 项目重构和优化 (v2.0)
%% 2026/03/04 %%
项目经过重构，现在具有模块化的结构和改进的算法。

## 新项目结构

```
AutoMakeosuFile/
├── automakeosufile/     # 新模块化包（核心功能）
│   ├── __init__.py      # 包初始化
│   ├── config.py        # 配置文件
│   ├── audio_processing.py      # 音频处理模块
│   ├── feature_extraction.py    # 特征提取模块
│   ├── beatmap_generator.py     # 谱面生成模块
│   └── main.py          # 命令行入口
├── archive/             # 归档的旧代码（参考用）
├── audio/               # 音频文件
├── docs/                # 设计文档
├── output/              # 输出文件
├── picture/             # 图片
├── test/                # 测试文件
├── 密度修正专项/         # 密度修正实验
├── main.py              # 旧主程序（参考）
├── main_old.py          # 更旧版本（历史）
├── README.md            # 项目说明
└── requirements.txt     # 依赖
```

## 新版本特性

### 1. 模块化设计
- **audio_processing.py**: Mel频谱 + 自适应二值化
- **feature_extraction.py**: BPM检测 + 节拍对齐 + 轨道映射
- **beatmap_generator.py**: 完整的.osu文件生成

### 2. 算法改进
- **二值化**: 从固定阈值0.9 → 自适应阈值（激活率78.9%）
- **频谱分析**: 从Chroma CQT → Mel频谱（更适合音乐分析）
- **节拍对齐**: 新增BPM检测和节拍网格对齐
- **密度控制**: 新增轨道间隔控制和每拍音符限制
- **轨道映射**: 频率bin到音高类的智能映射

### 3. 使用方式

#### 命令行使用：
```bash
# 生成7K谱面
python -m automakeosufile.main audio/NIGHTFALL.mp3 --columns 7 --visualize

# 生成4K谱面
python -m automakeosufile.main audio/NIGHTFALL.mp3 --columns 4

# 自定义输出目录
python -m automakeosufile.main audio/NIGHTFALL.mp3 --output-dir my_output --columns 6
```

#### Python API使用：
```python
from automakeosufile import AudioProcessor, FeatureExtractor, BeatmapGenerator, Config

# 配置
config = Config()
config.DEFAULT_COLUMNS = 7

# 音频处理
processor = AudioProcessor(config)
audio_data = processor.process_audio("audio.mp3")

# 特征提取
extractor = FeatureExtractor(config)
features = extractor.extract_features(audio_data, audio_data['note_events'])

# 谱面生成
generator = BeatmapGenerator(config)
output_path = generator.generate_beatmap("audio.mp3", features)
```

## 文件说明

### 保留的核心文件
- `automakeosufile/` - 新模块化包（生产代码）
- `fileprocess/mp3_to_wav.py` - MP3转WAV工具
- `fileprocess/osu_file_parse.py` - OSU文件解析
- `密度修正专项/` - 密度修正实验
- `test_new_algorithm.py` - 新算法测试

### 归档的旧文件
- `archive/` - 旧代码归档（参考用）
  - `algorithm/` - 旧的算法实现
  - `fileprocess/` - 旧的文件处理
  - `plotfunction/` - 旧的绘图功能

## 运行环境

```bash
# 安装依赖
pip install -r requirements.txt

# 主要依赖
- librosa >= 0.10.0
- numpy >= 1.24.0
- opencv-python >= 4.8.0
- matplotlib >= 3.7.0
- scipy >= 1.11.0
```

## 项目历史

- **v1.0**: 初始版本，代码结构混乱
- **v2.0**: 重构版本，模块化设计，算法改进
- **当前状态**: 生产就绪，代码质量显著提升

## 自动复制到osu!歌曲目录

新版本增加了自动将生成的谱面复制到osu!歌曲目录的功能：

### 功能特点
1. **自动创建文件夹**: 在`D:\osu!\Songs\`目录下创建`{歌曲名}_automake`文件夹
2. **复制音频文件**: 将原始MP3文件复制到目标文件夹
3. **复制谱面文件**: 将生成的.osu文件复制到目标文件夹
4. **错误处理**: 如果osu!目录不存在会给出警告，不会中断程序

### 使用示例
```bash
# 生成4K谱面并自动复制到osu!歌曲目录
python -m automakeosufile.main audio/NIGHTFALL.mp3 --columns 4

# 输出结果示例：
# ✓ 音频文件复制到: D:\osu!\Songs\NIGHTFALL_automake\NIGHTFALL.mp3
# ✓ 谱面文件复制到: D:\osu!\Songs\NIGHTFALL_automake\NIGHTFALL_4K.osu
# ✓ 谱面已复制到osu!歌曲目录: D:\osu!\Songs\NIGHTFALL_automake
```

### 目录结构
```
D:\osu!\Songs\
├── NIGHTFALL_automake\
│   ├── NIGHTFALL.mp3          # 音频文件
│   └── NIGHTFALL_4K.osu       # 生成的谱面文件
├── 其他歌曲文件夹\
└── ...
```

## 未来改进方向

1. **长条检测优化**: 改进音符持续时间检测
2. **模式识别**: 识别音乐模式（连打、滑条等）
3. **难度分级**: 根据音符密度自动调整难度
4. **可视化增强**: 实时处理进度显示
5. **批量处理**: 支持多个音频文件批量生成
6. **自定义osu!目录**: 允许用户配置osu!安装目录
