# AutoMakeosuFile 项目清理计划

## 清理目标
根据分析结果，清理不推荐使用的文件，统一项目结构，减少功能重复。

## 需要清理的文件分类

### 1. 主入口文件 (需要处理)
- [ ] `main.py` - 旧版本主程序，使用已废弃的模块
- [ ] `main_old.py` - 最旧版本，仅用于测试
- [ ] `密度修正专项/main.py` - 专项工具，可以保留但需要明确说明

### 2. 已废弃模块目录 (建议归档)
- [ ] `archive/` - 包含13个旧文件和实验代码
- [ ] `algorithm/` - 8个文件，功能已被automakeosufile模块替代
- [ ] `fileprocess/` - 3个文件，功能已被完全替代

### 3. 重复的OSU文件处理功能 (需要整合或删除)
- [ ] `fileprocess/osu_file_make.py` - 旧版本生成器
- [ ] `fileprocess/osu_file_parse.py` - 旧版本解析器  
- [ ] `temp/osu_parser.py` - 临时解析器
- [ ] `密度修正专项/OsuFile.py` - 专项解析器
- [ ] `密度修正专项/OsuFileVisualize.py` - 可视化工具
- [ ] `密度修正专项/test_osufile.py` - 测试文件

### 4. 其他重复功能
- [ ] `algorithm/binarize.py` - 旧版本二值化

## 清理策略

### 策略1: 归档备份 (推荐)
将不推荐的文件移动到专门的备份目录，而不是直接删除。

### 策略2: 重命名标记
将旧文件重命名为带有`_deprecated`或`_old`后缀，明确标识状态。

### 策略3: 功能整合
评估是否有有用功能可以整合到automakeosufile模块中。

## 具体执行步骤

### 步骤1: 创建备份目录
```
mkdir -p backup/20250305_cleanup
```

### 步骤2: 备份并标记旧的主文件
```
# 备份旧的主文件
mv main.py backup/20250305_cleanup/main_old_version.py
mv main_old.py backup/20250305_cleanup/main_old_experimental.py

# 创建说明文件
echo "这些是旧版本的主程序文件，已由automakeosufile/main.py替代" > backup/20250305_cleanup/README.txt
```

### 步骤3: 处理archive目录
```
# 整个目录备份
mv archive backup/20250305_cleanup/archive_original
```

### 步骤4: 处理algorithm和fileprocess目录
```
# 备份这些目录
mv algorithm backup/20250305_cleanup/algorithm_original
mv fileprocess backup/20250305_cleanup/fileprocess_original

# 创建空目录保持结构
mkdir algorithm
mkdir fileprocess
echo "# 此目录已清空，功能已迁移到automakeosufile模块" > algorithm/README.md
echo "# 此目录已清空，功能已迁移到automakeosufile模块" > fileprocess/README.md
```

### 步骤5: 更新根目录README
更新项目README.md，明确说明：
1. 当前使用automakeosufile模块
2. 旧代码已归档到backup目录
3. 如何运行当前版本

## 风险控制

1. **备份所有文件** - 不直接删除，先备份
2. **逐步执行** - 分步骤执行，每步验证
3. **保留git历史** - 所有更改通过git提交，可回退
4. **测试验证** - 清理后测试主要功能是否正常

## 预期效果

清理后项目结构：
```
AutoMakeosuFile/
├── automakeosufile/          # 唯一的主要代码目录
├── backup/                   # 备份的旧代码
├── temp/                     # 临时文件
├── output/                   # 输出文件
├── picture/                  # 图片文件
├── audio/                    # 音频文件
├── docs/                     # 文档
├── algorithm/                # 空目录，有README说明
├── fileprocess/              # 空目录，有README说明
└── README.md                 # 更新后的说明
```

## 验证检查清单

- [ ] automakeosufile/main.py 能正常运行
- [ ] 生成谱面功能正常
- [ ] 可视化功能正常
- [ ] 所有测试通过
- [ ] 文档已更新