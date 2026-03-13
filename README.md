# AutoMakeosu Core

这是一个只保留核心算法的 Python 仓库，用于从音频生成 `osu!mania` 谱面，以及基于参考谱面做参数优化。

仓库不再包含主程序侧的音频、图片、界面或项目级运行脚本。对外推荐直接把它当作普通 Python 包引用：

```python
from automakeosu import generate_beatmap, optimize_beatmap
```

## 安装

开发模式安装：

```bash
pip install -e .
```

或安装最小依赖：

```bash
pip install -r requirements.txt
```

## 默认测试路径

当前默认测试资源指向：

- 音频：`D:\OneDrive\Project\CyletixMusicGame\songs\Scattered Rose\Scattered Rose.mp3`
- 参考谱面：`D:\OneDrive\Project\CyletixMusicGame\songs\Scattered Rose\Scattered Rose.osu`

默认生成目录是该歌曲目录下的子文件夹：

- `D:\OneDrive\Project\CyletixMusicGame\songs\Scattered Rose\automakeosu_generated\`

生成谱面时，音频会默认复制到输出目录，保证导出的 `.osu` 可以直接引用同目录下的音频文件。

## 用法

最小生成示例：

```python
from automakeosu import generate_beatmap

result = generate_beatmap(
    r"D:\OneDrive\Project\CyletixMusicGame\songs\Scattered Rose\Scattered Rose.mp3",
    columns=7,
)

print(result["output_path"])
```

指定输出目录：

```python
from automakeosu import generate_beatmap

result = generate_beatmap(
    r"D:\OneDrive\Project\CyletixMusicGame\songs\Scattered Rose\Scattered Rose.mp3",
    columns=7,
    output_dir=r"D:\OneDrive\Project\CyletixMusicGame\songs\Scattered Rose\custom_output",
)
```

参考谱面优化：

```python
from automakeosu import optimize_beatmap

best_result = optimize_beatmap(
    audio_path=r"D:\OneDrive\Project\CyletixMusicGame\songs\Scattered Rose\Scattered Rose.mp3",
    reference_osu_path=r"D:\OneDrive\Project\CyletixMusicGame\songs\Scattered Rose\Scattered Rose.osu",
    columns=7,
    rounds=2,
)

print(best_result["similarity"])
```

## 公开接口

- `automakeosu.generate_beatmap(...)`
- `automakeosu.optimize_beatmap(...)`
- `automakeosu.Config`
- `automakeosu.get_default_song_paths()`

## 仓库结构

```text
.
├── algorithm/      # 内部算法实现
├── automakeosu/    # 对外公共 API
├── pyproject.toml
├── requirements.txt
└── README.md
```
