"""已废弃：旧版 generator 会误导到非 mania 主流程。"""


class OSUGenerator:
    def __init__(self):
        raise RuntimeError(
            "fileprocess.osu_file_make.OSUGenerator 已废弃；当前阶段先完成 evaluation-first 流程，后续再实现新的 mania generator"
        )
