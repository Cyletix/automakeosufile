"""已废弃：请改用 automakeosufile.parsers.osu_mania.parse_osu_file。"""

from automakeosufile.parsers.osu_mania import parse_osu_file


class OSUData:
    def __init__(self):
        self.parsed = None

    def read_from_file(self, filename):
        self.parsed = parse_osu_file(filename)
        raise RuntimeError(
            "fileprocess.osu_file_parse.OSUData 已废弃，请直接使用 automakeosufile.parsers.osu_mania.parse_osu_file"
        )

    def save_to_file(self, filename):
        raise RuntimeError(
            "旧版 OSUData.save_to_file 已废弃；当前阶段只保留 evaluation-first 主流程，不允许继续使用旧 parser 逻辑"
        )
