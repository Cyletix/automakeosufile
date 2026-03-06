import tkinter as tk
from tkinter import ttk, font, messagebox
import math


# --- 核心计算逻辑 (与之前版本相同) ---
def get_fc_analysis(judgments):
    perfect_notes = judgments.get("max_plus", 0) + judgments.get("max", 0)
    great_notes = judgments.get("great", 0)
    good_notes = judgments.get("good", 0)
    ok_notes = judgments.get("ok", 0)
    miss_notes = judgments.get("miss", 0)
    non_perfect_notes = great_notes + good_notes + ok_notes + miss_notes
    total_notes = perfect_notes + non_perfect_notes

    if total_notes == 0:
        return "请输入有效的判定数量。"

    if non_perfect_notes == 0:
        return (
            "分析结果:\n"
            "------------------------------------\n"
            "非完美判定数量为 0。\n\n"
            "恭喜！这已经是一次完美的表现 (Full Combo)！"
        )

    try:
        p_all_perfect = math.exp(-non_perfect_notes)
        expected_tries = 1 / p_all_perfect
    except OverflowError:
        return "非完美判定数量过大，无法计算一个实际的概率（几乎为零）。"

    log_1_minus_p = math.log(1 - p_all_perfect) if p_all_perfect < 1 else -float("inf")

    if log_1_minus_p == 0:
        tries_50_percent = tries_90_percent = tries_99_percent = float("inf")
    else:
        tries_50_percent = math.log(0.5) / log_1_minus_p
        tries_90_percent = math.log(0.1) / log_1_minus_p
        tries_99_percent = math.log(0.01) / log_1_minus_p

    result = (
        f"分析结果:\n"
        f"------------------------------------\n"
        f"总音符数: {total_notes}\n"
        f"完美判定数 (Perfect): {perfect_notes}\n"
        f"非完美判定数 (Non-Perfect): {non_perfect_notes}\n\n"
        f"【核心估算】\n"
        f"单次尝试打出满分的概率: {p_all_perfect:.4%} ({p_all_perfect:.6f})\n"
        f"平均需要的尝试次数 (期望值): {expected_tries:.1f} ≈ {math.ceil(expected_tries)} 次\n\n"
        f"【成功率分析】\n"
        f"有 50% 的把握达成满分，大约需要: {math.ceil(tries_50_percent)} 次\n"
        f"有 90% 的把握达成满分，大约需要: {math.ceil(tries_90_percent)} 次\n"
        f"有 99% 的把握达成满分，大约需要: {math.ceil(tries_99_percent)} 次\n\n"
        f"【重要提示】\n"
        f"此模型假设每次失误是独立的随机事件。如果您的失误总是\n"
        f"集中在特定难点，针对性练习会比盲目重试更有效。"
    )
    return result


# --- 界面主题颜色定义 ---
LIGHT_THEME = {
    "background": "#DAD2FF",
    "panel": "#DAD2FF",
    "text": "#1E1E2E",
    "button": "#B2A5FF",
    "button_hover": "#9A8AE0",
    "button_active": "#493D9E",
    "highlight": "#FFF2AF",
    "entry_bg": "#FFFFFF",
    "border": "#B2A5FF",
}

DARK_THEME = {
    "background": "#1E1E2E",
    "panel": "#2E2E3E",
    "text": "#D8DEE9",
    "button": "#44475A",
    "button_hover": "#6272A4",
    "button_active": "#BD93F9",
    "highlight": "#BD93F9",
    "entry_bg": "#1E1E2E",
    "border": "#44475A",
}


# --- GUI 界面构建 ---
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("osu!mania 满分尝试次数计算器")
        self.geometry("620x550")  # 增加初始高度以容纳标题
        self.minsize(580, 500)  # 设置最小窗口尺寸

        self.dark_mode = False  # 默认亮色模式
        self.entries = {}

        self.setup_styles()
        self.create_widgets()
        self.update_theme()  # 应用初始主题

    def create_widgets(self):
        """创建所有界面控件"""
        # --- 主框架 ---
        main_frame = ttk.Frame(self, padding="0 10 15 15", style="App.TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)  # 让报告区域垂直扩展

        # --- 标题和主题切换按钮 ---
        header_frame = ttk.Frame(
            main_frame, style="App.TFrame", padding=(15, 10, 15, 0)
        )
        header_frame.grid(row=0, column=0, sticky="ew")
        header_frame.columnconfigure(0, weight=1)

        title_label = ttk.Label(
            header_frame, text="满分尝试次数计算器", style="Title.TLabel"
        )
        title_label.grid(row=0, column=0, sticky="w")

        self.theme_button = ttk.Button(
            header_frame, text="🌙", style="Theme.TButton", command=self.toggle_theme
        )
        self.theme_button.grid(row=0, column=1, sticky="e")

        # --- 输入区域 ---
        input_frame = ttk.LabelFrame(
            main_frame,
            text=" 输入单次最佳表现的判定 ",
            style="App.TLabelframe",
            padding=(0, 10, 0, 10),
        )
        input_frame.grid(row=1, column=0, sticky="ew", padx=15, pady=15)
        input_frame.columnconfigure(1, weight=1)

        judgments_to_create = {
            "max_plus": "最高判定 (300g / MAX+):",
            "max": "完美判定 (300 / MAX):",
            "great": "Great (200):",
            "good": "Good (100):",
            "ok": "OK (50):",
            "miss": "Miss:",
        }

        for i, (key, text) in enumerate(judgments_to_create.items()):
            label = ttk.Label(input_frame, text=text, style="Input.TLabel")
            label.grid(row=i, column=0, sticky="w", padx=(15, 10), pady=8)

            entry = ttk.Entry(
                input_frame, width=20, style="App.TEntry", justify="center"
            )
            entry.grid(row=i, column=1, sticky="e", padx=(10, 15), pady=8)
            self.entries[key] = entry

        # --- 分析报告区域 ---
        output_frame = ttk.LabelFrame(
            main_frame, text=" 分析报告 ", style="App.TLabelframe"
        )
        output_frame.grid(row=2, column=0, sticky="nsew", padx=15)
        output_frame.rowconfigure(0, weight=1)
        output_frame.columnconfigure(0, weight=1)

        self.result_text = tk.Text(
            output_frame,
            wrap=tk.WORD,
            height=10,
            font=("Microsoft YaHei", 10),
            relief=tk.FLAT,
            borderwidth=0,
        )
        self.result_text.grid(row=0, column=0, sticky="nsew", padx=10, pady=(5, 10))

        # --- 底部按钮区域 ---
        button_frame = ttk.Frame(
            main_frame, style="App.TFrame", padding=(15, 15, 15, 5)
        )
        button_frame.grid(row=3, column=0, sticky="ew")
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)

        self.calc_button = ttk.Button(
            button_frame,
            text="开始计算",
            command=self.calculate,
            style="App.TButton",
            padding=10,
        )
        self.calc_button.grid(row=0, column=0, sticky="ew", padx=(0, 5))

        self.clear_button = ttk.Button(
            button_frame,
            text="清空",
            command=self.clear_fields,
            style="App.TButton",
            padding=10,
        )
        self.clear_button.grid(row=0, column=1, sticky="ew", padx=(5, 0))

        # --- 填充默认值 ---
        self.set_default_values()

    def setup_styles(self):
        """仅在初始化时定义所有样式名称和布局"""
        self.style = ttk.Style(self)
        self.style.theme_use("clam")

        # 定义一次所有样式，之后只通过 configure 修改颜色
        self.style.configure("App.TFrame")
        self.style.configure("Title.TLabel", font=("Microsoft YaHei", 14, "bold"))
        self.style.configure("Input.TLabel", font=("Microsoft YaHei", 11))
        self.style.configure("App.TLabelframe", borderwidth=1, relief="solid")
        self.style.configure(
            "App.TLabelframe.Label", font=("Microsoft YaHei", 10, "bold")
        )
        self.style.configure(
            "App.TButton",
            font=("Microsoft YaHei", 11, "bold"),
            borderwidth=0,
            relief="flat",
        )
        self.style.configure(
            "App.TEntry", font=("Microsoft YaHei", 10), borderwidth=1, relief="solid"
        )
        self.style.configure(
            "Theme.TButton",
            font=("Arial", 14),
            borderwidth=0,
            relief="flat",
            focuscolor="",
        )

    def update_theme(self):
        """根据当前模式更新所有控件的颜色"""
        theme = DARK_THEME if self.dark_mode else LIGHT_THEME

        # 更新 ttk 控件样式
        self.style.configure("App.TFrame", background=theme["background"])
        self.style.configure(
            "Title.TLabel", background=theme["background"], foreground=theme["text"]
        )
        self.style.configure(
            "Input.TLabel", background=theme["panel"], foreground=theme["text"]
        )
        self.style.configure(
            "App.TLabelframe", background=theme["panel"], bordercolor=theme["border"]
        )
        self.style.configure(
            "App.TLabelframe.Label", background=theme["panel"], foreground=theme["text"]
        )
        self.style.map("App.TEntry", bordercolor=[("focus", theme["button_active"])])
        self.style.configure(
            "App.TEntry",
            fieldbackground=theme["entry_bg"],
            foreground=theme["text"],
            insertcolor=theme["text"],
            bordercolor=theme["border"],
        )
        self.style.configure(
            "App.TButton", background=theme["button"], foreground=theme["text"]
        )
        self.style.map(
            "App.TButton",
            background=[
                ("pressed", theme["button_active"]),
                ("active", theme["button_hover"]),
            ],
        )
        self.style.configure(
            "Theme.TButton", background=theme["background"], foreground=theme["text"]
        )
        self.style.map("Theme.TButton", background=[("active", theme["button_hover"])])

        # 更新非 ttk 控件 (根窗口和 Text 组件)
        self.configure(background=theme["background"])
        self.result_text.config(
            background=theme["panel"],
            foreground=theme["text"],
            selectbackground=theme["highlight"],
            selectforeground=theme["text"],
        )

        # 更新切换按钮的文本
        self.theme_button.config(text="☀️" if self.dark_mode else "🌙")

    def toggle_theme(self):
        """切换主题"""
        self.dark_mode = not self.dark_mode
        self.update_theme()

    def set_default_values(self):
        self.entries["max_plus"].insert(0, "3109")
        self.entries["max"].insert(0, "468")
        self.entries["great"].insert(0, "3")
        self.entries["good"].insert(0, "0")
        self.entries["ok"].insert(0, "0")
        self.entries["miss"].insert(0, "1")

    def calculate(self):
        judgments = {}
        for key, entry in self.entries.items():
            try:
                value = entry.get()
                judgments[key] = int(value) if value else 0
            except ValueError:
                messagebox.showerror(
                    "输入错误", f"'{entry.get()}' 不是一个有效的数字。"
                )
                return

        analysis_result = get_fc_analysis(judgments)
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, analysis_result)
        self.result_text.config(state=tk.DISABLED)

    def clear_fields(self):
        for entry in self.entries.values():
            entry.delete(0, tk.END)
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete("1.0", tk.END)
        self.result_text.config(state=tk.DISABLED)
        self.entries["max_plus"].focus()


if __name__ == "__main__":
    app = App()
    app.mainloop()
