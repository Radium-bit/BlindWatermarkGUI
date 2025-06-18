## Copyright (c) 2025 Radium-bit
## SPDX-License-Identifier: Apache-2.0
## See LICENSE file for full terms
import os
import shutil
import tkinter as tk
from tkinter import messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD
import tempfile
import re
from PIL import Image
from blind_watermark import WaterMark

class App(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title("Blind‑Watermark GUI 工具")
        self.geometry("580x560")
        self.configure(bg="white")

        self.mode = tk.StringVar(value="embed")

        # 模式选择
        frm_mode = tk.Frame(self, bg="white")
        frm_mode.pack(pady=5)
        tk.Label(frm_mode, text="选择模式：", bg="white").pack(side="left")
        tk.Radiobutton(frm_mode, text="嵌入水印", variable=self.mode, value="embed", bg="white").pack(side="left")
        tk.Radiobutton(frm_mode, text="提取水印", variable=self.mode, value="extract", bg="white").pack(side="left")

        # 密码输入
        frm_pwd = tk.Frame(self, bg="white")
        frm_pwd.pack(pady=5, fill="x", padx=20)
        tk.Label(frm_pwd, text="密码（可空，默认为1234）：", bg="white").pack(side="left")
        self.entry_pwd = tk.Entry(frm_pwd)
        self.entry_pwd.insert(0, "")
        self.entry_pwd.pack(side="left", fill="x", expand=True)

        # 水印文本输入
        frm_wm = tk.Frame(self, bg="white")
        frm_wm.pack(pady=5, fill="both", padx=20)
        tk.Label(frm_wm, text="水印文本（仅嵌入时有效）：", bg="white").pack(anchor="w")
        self.text_wm = tk.Text(frm_wm, height=4)
        self.text_wm.pack(fill="both", expand=True)
        self.text_wm.insert("1.0", "Copyright@\nAuthor@")

        # 水印长度输入
        frm_len = tk.Frame(self, bg="white")
        frm_len.pack(pady=5, fill="x", padx=20)
        tk.Label(frm_len, text="提取水印长度（可空）：", bg="white").pack(side="left")
        self.entry_ws = tk.Entry(frm_len)
        self.entry_ws.insert(0, "")
        self.entry_ws.pack(side="left", fill="x", expand=True)

        # 原图尺寸输入
        frm_size = tk.Frame(self, bg="white")
        frm_size.pack(pady=5, fill="x", padx=20)
        tk.Label(frm_size, text="原图尺寸（如1920x1080，可空）：", bg="white").pack(side="left")
        self.entry_size = tk.Entry(frm_size)
        self.entry_size.insert(0, "")
        self.entry_size.pack(side="left", fill="x", expand=True)

        # 输出目录
        frm_out = tk.Frame(self, bg="white")
        frm_out.pack(pady=5, fill="x", padx=20)
        tk.Label(frm_out, text="输出目录（默认系统临时目录）：", bg="white").pack(side="left")
        self.entry_out = tk.Entry(frm_out)
        self.entry_out.insert(0, tempfile.gettempdir())
        self.entry_out.pack(side="left", fill="x", expand=True)

        # 拖拽区域
        lbl = tk.Label(self, text="请将图片拖入此区域", bg="#f0f0f0", fg="black", relief="ridge", borderwidth=2)
        lbl.pack(expand=True, fill="both", padx=20, pady=20)
        lbl.drop_target_register(DND_FILES)
        lbl.dnd_bind('<<Drop>>', self.on_drop)

    def on_drop(self, ev):
        for f in self.tk.splitlist(ev.data):
            f = f.strip('{}')
            try:
                if self.mode.get() == "embed":
                    self.embed_watermark(f)
                else:
                    self.extract_watermark(f)
            except Exception as e:
                messagebox.showerror("错误", str(e))

    def get_pwd(self):
        pwd = self.entry_pwd.get().strip()
        return pwd if pwd else "1234"

    def get_wm_text(self):
        return self.text_wm.get("1.0", "end").rstrip()

    def get_output_dir(self):
        path = self.entry_out.get().strip()
        if not path:
            path = tempfile.gettempdir()
        if not os.path.isdir(path):
            raise ValueError(f"输出目录不存在：{path}")
        return path

    def get_ws(self):
        val = self.entry_ws.get().strip()
        return int(val) if val.isdigit() else None

    def get_target_size(self):
        val = self.entry_size.get().strip()
        if re.match(r'^\d+x\d+$', val):
            w, h = map(int, val.lower().split('x'))
            return w, h
        return None

    def embed_watermark(self, filepath):
        output_dir = self.get_output_dir()
        os.makedirs(output_dir, exist_ok=True)

        name, ext = os.path.splitext(os.path.basename(filepath))
        image = Image.open(filepath)
        width, height = image.size

        tmp_in = os.path.join(output_dir, f"input{ext}")
        tmp_out = os.path.join(output_dir, f"output{ext}")
        image.save(tmp_in)

        wm_text = self.get_wm_text()
        pwd = self.get_pwd()

        bwm1 = WaterMark(password_img=int(pwd), password_wm=int(pwd))
        bwm1.read_img(tmp_in)
        bwm1.read_wm(wm_text, mode='str')
        bwm1.embed(tmp_out)

        wm_len = len(bwm1.wm_bit)
        dst_img = os.path.join(
            os.path.dirname(filepath),
            f"{name}-Watermark-ws{wm_len}-size{width}x{height}{ext}"
        )
        shutil.copy2(tmp_out, dst_img)

        messagebox.showinfo("嵌入成功", f"输出文件：\n{dst_img}\n水印长度：{wm_len} 尺寸：{width}x{height}")

        for f in [tmp_in, tmp_out]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass

    def extract_watermark(self, filepath):
        pwd = self.get_pwd()
        name = os.path.basename(filepath)
        ext = os.path.splitext(filepath)[1]

        output_dir = self.get_output_dir()
        tmp_in = os.path.join(output_dir, f"input.png")

        # 读取 ws
        wm_len = self.get_ws()
        if wm_len is None:
            m = re.search(r"ws(\d+)", name)
            if not m:
                raise ValueError("文件名中未找到 ws（如 ws256）\n可手动输入或改名")
            wm_len = int(m.group(1))

        # 读取原始尺寸
        target_size = self.get_target_size()
        if target_size is None:
            m = re.search(r"size(\d+)x(\d+)", name)
            if not m:
                raise ValueError("文件名中未找到原图尺寸（如 size1920x1080）\n可手动输入或改名")
            target_size = int(m.group(1)), int(m.group(2))

        # 判断是否需要 resize
        img = Image.open(filepath)
        if img.size == target_size:
            img.save(tmp_in)
        else:
            resized = img.resize(target_size, Image.LANCZOS) #使用LANCZOS算法重新匹配图像
            resized.save(tmp_in, format="PNG")

        bwm1 = WaterMark(password_img=int(pwd), password_wm=int(pwd))
        wm_extract = bwm1.extract(tmp_in, wm_shape=wm_len, mode='str')
        wm_extract = wm_extract.replace("\\n", "\n")
        messagebox.showinfo("提取成功", f"水印内容：\n{wm_extract}")

        if os.path.exists(tmp_in):
            try:
                os.remove(tmp_in)
            except:
                pass

if __name__ == "__main__":
    App().mainloop()
