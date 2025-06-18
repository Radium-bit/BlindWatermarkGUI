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
import json
from pathlib import Path
from PIL import Image
from blind_watermark import WaterMark

# 程序版本号
VERSION = r"0.1.1"

class App(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.config_path = os.path.join(os.environ["USERPROFILE"], "radiumbit.blindwatermark.config.json")
        self._saved_before_close = False
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.title(f"BlindWatermarkGUI v{VERSION}")
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
        self.entry_pwd = tk.Entry(frm_pwd, show="*")
        self.entry_pwd.insert(0, "")
        self.entry_pwd.pack(side="left", fill="x", expand=True)

        # 水印文本输入
        frm_wm = tk.Frame(self, bg="white")
        frm_wm.pack(pady=5, fill="both", padx=20)
        tk.Label(frm_wm, text="水印文本（仅嵌入时有效）：", bg="white").pack(anchor="w")
        self.text_wm = tk.Text(frm_wm, height=4)
        self.text_wm.pack(fill="both", expand=True)
        self.load_config()

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
        
        # 重置配置按钮
        frm_reset = tk.Frame(self, bg="white")
        frm_reset.pack(pady=5, fill="x", padx=20)
        tk.Button(frm_reset, text="重置配置", command=self.reset_config).pack(side="right")

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
        text = self.text_wm.get("1.0", "end").rstrip()
        self.save_config(text)
        return text
        
    def load_config(self):
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    self.entry_pwd.delete(0, tk.END)
                    self.entry_pwd.insert(0, config.get("pwd", ""))
                    self.text_wm.delete("1.0", tk.END)
                    self.text_wm.insert("1.0", config.get("last_wm_text", "Copyright@\nAuthor@"))
            else:
                self.text_wm.delete("1.0", tk.END)
                self.text_wm.insert("1.0", "Copyright@\nAuthor@")
        except Exception as e:
            print(f"加载配置失败: {e}")
            self.text_wm.delete("1.0", tk.END)
            self.text_wm.insert("1.0", "Copyright@\nAuthor@")
            
    def save_config(self, wm_text):
        try:
            config = {
                "pwd": self.entry_pwd.get().strip(),
                "last_wm_text": wm_text
            }
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            print(f"保存配置失败: {e}")
            
    def on_close(self):
        ## 窗口关闭事件处理
        try:
            if not self._saved_before_close:
                wm_text = self.text_wm.get("1.0", "end").rstrip()
                self.save_config(wm_text)
                self._saved_before_close = True
        except Exception as e:
            print(f"关闭时保存配置出错: {e}")
        self.destroy()
            
    def reset_config(self):
        try:
            if os.path.exists(self.config_path):
                os.remove(self.config_path)
                self.entry_pwd.delete(0, tk.END)
                self.text_wm.delete("1.0", tk.END)
                self.text_wm.insert("1.0", "Copyright@\nAuthor@")
                messagebox.showinfo("重置成功", "配置文件已删除，已恢复默认设置")
            else:
                messagebox.showinfo("提示", "配置文件不存在，无需重置")
        except Exception as e:
            messagebox.showerror("重置失败", f"重置配置时出错: {e}")

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

        messagebox.showinfo("嵌入成功", f"输出文件：\n{dst_img}\n\n【请完善保存以下内容！】\n水印长度：{wm_len} 尺寸：{width}x{height}")

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
    app = App()
    try:
        app.mainloop()
    except Exception as e:
        print(f"程序运行时出错: {e}")
    finally:
        try:
            # 在窗口关闭前保存配置
            if hasattr(app, '_saved_before_close') and not app._saved_before_close:
                wm_text = app.text_wm.get("1.0", "end").rstrip()
                app.save_config(wm_text)
                app._saved_before_close = True
        except Exception as e:
            print(f"保存配置时出错: {e}")
