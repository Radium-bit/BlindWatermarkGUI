## Copyright (c) 2025 Radium-bit
## SPDX-License-Identifier: Apache-2.0
## See LICENSE file for full terms

# 程序版本号
VERSION = r""
# 开发模型路径
DEV_MODEL_PATH = r""
# SitePackagePath
SITE_PACKAGE_PATH = r""

import os
import sys
from dotenv import load_dotenv
import tkinter as tk
from tkinter import messagebox, Checkbutton, BooleanVar, Frame
from tkinterdnd2 import DND_FILES, TkinterDnD
import tempfile
import re
import json
from pathlib import Path
from PIL import Image, ImageTk
import webbrowser

# 导入水印模块
from watermark.embed import WatermarkEmbedder
from watermark.extract import WatermarkExtractor


class App(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.version = " Unknown"
        # 设置模型路径环境变量（必须在导入qreader前执行）
        self.set_model_path()
        # 初始化QReader
        from debug.module_tracker import start_tracking ##Module import Tracker(Debug USE)
        ## [DEBUG] 用于测试记录QReader所使用的隐藏导入模块
        # start_tracking()
        from qreader import QReader
        self.qreader = QReader()
        self.processing_window = None
        self.processing_label = None
        self.processing_animation = None
        self.processing_active = False
        self.config_path = os.path.join(os.environ["USERPROFILE"], "radiumbit.blindwatermark.config.json")
        self._saved_before_close = False
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self._load_build_env()
        self.title(f"BlindWatermarkGUI v{self.version}")
        self.geometry("580x560")
        self.configure(bg="white")
        self.qr_window = None

        # 初始化水印处理器
        self.embedder = WatermarkEmbedder(self)
        self.extractor = WatermarkExtractor(self)

        self.mode = tk.StringVar(value="embed")

        # 模式选择
        frm_mode = tk.Frame(self, bg="white")
        frm_mode.pack(pady=5)
        tk.Label(frm_mode, text="选择模式：", bg="white").pack(side="left")
        tk.Radiobutton(frm_mode, text="嵌入水印", variable=self.mode, value="embed", bg="white").pack(side="left")
        tk.Radiobutton(frm_mode, text="提取水印", variable=self.mode, value="extract", bg="white").pack(side="left")

        # 输出格式选择
        self.output_format = tk.StringVar(value="PNG")
        format_frame = tk.Frame(self, bg="white")
        format_frame.pack(pady=5, fill="x", padx=20)
        tk.Label(format_frame, text="输出格式:", bg="white").pack(side="left")
        tk.Radiobutton(format_frame, text="PNG", variable=self.output_format, value="PNG", bg="white").pack(side="left", padx=5)
        tk.Radiobutton(format_frame, text="JPG", variable=self.output_format, value="JPG", bg="white").pack(side="left", padx=5)
        
        # 增强模式选项
        self.enhanced_mode = BooleanVar(value=False)
        self.enhanced_check = Checkbutton(
            format_frame, 
            text="增强抗干扰模式",
            variable=self.enhanced_mode,
            command=self.show_enhanced_warning
        )
        self.enhanced_check.pack(side='top', padx=5, pady=5)
        
        # 兼容性模式选项
        self.compatibility_mode = BooleanVar(value=False)
        self.compatibility_check = Checkbutton(
            format_frame, 
            text="启用兼容模式",
            variable=self.compatibility_mode,
            command=self.show_compatibility_info
        )
        self.compatibility_check.pack(side='top', padx=5, pady=5)
        
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
        lbl = tk.Label(self, text="请将图片拖入此区域", bg="#f0f0f0", fg="black", relief="ridge", borderwidth=2, height=10)
        lbl.pack(expand=True, fill="both", padx=20, pady=20)
        
        # 二维码水印设置
        self.qr_frame = tk.Frame(self)
        self.qr_frame.pack(pady=10)
        

        lbl.drop_target_register(DND_FILES)
        lbl.dnd_bind('<<Drop>>', self.on_drop)
        
        # 项目地址链接
        self.project_address_label = tk.Label(self, text="项目地址:", font=("Arial", 10))
        self.project_address_label.pack(pady=(10,0))
        self.project_link = tk.Label(self, text="Radium-bit/BlindWatermarkGUI", fg="blue", cursor="hand2", font=("Arial", 10))
        self.project_link.pack(pady=(0,5))
        self.project_link.bind("<Button-1>", lambda e: webbrowser.open("https://github.com/Radium-bit/BlindWatermarkGUI"))

    def _load_build_env(self):
        """加载BUILD.ENV配置文件并设置版本号"""
        try:
            # 获取基础路径
            if getattr(sys, 'frozen', False):
                # 打包环境
                base_path = sys._MEIPASS
                build_env_path = os.path.join(base_path, 'BUILD.ENV')
            else:
                # 开发环境
                build_env_path = 'BUILD.ENV'
            # 检查文件是否存在
            if os.path.exists(build_env_path):
                load_dotenv(build_env_path)
                version = os.getenv('VERSION')
                if version:
                    self.version = version
                else:
                    print("警告：BUILD.ENV中未找到VERSION配置")
                    self.version = " Unknown"  # 默认版本
            else:
                print(f"警告：未找到BUILD.ENV文件，路径：{build_env_path}")
                self.version = " Unknown"  # 默认版本
        except Exception as e:
            print(f"加载BUILD.ENV失败: {e}")
            self.version = " Unknown"  # 默认版本

    def on_drop(self, ev):
        for f in self.tk.splitlist(ev.data):
            f = f.strip('{}')
            try:
                if not os.path.exists(f):
                    messagebox.showerror("错误", "文件不存在或路径无效")
                    return
                    
                if self.mode.get() == "embed":
                    if self.compatibility_mode.get():
                        self.embedder.embed_watermark_v013(f)
                    else:
                        self.embedder.embed_watermark(f)
                else:
                    if self.compatibility_mode.get():
                        self.extractor.extract_watermark_v013(f)
                    else:
                        self.extractor.extract_watermark(f)
            except Exception as e:
                messagebox.showerror("错误", str(e))
                # 确保处理窗口关闭
                if hasattr(self, 'hide_processing_window'):
                    self.hide_processing_window()

    def get_pwd(self):
        pwd = self.entry_pwd.get().strip()
        return pwd if pwd else "1234"

    def get_wm_text(self):
        text = self.text_wm.get("1.0", "end").rstrip()
        self.save_config(text)
        return text
        
    def set_model_path(self):
        """设置qreader模型路径环境变量"""
        if getattr(sys, 'frozen', False):
            # 打包环境：使用临时解压目录
            base_path = sys._MEIPASS
            model_path = os.path.join(base_path, "qrdet", ".model", "qrdet-s.pt")
        else:
            # 开发环境：使用默认安装路径，从DOT.ENV加载
            if not os.path.exists('DEV.ENV'):
                print("警告：未找到DEV.ENV文件，请创建并配置路径")
            else:
                load_dotenv('DEV.ENV')
                envpath = os.getenv('SITE_PACKAGE_PATH')
                model_path = os.path.join(envpath, "qrdet", ".model", "qrdet-s.pt")
        
        # 关键！设置环境变量阻止下载
        os.environ["QRDET_MODEL_PATH"] = model_path
        print(f"模型路径已设置为: {model_path}")  # 调试用

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
        from debug.module_tracker import stop_and_save_tracking
        ## [DEBUG] 用于测试记录QReader所使用的隐藏导入模块
        # stop_and_save_tracking()
        if not self._saved_before_close:
            try:
                with open(self.config_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "pwd": self.entry_pwd.get().strip(),
                        "last_wm_text": self.text_wm.get("1.0", "end").rstrip()
                    }, f, ensure_ascii=False, indent=4)
                app._saved_before_close = True
            except Exception as e:
                print(f"保存配置时出错: {e}")
        self.destroy()
            
    def show_processing_window(self, message):
        """显示处理中窗口"""
        if not self.processing_window:
            self.processing_window = tk.Toplevel(self)
            self.processing_window.title("处理中")
            self.processing_window.geometry("300x100")
            self.processing_window.resizable(False, False)
            self.processing_window.protocol("WM_DELETE_WINDOW", lambda: None)  # 禁用关闭按钮
            
            self.processing_label = tk.Label(self.processing_window, text=message)
            self.processing_label.pack(pady=10)
            
            # 简单的动画效果
            self.processing_animation = tk.Label(self.processing_window, text="◐")
            self.processing_animation.pack()
            self.animate_processing()
            
        self.processing_active = True
    
    def hide_processing_window(self):
        """隐藏处理中窗口"""
        if self.processing_window:
            self.processing_active = False
            self.processing_window.destroy()
            self.processing_window = None
            
    def show_qr_code(self, qr_path, text=None, status=None, *images):
        ## 显示二维码窗口
        if self.qr_window:
            self.qr_window.destroy()
            
        self.qr_window = tk.Toplevel(self)
        self.qr_window.title("水印提取结果")
        self.qr_window.geometry("600x700")
        
        # 清理文件回调
        def cleanup_callback():
            return True

        # 添加窗口关闭事件处理
        def on_close():
            if qr_path and os.path.exists(qr_path):
                try:
                    os.remove(qr_path)
                    return cleanup_callback
                except:
                    return cleanup_callback
                    pass
            self.qr_window.destroy()
            
        self.qr_window.protocol("WM_DELETE_WINDOW", on_close)
        
        # 显示状态标题
        if status==True:
            tk.Label(self.qr_window, text="水印提取成功", font=("Arial", 12, "bold")).pack()
        else:
            tk.Label(self.qr_window, text="水印提取失败，请检查下方是否存在二维码图像", font=("Arial", 12, "bold")).pack()
        
        # 处理传入的图片
        if images:
            for img_data in images:
                if img_data is not None:
                    size, img_array = img_data
                    # 将numpy数组转换回PIL Image
                    img = Image.fromarray(img_array)
                    # 对小于256的图片进行放大
                    if max(img.size) < 256:
                        scale = 256 / max(img.size)
                        new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
                        img = img.resize(new_size, Image.Resampling.LANCZOS)
                    img_tk = ImageTk.PhotoImage(img)
                    label = tk.Label(self.qr_window, image=img_tk)
                    label.image = img_tk  # 保持引用
                    label.pack(pady=5)
                    # 显示尺寸标签
                    tk.Label(self.qr_window, text=f"尺寸: {size}").pack()
        elif qr_path:
            try:
                img = Image.open(qr_path)
                img_tk = ImageTk.PhotoImage(img)
                label = tk.Label(self.qr_window, image=img_tk)
                label.image = img_tk  # 保持引用
                label.pack(pady=5)
            except Exception as e:
                messagebox.showerror("错误", f"无法显示图片: {str(e)}")
        # 显示解码文本（如果有）
        if text:
            tk.Label(self.qr_window, text="解码文本:", font=("Arial", 10, "bold")).pack()
            text_label = tk.Label(self.qr_window, text=text, wraplength=380, justify="left")
            text_label.pack(pady=5)
        
        # 关闭按钮
        close_btn = tk.Button(self.qr_window, text="关闭", 
                            command=lambda: [os.unlink(qr_path) if qr_path and os.path.exists(qr_path) else None,
                            self.qr_window.destroy()])
        close_btn.pack(pady=10)
    
    def animate_processing(self):
        """处理动画效果"""
        if not self.processing_active:
            return
            
        frames = ["◐", "◓", "◑", "◒"]
        current_frame = self.processing_animation.cget("text")
        next_frame = frames[(frames.index(current_frame) + 1) % len(frames)]
        self.processing_animation.config(text=next_frame)
        self.after(200, self.animate_processing)
    
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

    def show_enhanced_warning(self):
        # print("enhanced_mode_status",self.enhanced_mode.get())
        if self.enhanced_mode.get():
            if self.compatibility_mode.get():
                messagebox.showwarning("注意", "v0.1.3兼容模式下\n无法增强处理")
                self.enhanced_mode.set(False)
            else:
                messagebox.showwarning("提示", "增强模式会轻微降低图像质量！\n但可提高抗干扰能力，\n请确保您的图片不会丢失重要信息。")

    def show_compatibility_info(self):
        if self.compatibility_mode.get():
            messagebox.showinfo("兼容模式", "已启用v0.1.3兼容模式\n仅嵌入文本水印（抗干扰差！）\n且不能解析新版水印\n仅作为兼容选项，不再推荐使用")
            if self.enhanced_mode.get():
                messagebox.showwarning("注意", "v0.1.3兼容模式下\n不可使用增强处理")
                self.enhanced_mode.set(False)
        else:
            pass


# 抑制QReader的特定编码解析警告
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='Double decoding failed')

if __name__ == "__main__":
    try:
        from hooks.torch_fixes import apply_torch_numpy_fix, disable_problematic_features
        disable_problematic_features()
        apply_torch_numpy_fix()
    except ImportError:
        print("Torch修复模块未找到，继续运行...")
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