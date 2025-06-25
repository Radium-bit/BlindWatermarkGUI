## Copyright (c) 2025 Radium-bit
## SPDX-License-Identifier: Apache-2.0
## See LICENSE file for full terms

# 程序版本号
VERSION = r""
# 开发模型路径
DEV_MODEL_PATH = r""
# SitePackagePath
SITE_PACKAGE_PATH = r""

# 全局变量用于启动画面
splash_window = None
splash_status_label = None
main_root = None  # 主根窗口

def create_main_root():
    """创建主根窗口（不可见）"""
    global main_root
    import tkinter as tk
    
    main_root = tk.Tk()
    main_root.withdraw()  # 立即隐藏
    return main_root

def create_splash_screen():
    """创建启动画面"""
    global splash_window, splash_status_label
    import tkinter as tk
    
    # 创建独立的启动窗口，不依赖 main_root
    splash_window = tk.Tk()
    splash_window.title("正在启动 BlindWatermarkGUI")
    
    # 计算居中位置
    window_width = 400
    window_height = 250
    screen_width = splash_window.winfo_screenwidth()
    screen_height = splash_window.winfo_screenheight()
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    
    splash_window.geometry(f"{window_width}x{window_height}+{x}+{y}")
    splash_window.resizable(False, False)
    splash_window.configure(bg="#2c3e50")
    splash_window.overrideredirect(True)  # 无边框窗口
    
    # 创建界面元素
    tk.Label(splash_window, text="BlindWatermarkGUI", 
            font=("Arial", 16, "bold"), fg="white", bg="#2c3e50").pack(pady=(30, 10))
    
    tk.Label(splash_window, text="盲水印处理工具", 
            font=("Arial", 10), fg="#bdc3c7", bg="#2c3e50").pack(pady=(0, 20))
    
    splash_status_label = tk.Label(splash_window, text="正在初始化...", 
                                font=("Arial", 10), fg="#3498db", bg="#2c3e50")
    splash_status_label.pack(pady=10)
    
    # 进度动画
    progress_label = tk.Label(splash_window, text="◐", 
                            font=("Arial", 14), fg="#ffffff", bg="#2c3e50")
    progress_label.pack(pady=5)
    
    tk.Label(splash_window, text="请勿重复点击程序", 
            font=("Arial", 9), fg="#e67e22", bg="#2c3e50").pack(pady=(1, 5))
    
    tk.Label(splash_window, text="2025 © Radium-bit", 
            font=("Arial", 8), fg="#ffffff", bg="#2c3e50").pack(side="bottom", pady=5)
    
    # 动画控制变量
    animation_running = [True]
    animation_job = [None]  # 存储 after 任务 ID
    
    # 启动进度动画
    def animate_progress():
        if splash_window and animation_running[0]:
            try:
                if splash_window.winfo_exists():
                    frames = ["◐", "◓", "◑", "◒"]
                    current = progress_label.cget("text")
                    next_frame = frames[(frames.index(current) + 1) % len(frames)]
                    progress_label.config(text=next_frame)
                    splash_window.after(200, animate_progress)
                else:
                    animation_running[0] = False
            except (tk.TclError, ValueError):
                animation_running[0] = False

    # 改进的停止动画函数
    def stop_animation():
        animation_running[0] = False
        if animation_job[0] is not None:
            try:
                splash_window.after_cancel(animation_job[0])
            except:
                pass
            animation_job[0] = None
    
    # 存储动画停止函数到窗口属性
    # splash_window.stop_animation = lambda: animation_running.__setitem__(0, False)
    splash_window.stop_animation = stop_animation
    
    animate_progress()
    splash_window.update()
    return splash_window

def update_splash_status(text):
    """更新启动画面状态"""
    global splash_window, splash_status_label
    if splash_window and splash_status_label:
        try:
            if splash_window.winfo_exists():
                splash_status_label.config(text=text)
                splash_window.update()
        except tk.TclError:
            pass

def close_splash():
    """关闭启动画面"""
    global splash_window
    if splash_window:
        try:
            print("正在关闭启动画面...")
            # 停止动画
            if hasattr(splash_window, 'stop_animation'):
                splash_window.stop_animation()
                print("动画已停止")
            # 销毁启动窗口
            splash_window.destroy()
            splash_window = None
            print("启动画面已完全关闭")
        except Exception as e:
            print(f"关闭启动画面时出错: {e}")
            splash_window = None

## 启动前检查VC Redist
def check_vc_redist():
    from tkinter import messagebox
    """检查Visual C++ Redistributable x64是否已安装"""
    try:
        # 检查注册表中是否存在VC++ Redist
        import winreg
        
        # 常见的VC++ Redist注册表路径
        redist_paths = [
            r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
            r"SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
            r"SOFTWARE\Classes\Installer\Dependencies\Microsoft.VS.VC_RuntimeMinimumVSU_amd64,v14",
            r"SOFTWARE\Classes\Installer\Dependencies\{e2803110-78b3-4664-a479-3611a381656a}",
        ]
        
        for path in redist_paths:
            try:
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path):
                    print("Found VC Redist.")
                    return True
            except FileNotFoundError:
                continue
                
        return False
    except ImportError:
        print("ImportError")
        # 如果不是Windows系统，跳过检查
        return True
    except Exception as e:
        messagebox.showerror("找不到VC++ Redist，请安装后重新打开本程序")
        print(f"检查VC++ Redist时出错: {e}")
        return False

def install_vc_redist():
    from tkinter import messagebox
    import subprocess
    """
    安装Visual C++ Redistributable。
    该方法会打开安装包，然后程序会退出，由用户完成安装。
    """
    redist_path = ""
    if getattr(sys, 'frozen', False):
        # 打包环境：使用临时解压目录
        base_path = sys._MEIPASS
        redist_path = os.path.join(base_path, "VC_redist.x64.exe")

    if not os.path.exists(redist_path):
        error_message = f"未找到VC++ Redist安装包: {redist_path}\n请确保 'VC_redist.x64.exe' 文件与本程序在同一目录下。"
        messagebox.showerror("安装错误", error_message)
        raise FileNotFoundError(error_message)

    try:
        # 以交互式方式运行安装程序
        # 注意：这里我们不再使用 /quiet 参数，并移除了 check=True
        # 因为我们希望程序打开安装包后就退出，不等待安装结果
        subprocess.Popen([redist_path])
        
        # 提示用户手动完成安装
        messagebox.showinfo("安装提示", "VC++ Redist 安装程序已启动，请按照屏幕上的指示完成安装后再启动本程序。\n本程序即将退出。")
        
        # 程序退出
        sys.exit() 

    except Exception as e:
        error_message = f"启动VC++ Redist安装程序时发生错误: {e}"
        messagebox.showerror("启动错误：", error_message)
        raise

def import_modules_progressively():
    """逐步导入模块"""
    global imported_modules, main_root
    #开始加载模块
    try:
        # 第一阶段：基础模块
        update_splash_status("加载基础模块...")
        from dotenv import load_dotenv
        imported_modules['load_dotenv'] = load_dotenv
        update_splash_status("加载UI框架...")
        from tkinterdnd2 import DND_FILES, TkinterDnD
        imported_modules['DND_FILES'] = DND_FILES
        imported_modules['TkinterDnD'] = TkinterDnD
        
        # 安全地重新创建支持拖拽的根窗口
        update_splash_status("重新初始化主窗口...")
        if main_root:
            try:
                main_root.destroy()
            except:
                pass  # 忽略销毁错误
        
        main_root = TkinterDnD.Tk()
        main_root.withdraw()
        
        # 第二阶段：常用库
        update_splash_status("加载图像处理模块...")
        import tempfile
        import re
        import json
        from pathlib import Path
        from PIL import Image, ImageTk
        import webbrowser
        imported_modules.update({
            'tempfile': tempfile,
            're': re,
            'json': json,
            'Path': Path,
            'Image': Image,
            'ImageTk': ImageTk,
            'webbrowser': webbrowser
        })
        
        # 第三阶段：水印模块
        update_splash_status("加载水印处理模块...")
        from watermark.embed import WatermarkEmbedder
        from watermark.extract import WatermarkExtractor
        imported_modules['WatermarkEmbedder'] = WatermarkEmbedder
        imported_modules['WatermarkExtractor'] = WatermarkExtractor
        
        return True
        
    except Exception as e:
        print(f"模块导入失败: {e}")
        return False
    
    # 启动进度动画
    def animate_progress():
        if (splash_window and splash_window.winfo_exists() and 
            animation_running[0] and main_root and main_root.winfo_exists()):
            try:
                frames = ["◐", "◓", "◑", "◒"]
                current = progress_label.cget("text")
                next_frame = frames[(frames.index(current) + 1) % len(frames)]
                progress_label.config(text=next_frame)
                main_root.after(200, animate_progress)
            except (tk.TclError, ValueError):
                # 如果窗口已被销毁或其他错误，停止动画
                animation_running[0] = False
    
    # 存储动画停止函数到窗口属性
    splash_window.stop_animation = lambda: animation_running.__setitem__(0, False)
    
    animate_progress()
    splash_window.update()
    return splash_window

def update_splash_status(text):
    """更新启动画面状态"""
    global splash_window, splash_status_label, main_root
    if (splash_window and splash_status_label and 
        splash_window.winfo_exists() and main_root and main_root.winfo_exists()):
        try:
            splash_status_label.config(text=text)
            splash_window.update()
        except tk.TclError:
            pass

def close_splash():
    """关闭启动画面"""
    global splash_window
    if splash_window:
        try:
            print("正在关闭启动画面...")
            # 停止动画
            if hasattr(splash_window, 'stop_animation'):
                splash_window.stop_animation()
                print("动画已停止")
            # 销毁启动窗口
            splash_window.destroy()
            splash_window = None
            print("启动画面已完全关闭")
        except Exception as e:
            print(f"关闭启动画面时出错: {e}")
            splash_window = None

# 延迟导入的模块将在这里存储
imported_modules = {}

def import_modules_progressively():
    """逐步导入模块"""
    global imported_modules, main_root
    
    try:
        # P1：基础模块
        update_splash_status("加载基础模块...")
        from dotenv import load_dotenv
        imported_modules['load_dotenv'] = load_dotenv
        
        update_splash_status("加载UI框架...")
        from tkinterdnd2 import DND_FILES, TkinterDnD
        imported_modules['DND_FILES'] = DND_FILES
        imported_modules['TkinterDnD'] = TkinterDnD

        if main_root:
            main_root.destroy()
        main_root = TkinterDnD.Tk()
        main_root.withdraw()
        
        # P2：常用库
        update_splash_status("加载图像处理模块...")
        import tempfile
        import re
        import json
        from pathlib import Path
        from PIL import Image, ImageTk
        import webbrowser
        imported_modules.update({
            'tempfile': tempfile,
            're': re,
            'json': json,
            'Path': Path,
            'Image': Image,
            'ImageTk': ImageTk,
            'webbrowser': webbrowser
        })
        
        # P3：水印模块
        update_splash_status("加载水印处理模块...")
        from watermark.embed import WatermarkEmbedder
        from watermark.extract import WatermarkExtractor
        imported_modules['WatermarkEmbedder'] = WatermarkEmbedder
        imported_modules['WatermarkExtractor'] = WatermarkExtractor
        
        return True
        
    except Exception as e:
        print(f"模块导入失败: {e}")
        return False

def create_app_class():
    """创建App类，在导入完成后调用"""
    
    class App:
        def __init__(self, root):
            global main_root
            self.root = root
            self.root.title("正在初始化...")
            
            self.version = " Unknown"
            
            # P4：设定识别模型
            update_splash_status("设定识别模型...")
            self.set_model_path()
            
            update_splash_status("加载QReader（这可能需要一些时间）...")
            from debug.module_tracker import start_tracking ##Module import Tracker(Debug USE)
            ## [DEBUG] 用于测试记录QReader所使用的隐藏导入模块
            # start_tracking()
            from qreader import QReader
            self.qreader = QReader()
            
            # P5：初始化界面
            update_splash_status("构建用户界面...")
            
            self.processing_window = None
            self.processing_label = None
            self.processing_animation = None
            self.processing_active = False
            self.config_path = os.path.join(os.environ["USERPROFILE"], "radiumbit.blindwatermark.config.json")
            self._saved_before_close = False
            self.root.protocol("WM_DELETE_WINDOW", self.on_close)
            self._load_build_env()
            self.root.title(f"BlindWatermarkGUI v{self.version}")
            self.root.geometry("580x760")
            self.root.configure(bg="white")
            # self.qr_window = None

            # 初始化水印处理器
            self.embedder = imported_modules['WatermarkEmbedder'](self)
            self.extractor = imported_modules['WatermarkExtractor'](self)

            # 创建界面
            self.create_ui()
            
            # 完成初始化
            update_splash_status("启动完成！")
            
            # 延迟关闭启动画面并显示主窗口
            self.root.after(500, self.finish_startup)

        def create_ui(self):
            """创建用户界面"""
            import tkinter as tk
            
            self.mode = tk.StringVar(value="embed")

            # 模式选择
            horizontal_frame = tk.Frame(self.root, bg="white")
            horizontal_frame.pack(fill="x", padx=20)
            # 左侧容器：包含标题和模式选择
            left_container = tk.Frame(horizontal_frame, bg="white")
            left_container.pack(side="left", anchor="sw")

            # 标题
            title_frame = tk.Frame(left_container, bg="white")
            title_frame.pack(anchor="w")
            tk.Label(title_frame, text="BlindWatermarkGUI", bg="white", font=("Comic Sans MS", 20, "italic")).pack(side="left")
            frm_mode = tk.Frame(left_container, bg="white")
            frm_mode.pack(side="left", anchor="sw", pady=0)
            tk.Label(frm_mode, text="选择模式：", bg="white").pack(side="left")
            tk.Radiobutton(frm_mode, text="嵌入水印", variable=self.mode, value="embed", bg="white").pack(side="left")
            tk.Radiobutton(frm_mode, text="提取水印", variable=self.mode, value="extract", bg="white").pack(side="left",padx=5)
            
            placeholder_frame = tk.Frame(horizontal_frame, bg="white")  # 透明占位
            placeholder_frame.pack(side="right", anchor="ne", pady=30)
            format_frame = tk.Frame(self.root, bg="white")
            # 增强模式选项
            from tkinter import BooleanVar, Checkbutton
            options_frame_up = tk.Frame(horizontal_frame, bg="white")
            options_frame_up.pack(side="right", anchor="se", pady=(20,0))
            options_frame_down = tk.Frame(format_frame, bg="white")
            options_frame_down.pack(side="right", anchor="ne", pady=(0,2))
            self.enhanced_mode = BooleanVar(value=False)
            self.enhanced_check = Checkbutton(
                options_frame_up, 
                text="增强水印模式",
                variable=self.enhanced_mode,
                command=self.show_enhanced_warning
            )
            self.enhanced_check.pack(anchor='e',pady=(0,2))

            # 兼容性模式选项
            self.compatibility_mode = BooleanVar(value=False)
            self.compatibility_check = Checkbutton(
                options_frame_up, 
                text="启用兼容模式",
                variable=self.compatibility_mode,
                command=self.toggle_compatibility_mode
            )
            self.compatibility_check.pack(anchor='e')

            # 原图显示选项
            self.show_orignal_extract_picture = BooleanVar(value=False)
            self.show_orignal_extract_picture_check = Checkbutton(
                options_frame_down, 
                text="提取显示原图",
                variable=self.show_orignal_extract_picture,
            )
            self.show_orignal_extract_picture_check.pack(anchor='e')

            # 输出格式选择（单独一行）
            self.output_format = tk.StringVar(value="PNG")
            format_frame.pack(pady=2, fill="x", anchor="nw", padx=20)
            tk.Label(format_frame, text="输出格式:", bg="white").pack(side="left")
            tk.Radiobutton(format_frame, text="PNG", variable=self.output_format, value="PNG", bg="white").pack(side="left", padx=5)
            tk.Radiobutton(format_frame, text="JPG", variable=self.output_format, value="JPG", bg="white").pack(side="left", padx=5)
            
            # 密码输入
            frm_pwd = tk.Frame(self.root, bg="white")
            frm_pwd.pack(pady=5, fill="x", padx=20)
            tk.Label(frm_pwd, text="密码（可空，默认为1234）：", bg="white").pack(side="left")
            self.entry_pwd = tk.Entry(frm_pwd, show="*")
            self.entry_pwd.insert(0, "")
            self.entry_pwd.pack(side="left", fill="x", expand=True)

            # 水印文本输入
            frm_wm = tk.Frame(self.root, bg="white")
            frm_wm.pack(pady=5, fill="both", padx=20)
            tk.Label(frm_wm, text="水印文本（仅嵌入时有效）：", bg="white").pack(anchor="w")
            self.text_wm = tk.Text(frm_wm, height=4)
            self.text_wm.pack(fill="both", expand=True)
            self.load_config()

            # 水印长度输入 - 兼容模式专用，默认隐藏
            self.frm_len = tk.Frame(self.root, bg="white")
            tk.Label(self.frm_len, text="提取水印长度（可空）：", bg="white").pack(side="left")
            self.entry_ws = tk.Entry(self.frm_len)
            self.entry_ws.insert(0, "")
            self.entry_ws.pack(side="left", fill="x", expand=True)

            # 原图尺寸输入
            self.frm_size = tk.Frame(self.root, bg="white")
            self.frm_size.pack(pady=5, fill="x", padx=20)
            tk.Label(self.frm_size, text="原图尺寸（如1920x1080，可空）：", bg="white").pack(side="left")
            self.entry_size = tk.Entry(self.frm_size)
            self.entry_size.insert(0, "")
            self.entry_size.pack(side="left", fill="x", expand=True)

            # 输出目录
            frm_out = tk.Frame(self.root, bg="white")
            frm_out.pack(pady=5, fill="x", padx=20)
            tk.Label(frm_out, text="临时目录（默认系统目录）：", bg="white").pack(side="left")
            self.entry_out = tk.Entry(frm_out)
            self.entry_out.insert(0, imported_modules['tempfile'].gettempdir())
            self.entry_out.pack(side="left", fill="x", expand=True)
            
            # 重置配置按钮
            frm_reset = tk.Frame(self.root, bg="white")
            frm_reset.pack(pady=5, fill="x", padx=20)
            tk.Button(frm_reset, text="重置配置", command=self.reset_config).pack(side="right")

            # 拖拽区域
            from tkinterdnd2 import DND_FILES
            lbl = tk.Label(self.root, text="请将图片拖入此区域", bg="#f0f0f0", fg="black", relief="ridge", borderwidth=2, height=10)
            lbl.pack(expand=True, fill="both", padx=20, pady=20)
            
            # 二维码水印设置
            self.qr_frame = tk.Frame(self.root)
            self.qr_frame.pack(pady=10)

            lbl.drop_target_register(imported_modules['DND_FILES'])
            lbl.dnd_bind('<<Drop>>', self.on_drop)
            
            # 项目地址链接
            self.project_address_label = tk.Label(self.root, text="项目地址:", font=("Arial", 10))
            self.project_address_label.pack(pady=(1,0))
            self.project_link = tk.Label(self.root, text="Radium-bit/BlindWatermarkGUI", fg="blue", cursor="hand2", font=("Arial", 10))
            self.project_link.pack(pady=(0,5))
            self.project_link.bind("<Button-1>", lambda e: imported_modules['webbrowser'].open("https://github.com/Radium-bit/BlindWatermarkGUI"))

        def finish_startup(self):
            """完成启动过程"""
            try:
                print("开始完成启动过程...")
                # 关闭启动画面
                close_splash()
                print("启动画面已关闭")
                
                # 显示主窗口
                self.root.deiconify()
                print("主窗口已显示")
                self.root.lift()
                self.root.focus_force()
                
                # 确保窗口获得焦点并保持活跃
                self.root.attributes('-topmost', True)
                self.root.after(100, lambda: self.root.attributes('-topmost', False))
                
                print("启动过程完成")
            except Exception as e:
                print(f"完成启动时出错: {e}")
                import traceback
                traceback.print_exc()

        def toggle_compatibility_mode(self):
            """切换兼容模式时显示/隐藏相关输入框"""
            if self.compatibility_mode.get():
                # 启用兼容模式 - 显示输入框
                self.frm_len.pack(pady=5, fill="x", padx=20, before=self.entry_out.master)
                # 显示兼容模式信息
                self.show_compatibility_info()
            else:
                # 禁用兼容模式 - 隐藏输入框
                self.frm_len.pack_forget()

        def _load_build_env(self):
            """加载APP.ENV配置文件并设置版本号"""
            try:
                # 获取基础路径
                if getattr(sys, 'frozen', False):
                    # 打包环境
                    base_path = sys._MEIPASS
                    build_env_path = os.path.join(base_path, 'APP.ENV')
                else:
                    # 开发环境
                    build_env_path = 'APP.ENV'
                # 检查文件是否存在
                if os.path.exists(build_env_path):
                    imported_modules['load_dotenv'](build_env_path)
                    version = os.getenv('VERSION')
                    if version:
                        self.version = version
                    else:
                        print("警告：APP.ENV中未找到VERSION配置")
                        self.version = " Unknown"  # 默认版本
                else:
                    print(f"警告：未找到APP.ENV文件，路径：{build_env_path}")
                    self.version = " Unknown"  # 默认版本
            except Exception as e:
                print(f"加载APP.ENV失败: {e}")
                self.version = " Unknown"  # 默认版本

        def on_drop(self, ev):
            from tkinter import messagebox
            for f in self.root.tk.splitlist(ev.data):
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
                    imported_modules['load_dotenv']('DEV.ENV')
                    envpath = os.getenv('SITE_PACKAGE_PATH')
                    model_path = os.path.join(envpath, "qrdet", ".model", "qrdet-s.pt")
            
            # 关键！设置环境变量阻止下载
            os.environ["QRDET_MODEL_PATH"] = model_path
            print(f"模型路径已设置为: {model_path}")  # 调试用

        def load_config(self):
            import tkinter as tk
            try:
                if os.path.exists(self.config_path):
                    with open(self.config_path, "r", encoding="utf-8") as f:
                        config = imported_modules['json'].load(f)
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
                    imported_modules['json'].dump(config, f, indent=4)
            except Exception as e:
                print(f"保存配置失败: {e}")
                
        def on_close(self):
            print("程序关闭事件触发")
            try:
                from debug.module_tracker import stop_and_save_tracking
                ## [DEBUG] 用于测试记录QReader所使用的隐藏导入模块
                # stop_and_save_tracking()
                if not self._saved_before_close:
                    print("保存配置中...")
                    try:
                        with open(self.config_path, "w", encoding="utf-8") as f:
                            imported_modules['json'].dump({
                                "pwd": self.entry_pwd.get().strip(),
                                "last_wm_text": self.text_wm.get("1.0", "end").rstrip()
                            }, f, ensure_ascii=False, indent=4)
                        self._saved_before_close = True
                        print("配置保存成功")
                    except Exception as e:
                        print(f"保存配置时出错: {e}")
                print("销毁窗口...")
                self.root.destroy()
                print("窗口已销毁")
            except Exception as e:
                print(f"关闭程序时出错: {e}")
                import traceback
                traceback.print_exc()
                self.root.destroy()
                
        def show_processing_window(self, message):
            """显示处理中窗口"""
            import tkinter as tk
            if not self.processing_window:
                self.processing_window = tk.Toplevel(self.root)
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
            import tkinter as tk
            from tkinter import messagebox
            # if self.qr_window:
            #     self.qr_window.destroy()
            # 每次调用都创建一个新的 Toplevel 窗口实例
            
            new_qr_window = tk.Toplevel(self.root)
            new_qr_window.title("水印提取结果")
            new_qr_window.geometry("600x700")
            
            # 清理文件回调
            def cleanup_callback():
                return True

            # 添加窗口关闭事件处理
            def on_close_show_qr_window():
                if qr_path and os.path.exists(qr_path):
                    try:
                        os.remove(qr_path)
                        return cleanup_callback
                    except:
                        return cleanup_callback
                        pass
                new_qr_window.destroy()
                
            new_qr_window.protocol("WM_DELETE_WINDOW", on_close_show_qr_window)
            
            # 显示状态标题
            if status==True:
                tk.Label(new_qr_window, text="水印提取成功", font=("Arial", 12, "bold")).pack()
            else:
                tk.Label(new_qr_window, text="水印提取失败，请检查下方是否存在二维码图像", font=("Arial", 12, "bold")).pack()
            
            # 处理传入的图片
            if images:
                for img_data in images:
                    if img_data is not None:
                        size, img_array = img_data
                        # 将numpy数组转换回PIL Image
                        img = imported_modules['Image'].fromarray(img_array)
                        # 对小于256的图片进行放大
                        if max(img.size) < 256:
                            scale = 256 / max(img.size)
                            new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
                            img = img.resize(new_size, imported_modules['Image'].Resampling.LANCZOS)
                        img_tk = imported_modules['ImageTk'].PhotoImage(img)
                        label = tk.Label(new_qr_window, image=img_tk)
                        label.image = img_tk  # 保持引用
                        label.pack(pady=5)
                        # 显示尺寸标签
                        tk.Label(new_qr_window, text=f"尺寸: {size}").pack()
            elif qr_path:
                try:
                    img = imported_modules['Image'].open(qr_path)
                    img_tk = imported_modules['ImageTk'].PhotoImage(img)
                    label = tk.Label(new_qr_window, image=img_tk)
                    label.image = img_tk  # 保持引用
                    label.pack(pady=5)
                except Exception as e:
                    messagebox.showerror("错误", f"无法显示图片: {str(e)}")
            # 显示解码文本（如果有）
            if text:
                tk.Label(new_qr_window, text="解码文本:", font=("Arial", 10, "bold")).pack()
                text_label = tk.Label(new_qr_window, text=text, wraplength=380, justify="left")
                text_label.pack(pady=5)
            
            # 关闭按钮
            close_btn = tk.Button(new_qr_window, text="关闭", 
                                command=lambda: [os.unlink(qr_path) if qr_path and os.path.exists(qr_path) else None,
                                new_qr_window.destroy()])
            close_btn.pack(pady=10)
        
        def animate_processing(self):
            """处理动画效果"""
            if not self.processing_active:
                return
                
            frames = ["◐", "◓", "◑", "◒"]
            current_frame = self.processing_animation.cget("text")
            next_frame = frames[(frames.index(current_frame) + 1) % len(frames)]
            self.processing_animation.config(text=next_frame)
            self.root.after(200, self.animate_processing)
        
        def reset_config(self):
            import tkinter as tk
            from tkinter import messagebox
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
                path = imported_modules['tempfile'].gettempdir()
            if not os.path.isdir(path):
                raise ValueError(f"输出目录不存在：{path}")
            return path

        def get_ws(self):
            val = self.entry_ws.get().strip()
            return int(val) if val.isdigit() else None

        def get_target_size(self):
            val = self.entry_size.get().strip()
            if imported_modules['re'].match(r'^\d+x\d+$', val):
                w, h = map(int, val.lower().split('x'))
                return w, h
            return None

        def show_enhanced_warning(self):
            from tkinter import messagebox
            # print("enhanced_mode_status",self.enhanced_mode.get())
            if self.enhanced_mode.get():
                if self.compatibility_mode.get():
                    messagebox.showwarning("注意", "v0.1.3兼容模式下\n无法增强处理")
                    self.enhanced_mode.set(False)
                else:
                    messagebox.showwarning("提示", "增强模式会轻微降低图像质量！\n但可提高抗干扰能力，\n请确保您的图片不会丢失重要信息。")

        def show_compatibility_info(self):
            """显示兼容模式信息（不触发输入框显示/隐藏）"""
            from tkinter import messagebox
            if self.compatibility_mode.get():
                messagebox.showinfo("兼容模式", "已启用v0.1.3兼容模式\n仅嵌入文本水印（抗干扰差！）\n且不能解析新版水印\n仅作为兼容选项，不再推荐使用")
                if self.enhanced_mode.get():
                    messagebox.showwarning("注意", "v0.1.3兼容模式下\n不可使用增强处理")
                    self.enhanced_mode.set(False)
    
    return App

# 抑制QReader的特定编码解析警告
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='Double decoding failed')

if __name__ == "__main__":
    # 第一步：只导入最基本的模块
    import os
    import sys
    import tkinter as tk
    
    # 第二步：创建主根窗口
    try:
        main_root = create_main_root()
        print("主根窗口创建成功")
    except Exception as e:
        print(f"无法创建主根窗口: {e}")
        sys.exit(1)
    
    # 第三步：显示启动画面
    try:
        splash = create_splash_screen()
        print("启动画面创建成功")
    except Exception as e:
        print(f"无法创建启动画面: {e}")
        # 如果启动画面创建失败，继续运行但不显示启动画面
        pass
    
    try:
        ## 在加载UI这些模块之前，先检查是否存在Visual C++ Redist x64 
        ## 不存在的话帮用户打开进行安装，安装包在打包环境的根目录中，叫VC_redist.x64.exe
        ## 如果无法打开这个应用程序，也检测不到vc64，立即终止抛出异常
        print("Checking VC++ Redist...")
        if not check_vc_redist():
            print("Installing VC++ Redist...")
            install_vc_redist()

        # 第四步：逐步导入其他模块
        update_splash_status("准备加载程序模块...")
        
        if not import_modules_progressively():
            close_splash()
            print("模块导入失败，程序退出")
            sys.exit(1)
        
        # 第五步：应用torch修复
        update_splash_status("应用性能优化...")
        try:
            from hooks.torch_fixes import apply_torch_numpy_fix, disable_problematic_features
            disable_problematic_features()
            apply_torch_numpy_fix()
        except ImportError:
            print("Torch修复模块未找到，继续运行...")
        
        # 第六步：创建主应用
        update_splash_status("创建主应用...")
        
        # 动态创建App类
        App = create_app_class()
        print("App类创建成功")
        
        app = App(main_root)
        print("App实例创建成功")
        
        # 确保应用完全初始化
        main_root.update()
        print("App初始化完成")
        
        # 第七步：运行主循环
        print("准备启动主循环...")
        try:
            print("主循环开始运行")
            main_root.mainloop()
            print("主循环正常结束")
        except Exception as e:
            print(f"程序运行时出错: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("进入程序清理阶段...")
            try:
                # 在窗口关闭前保存配置
                if hasattr(app, '_saved_before_close') and not app._saved_before_close:
                    wm_text = app.text_wm.get("1.0", "end").rstrip()
                    app.save_config(wm_text)
                    app._saved_before_close = True
                print("配置保存完成")
            except Exception as e:
                print(f"保存配置时出错: {e}")
                import traceback
                traceback.print_exc()
                
    except Exception as e:
        close_splash()
        print(f"程序启动失败: {e}")
        import traceback
        traceback.print_exc()
        try:
            tk.messagebox.showerror("启动失败", f"程序启动时出错:\n{str(e)}")
        except:
            pass
        sys.exit(1)