## Copyright (c) 2025 Radium-bit
## SPDX-License-Identifier: Apache-2.0
## See LICENSE file for full terms

"""
原生启动画面（纯 ctypes + user32/gdi32）。

为什么不用 tkinter/Qt 画启动画面：
打包产物里，启动画面必须在"任何三方库导入之前"出现，否则用户会经历一段
双击后毫无反应的空白期。本模块只依赖标准库 ctypes，导入耗时以毫秒计，
因此可以在最早时机创建窗口，并在后续长耗时导入之间用 pump() 保持响应。

用法：
    from app.ui_native import splash
    splash.create("0.2.11")          # 尽可能早
    splash.set_status("加载核心计算模块...")
    splash.pump()                    # 在长耗时导入之间调用
    splash.close()                   # 主窗口即将显示前

非 Windows 平台或创建失败时，所有函数都降级为 no-op，调用方不必判空。
"""

from __future__ import annotations

import ctypes
import math
import os
import sys
import threading
import time
from ctypes import wintypes

_ENABLED = sys.platform.startswith("win")

# ── 外观参数 ─────────────────────────────────────────────────────────
_CLR_BG = (0x2C, 0x3E, 0x50)      # 深蓝灰，与经典 Tk 启动画面一致
_CLR_TITLE = (0xEC, 0xF0, 0xF1)
_CLR_SUBTITLE = (0xB4, 0xC2, 0xCF)
_CLR_STATUS = (0x5D, 0xAD, 0xE2)
_CLR_HINT = (0xE6, 0x7E, 0x22)
_CLR_VERSION = (0x8F, 0xA6, 0xB8)

_FACE = "Microsoft YaHei UI"
_WINDOW_W, _WINDOW_H = 460, 250
_TIMER_MS = 120
_SPINNER_STEP_DEG = 30

# ── Win32 常量 ───────────────────────────────────────────────────────
_WS_POPUP = 0x80000000
_WS_EX_TOPMOST = 0x00000008
_WS_EX_TOOLWINDOW = 0x00000080
_SW_SHOWNOACTIVATE = 4
_WM_DESTROY = 0x0002
_WM_PAINT = 0x000F
_WM_TIMER = 0x0113
_WM_ERASEBKGND = 0x0014
_WM_PRINTCLIENT = 0x0318
_PM_REMOVE = 0x0001
_DT_CENTER = 0x0001
_DT_VCENTER = 0x0004
_DT_SINGLELINE = 0x0020
_DT_NOPREFIX = 0x0800
_TRANSPARENT = 1
_NULL_BRUSH = 5
_ERROR_CLASS_ALREADY_EXISTS = 1410


class _State:
    hwnd = None
    enabled = False
    status_text = "正在初始化..."
    version_text = ""
    title = "BlindWatermarkGUI"
    subtitle = "盲水印处理工具"
    spinner_angle = 0
    dpi_scale = 1.0
    opened_at = None


_state = _State()

if _ENABLED:
    _user32 = ctypes.WinDLL("user32", use_last_error=True)
    _gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    _LRESULT = ctypes.c_ssize_t
    _WNDPROC = ctypes.WINFUNCTYPE(
        _LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
    )

    # 注意：ctypes.wintypes 在 64 位下把 HRGN / HBRUSH / HFONT / HPEN / HGDIOBJ
    # 都定义成 32 位 c_int，而 GDI 返回的是完整 64 位句柄（形如 0xFFFFFFFFxxxxxxxx）。
    # 用它们当 restype/argtypes 会让句柄被截断，传参时抛
    # "OverflowError: int too long to convert"（这正是启动画面最初静默失败的根因）。
    # 因此凡涉及 GDI 句柄的签名一律使用指针宽度的 HANDLE。
    HANDLE = ctypes.c_void_p

    class PAINTSTRUCT(ctypes.Structure):
        """ctypes.wintypes 没有提供 PAINTSTRUCT，这里按 windef.h 补齐。"""

        _fields_ = [
            ("hdc", wintypes.HDC),
            ("fErase", wintypes.BOOL),
            ("rcPaint", wintypes.RECT),
            ("fRestore", wintypes.BOOL),
            ("fIncUpdate", wintypes.BOOL),
            ("rgbReserved", ctypes.c_byte * 32),
        ]

    class WNDCLASSW(ctypes.Structure):
        _fields_ = [
            ("style", wintypes.UINT),
            ("lpfnWndProc", _WNDPROC),
            ("cbClsExtra", ctypes.c_int),
            ("cbWndExtra", ctypes.c_int),
            ("hInstance", wintypes.HINSTANCE),
            ("hIcon", HANDLE),
            ("hCursor", HANDLE),
            ("hbrBackground", HANDLE),
            ("lpszMenuName", wintypes.LPCWSTR),
            ("lpszClassName", wintypes.LPCWSTR),
        ]

    _user32.CreateWindowExW.restype = wintypes.HWND
    _user32.CreateWindowExW.argtypes = [
        wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID,
    ]
    _user32.DefWindowProcW.restype = _LRESULT
    _user32.DefWindowProcW.argtypes = [
        wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
    ]
    _user32.RegisterClassW.restype = wintypes.ATOM
    _user32.RegisterClassW.argtypes = [ctypes.POINTER(WNDCLASSW)]
    _user32.GetDC.restype = wintypes.HDC
    _user32.GetDC.argtypes = [wintypes.HWND]
    _user32.ReleaseDC.restype = ctypes.c_int
    _user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
    _user32.BeginPaint.restype = wintypes.HDC
    _user32.BeginPaint.argtypes = [wintypes.HWND, ctypes.POINTER(PAINTSTRUCT)]
    _user32.EndPaint.restype = wintypes.BOOL
    _user32.EndPaint.argtypes = [wintypes.HWND, ctypes.POINTER(PAINTSTRUCT)]
    _user32.FillRect.restype = ctypes.c_int
    _user32.FillRect.argtypes = [wintypes.HDC, ctypes.POINTER(wintypes.RECT), HANDLE]
    _user32.DrawTextW.restype = ctypes.c_int
    _user32.DrawTextW.argtypes = [
        wintypes.HDC, wintypes.LPCWSTR, ctypes.c_int,
        ctypes.POINTER(wintypes.RECT), wintypes.UINT,
    ]
    _user32.SetTimer.restype = wintypes.UINT
    _user32.SetTimer.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.UINT, wintypes.LPVOID]
    _user32.KillTimer.restype = wintypes.BOOL
    _user32.KillTimer.argtypes = [wintypes.HWND, wintypes.UINT]
    _user32.InvalidateRect.restype = wintypes.BOOL
    _user32.InvalidateRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT), wintypes.BOOL]
    _user32.PeekMessageW.restype = wintypes.BOOL
    _user32.PeekMessageW.argtypes = [
        ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT, wintypes.UINT
    ]
    _user32.GetSystemMetrics.restype = ctypes.c_int
    _user32.GetSystemMetrics.argtypes = [ctypes.c_int]
    _user32.SetWindowPos.restype = wintypes.BOOL
    _user32.SetWindowPos.argtypes = [
        wintypes.HWND, HANDLE, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.UINT
    ]
    _user32.RedrawWindow.restype = wintypes.BOOL
    _user32.RedrawWindow.argtypes = [
        wintypes.HWND, ctypes.POINTER(wintypes.RECT), HANDLE, wintypes.UINT
    ]
    _user32.GetWindowRect.restype = wintypes.BOOL
    _user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]

    _user32.SetWindowRgn.restype = ctypes.c_int
    _user32.SetWindowRgn.argtypes = [wintypes.HWND, HANDLE, wintypes.BOOL]

    _gdi32.CreateSolidBrush.restype = HANDLE
    _gdi32.CreateSolidBrush.argtypes = [wintypes.COLORREF]
    _gdi32.CreateRoundRectRgn.restype = HANDLE
    _gdi32.CreateRoundRectRgn.argtypes = [ctypes.c_int] * 6
    _gdi32.CreateFontW.restype = HANDLE
    _gdi32.CreateFontW.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD,
        wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD,
        wintypes.LPCWSTR,
    ]
    _gdi32.CreatePen.restype = HANDLE
    _gdi32.CreatePen.argtypes = [ctypes.c_int, ctypes.c_int, wintypes.COLORREF]
    _gdi32.SelectObject.restype = HANDLE
    _gdi32.SelectObject.argtypes = [wintypes.HDC, HANDLE]
    _gdi32.DeleteObject.restype = wintypes.BOOL
    _gdi32.DeleteObject.argtypes = [HANDLE]
    _gdi32.GetStockObject.restype = HANDLE
    _gdi32.GetStockObject.argtypes = [ctypes.c_int]
    _gdi32.SetBkMode.restype = ctypes.c_int
    _gdi32.SetBkMode.argtypes = [wintypes.HDC, ctypes.c_int]
    _gdi32.SetTextColor.restype = wintypes.COLORREF
    _gdi32.SetTextColor.argtypes = [wintypes.HDC, wintypes.COLORREF]
    _gdi32.Arc.restype = wintypes.BOOL
    _gdi32.Arc.argtypes = [wintypes.HDC] + [ctypes.c_int] * 8

    _kernel32.GetModuleHandleW.restype = wintypes.HMODULE
    _kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]

    _class_name = "BlindWatermarkGUI_NativeSplash.2"
    _brush_bg = None
    _font_title = None
    _font_subtitle = None
    _font_status = None
    _font_hint = None
    _font_small = None


def _colorref(rgb: tuple[int, int, int]) -> int:
    """RGB 元组转 COLORREF（0x00BBGGRR）。"""
    return rgb[2] << 16 | rgb[1] << 8 | rgb[0]


def _scale(value: int) -> int:
    return int(round(value * _state.dpi_scale))


if _ENABLED:
    def _wndproc(hwnd, msg, wparam, lparam):
        try:
            if msg == _WM_PAINT:
                ps = PAINTSTRUCT()
                hdc = _user32.BeginPaint(hwnd, ctypes.byref(ps))
                if hdc:
                    _draw(hdc)
                    _user32.EndPaint(hwnd, ctypes.byref(ps))
                return 0
            if msg == _WM_PRINTCLIENT:
                # PrintWindow / 截图工具取窗口内容时走这里。没有这个分支，
                # 抓到的是空白色块（启动画面本身正常，只是画不出去）。
                _draw(ctypes.c_void_p(wparam))
                return 0
            if msg == _WM_ERASEBKGND:
                return 1
            if msg == _WM_TIMER:
                _state.spinner_angle = (_state.spinner_angle + _SPINNER_STEP_DEG) % 360
                _user32.InvalidateRect(hwnd, None, False)
                return 0
            if msg == _WM_DESTROY:
                return 0
        except Exception:
            pass
        return _user32.DefWindowProcW(hwnd, msg, wparam, lparam)

    _wndproc_ref = _WNDPROC(_wndproc)

    def _draw(hdc) -> None:
        """把所有内容画到给定 DC。WM_PAINT 与 WM_PRINTCLIENT 共用这一条路径。"""
        # 背景（整块客户区，保证清除上一帧内容）
        rect = wintypes.RECT(0, 0, _scale(_WINDOW_W), _scale(_WINDOW_H))
        _user32.FillRect(hdc, ctypes.byref(rect), _brush_bg)
        _gdi32.SetBkMode(hdc, _TRANSPARENT)

        _draw_text(hdc, _state.title, 30, 36, _CLR_TITLE, _font_title)
        _draw_text(hdc, _state.subtitle, 68, 24, _CLR_SUBTITLE, _font_subtitle)
        if _state.version_text:
            _draw_text(hdc, _state.version_text, 92, 20, _CLR_VERSION, _font_small)

        _draw_spinner(hdc)
        _draw_text(hdc, _state.status_text, 186, 24, _CLR_STATUS, _font_status)
        _draw_text(hdc, "请勿重复点击程序", 214, 20, _CLR_HINT, _font_hint)

    def _draw_text(hdc, text: str, y: int, height: int, color, font) -> None:
        rect = wintypes.RECT(0, _scale(y), _scale(_WINDOW_W), _scale(y + height))
        old_font = _gdi32.SelectObject(hdc, font)
        _gdi32.SetTextColor(hdc, _colorref(color))
        _user32.DrawTextW(
            hdc, text, -1, ctypes.byref(rect),
            _DT_CENTER | _DT_VCENTER | _DT_SINGLELINE | _DT_NOPREFIX,
        )
        _gdi32.SelectObject(hdc, old_font)

    def _draw_spinner(hdc) -> None:
        cx, cy = _scale(_WINDOW_W // 2), _scale(152)
        radius = _scale(15)
        null_brush = _gdi32.GetStockObject(_NULL_BRUSH)
        old_brush = _gdi32.SelectObject(hdc, null_brush)
        for index in range(4):
            shade = 235 - index * 30  # 亮度递减形成拖尾，但不至于暗到看不见
            pen = _gdi32.CreatePen(0, max(2, _scale(3)), _colorref((shade, shade, shade)))
            if not pen:
                continue
            old_pen = _gdi32.SelectObject(hdc, pen)
            start = (_state.spinner_angle + index * 90) % 360
            end = start + 55
            _gdi32.Arc(
                hdc,
                cx - radius, cy - radius, cx + radius, cy + radius,
                cx + int(radius * math.cos(math.radians(start))),
                cy + int(radius * math.sin(math.radians(start))),
                cx + int(radius * math.cos(math.radians(end))),
                cy + int(radius * math.sin(math.radians(end))),
            )
            _gdi32.SelectObject(hdc, old_pen)
            _gdi32.DeleteObject(pen)
        _gdi32.SelectObject(hdc, old_brush)


def _init_dpi_awareness() -> float:
    """按显示器设置 DPI 感知，返回缩放系数；任何失败都回退 1.0。"""
    try:
        # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 (Win10 1703+)
        _user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except Exception:
        try:
            ctypes.WinDLL("shcore", use_last_error=True).SetProcessDpiAwareness(2)
        except Exception:
            try:
                _user32.SetProcessDPIAware()
            except Exception:
                return 1.0
    try:
        dpi = _user32.GetDpiForSystem()
        if dpi:
            return dpi / 96.0
    except Exception:
        pass
    return 1.0


def _wndproc_thread() -> None:
    """后台消息泵：让启动画面在主线程忙于导入时也保持响应与动画。

    窗口消息由哪个线程 Dispatch 就由哪个线程执行 WndProc。本 WndProc 只做
    GDI 绘制与状态读取，不涉及 Tk/Qt 的线程亲和性约束，因此放在独立线程里
    是安全的；主线程需要时也可以直接调用 pump()。
    """
    while _state.enabled:
        pump()
        time.sleep(0.05)


def _debug(msg: str) -> None:
    """启动画面调试日志：仅在设置 BW_SPLASH_DEBUG=1 时写文件，用于排查打包环境问题。"""
    if os.environ.get("BW_SPLASH_DEBUG") != "1":
        return
    try:
        import tempfile

        path = os.path.join(tempfile.gettempdir(), "bw_splash_debug.log")
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    except Exception:
        pass


def create(version_text: str = "") -> bool:
    """创建并显示启动画面。返回是否成功；失败时调用方照常继续启动。"""
    if not _ENABLED or _state.enabled:
        return False
    try:
        global _brush_bg, _font_title, _font_subtitle, _font_status, _font_hint, _font_small
        _state.version_text = f"v{version_text}" if version_text else ""
        _state.dpi_scale = _init_dpi_awareness()
        _debug(f"begin dpi_scale={_state.dpi_scale}")

        _brush_bg = _gdi32.CreateSolidBrush(_colorref(_CLR_BG))
        _font_title = _gdi32.CreateFontW(
            -_scale(26), 0, 0, 0, 700, 0, 0, 0, 1, 4, 0, 0, 0, _FACE)
        _font_subtitle = _gdi32.CreateFontW(
            -_scale(13), 0, 0, 0, 400, 0, 0, 0, 1, 4, 0, 0, 0, _FACE)
        _font_status = _gdi32.CreateFontW(
            -_scale(13), 0, 0, 0, 400, 0, 0, 0, 1, 4, 0, 0, 0, _FACE)
        _font_hint = _gdi32.CreateFontW(
            -_scale(11), 0, 0, 0, 400, 0, 0, 0, 1, 4, 0, 0, 0, _FACE)
        _font_small = _gdi32.CreateFontW(
            -_scale(11), 0, 0, 0, 400, 0, 0, 0, 1, 4, 0, 0, 0, _FACE)
        if not (_brush_bg and _font_title and _font_status):
            return False

        wc = WNDCLASSW()
        wc.style = 0
        wc.lpfnWndProc = _wndproc_ref
        wc.hInstance = _kernel32.GetModuleHandleW(None)
        wc.hbrBackground = _brush_bg
        wc.lpszClassName = _class_name
        if not _user32.RegisterClassW(ctypes.byref(wc)):
            if ctypes.get_last_error() != _ERROR_CLASS_ALREADY_EXISTS:
                return False

        width, height = _scale(_WINDOW_W), _scale(_WINDOW_H)
        x = max(0, (_user32.GetSystemMetrics(0) - width) // 2)
        y = max(0, (_user32.GetSystemMetrics(1) - height) // 2)
        # 排查用：BW_SPLASH_POS="x,y" 可强制指定位置（例如避开所有前台窗口验证渲染）
        _forced = os.environ.get("BW_SPLASH_POS", "")
        if "," in _forced:
            try:
                _fx, _fy = (int(v.strip()) for v in _forced.split(",", 1))
                x, y = _fx, _fy
            except ValueError:
                pass

        hwnd = _user32.CreateWindowExW(
            _WS_EX_TOPMOST | _WS_EX_TOOLWINDOW,
            _class_name,
            "正在启动 BlindWatermarkGUI",
            _WS_POPUP,
            x, y, width, height,
            None, None, wc.hInstance, None,
        )
        if not hwnd:
            _debug("CreateWindowExW 返回空")
            return False

        rgn = None
        if os.environ.get("BW_SPLASH_NO_RGN") != "1":
            rgn = _gdi32.CreateRoundRectRgn(0, 0, width + 1, height + 1, _scale(12), _scale(12))
        rgn_ok = _user32.SetWindowRgn(hwnd, rgn, True) if rgn else None

        _state.hwnd = hwnd
        _state.opened_at = time.perf_counter()
        _user32.ShowWindow(hwnd, _SW_SHOWNOACTIVATE)
        _user32.UpdateWindow(hwnd)
        _user32.SetTimer(hwnd, 1, _TIMER_MS, None)

        # 显式再声明一次 TOPMOST 并强制刷新：打包环境里仅靠创建时的
        # WS_EX_TOPMOST 有时不会把窗口提到前台窗口之上（例如前台是全屏/最大化窗口）。
        _user32.SetWindowPos(
            hwnd, ctypes.c_void_p(-1),  # HWND_TOPMOST
            0, 0, 0, 0, 0x0001 | 0x0002 | 0x0010,  # SWP_NOSIZE|SWP_NOMOVE|SWP_NOACTIVATE
        )
        _user32.RedrawWindow(hwnd, None, None, 0x0001 | 0x0100 | 0x0080)

        _state.enabled = True
        _state.spinner_angle = 0
        threading.Thread(
            target=_wndproc_thread, name="native-splash-pump", daemon=True
        ).start()

        rect = wintypes.RECT()
        _user32.GetWindowRect(hwnd, ctypes.byref(rect))
        _debug(
            f"created hwnd={hwnd} geom=({x},{y},{width},{height}) "
            f"rect=({rect.left},{rect.top},{rect.right},{rect.bottom}) "
            f"rgn={rgn} rgn_set={rgn_ok} visible={_user32.IsWindowVisible(hwnd)}"
        )
        return True
    except Exception as e:
        # 启动画面失败不能影响主流程，但必须留下痕迹便于排查
        import traceback
        print(f"[启动画面] 创建失败，将跳过启动画面: {type(e).__name__}: {e}")
        traceback.print_exc()
        _state.enabled = False
        _state.hwnd = None
        return False


def set_status(text: str) -> None:
    """更新启动画面状态文字（可在主线程任意时刻调用）。"""
    if not _state.enabled:
        return
    _state.status_text = text
    try:
        _user32.InvalidateRect(_state.hwnd, None, False)
    except Exception:
        pass


def pump() -> None:
    """处理挂起的窗口消息，保持启动画面响应且不阻塞。

    在长耗时导入之间调用即可：窗口动画由定时器驱动，不会出现"无响应"。
    """
    if not _state.enabled:
        return
    try:
        msg = wintypes.MSG()
        while _user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, _PM_REMOVE):
            _user32.TranslateMessage(ctypes.byref(msg))
            _user32.DispatchMessageW(ctypes.byref(msg))
    except Exception:
        pass


def close() -> None:
    """销毁启动画面并返回自创建以来的毫秒数。可重复调用。"""
    if not _state.enabled:
        return 0
    hwnd = _state.hwnd
    _state.enabled = False
    _state.hwnd = None
    ms = 0
    if _state.opened_at is not None:
        ms = int((time.perf_counter() - _state.opened_at) * 1000)
        _state.opened_at = None
    try:
        if hwnd:
            _user32.KillTimer(hwnd, 1)
            _user32.DestroyWindow(hwnd)
            pump()
    except Exception:
        pass
    return ms
