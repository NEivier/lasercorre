import threading
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtGui import QColor, QPalette
from matplotlib import pyplot as plt
from sklearn.ensemble import IsolationForest
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import time
from datetime import datetime
import os
import webbrowser
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import tempfile
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import pandas as pd
from scipy import stats
from pyqtgraph.opengl import GLViewWidget, GLSurfacePlotItem

# ---------------------
# 模块0：设备接口
# ---------------------
class DeviceNotConnectedError(Exception):
    pass


class VirtualLaserSensor:
    def __init__(self):
        self.t = 0

    def read_data(self, sample_size=100):
        self.t += 1
        raw_data = np.random.normal(5, 1, sample_size) + np.sin(np.linspace(0, 4 * np.pi * self.t, sample_size)) * 5
        return raw_data


class RealLaserSensor:
    def __init__(self):
        if not self._check_connection():
            raise DeviceNotConnectedError("设备未连接")
        self.t = 0

    def _check_connection(self):
        return False  # 修改这里为True测试真实设备模式

    def read_data(self, sample_size=100):
        self.t += 1
        return np.random.normal(3, 0.8, sample_size) + np.cos(np.linspace(0, 2 * np.pi * self.t, sample_size)) * 4


class DataFetcher:
    def fetch_data(self):
        raise NotImplementedError


class VirtualDataFetcher(DataFetcher):
    def __init__(self):
        self.sensor = VirtualLaserSensor()

    def fetch_data(self):
        return self.sensor.read_data()


class RealDataFetcher(DataFetcher):
    def __init__(self):
        try:
            self.sensor = RealLaserSensor()
        except DeviceNotConnectedError:
            raise

    def fetch_data(self):
        return self.sensor.read_data()


# ---------------------
# 模块1：数据处理核心（保持不变）
# ---------------------
class LaserDataProcessor:
    def __init__(self, data_fetcher=None):
        self.calibration_params = {'f1': 1.2, 'f2': 0.8}
        self.env_factors = {
            'temp': 25,
            'humidity': 50,
            'omega': 0.0,  # 旋转角频率
            'theta': 0.0,  # 预设标准夹角
            'r': 0.0,  # 横向距离
            'h0': 1.0,  # 竖向间距
            'ti': 0.0,  # 上方光敏元件响应时间
            'tj': 0.0  # 下方光敏元件响应时间
        }

        if data_fetcher is None:
            try:
                self.data_fetcher = RealDataFetcher()
                self.data_source = "真实设备"
            except DeviceNotConnectedError:
                self.data_fetcher = VirtualDataFetcher()
                self.data_source = "模拟数据"
        else:
            self.data_fetcher = data_fetcher
            self.data_source = "自定义数据源"

    def get_raw_data(self):
        return self.data_fetcher.fetch_data()

    def adaptive_filter(self, raw_data):
        cleaned = np.convolve(raw_data, [0.2, 0.6, 0.2], mode='same')
        clf = IsolationForest(contamination=0.1)
        anomalies = clf.fit_predict(cleaned.reshape(-1, 1))
        cleaned[anomalies == -1] = np.median(cleaned)
        return cleaned

    def dynamic_calibration(self, data):
        temp_factor = 1 + (self.env_factors['temp'] - 25) * 0.01
        humidity_factor = 1 + (self.env_factors['humidity'] - 50) * 0.005
        return data * self.calibration_params['f1'] * temp_factor + self.calibration_params['f2'] * humidity_factor

    def analyze_calibration(self, raw, filtered, calibrated):
        """执行全面的校准效果分析"""
        analysis = {}

        # 基础统计
        analysis['raw_stats'] = self._calculate_basic_stats(raw)
        analysis['filtered_stats'] = self._calculate_basic_stats(filtered)
        analysis['calibrated_stats'] = self._calculate_basic_stats(calibrated)

        # 校准效果指标
        analysis['improvement_ratio'] = {
            'std': self._calc_improvement(raw, calibrated, 'std'),
            'rms': self._calc_improvement(raw, calibrated, 'rms'),
            'dynamic_range': (analysis['calibrated_stats']['range'] /
                              analysis['raw_stats']['range'] - 1)
        }

        # 残差分析
        residuals = calibrated - filtered
        analysis['residual_analysis'] = {
            'max_abs': np.max(np.abs(residuals)),
            'std': np.std(residuals),
            'skewness': stats.skew(residuals),
            'kurtosis': stats.kurtosis(residuals),
            'norm_test': stats.normaltest(residuals)[1]  # p-value
        }

        # 趋势分析
        analysis['trend_analysis'] = self._calculate_trend(calibrated)

        # 频域分析
        analysis['frequency_analysis'] = self._frequency_domain_analysis(
            raw, calibrated)

        return analysis

    def _calculate_basic_stats(self, data):
        """计算基本统计指标"""
        return {
            'mean': np.nanmean(data),
            'std': np.nanstd(data),
            'range': np.ptp(data),
            'rms': np.sqrt(np.mean(np.square(data))),
            'snr': 10 * np.log10(np.mean(data ** 2) / np.var(data))
        }

    def _calc_improvement(self, raw, calibrated, metric):
        """计算改进比例"""
        raw_val = self._calculate_basic_stats(raw)[metric]
        cal_val = self._calculate_basic_stats(calibrated)[metric]
        return (raw_val - cal_val) / raw_val

    def _calculate_trend(self, data, window_size=20):
        """使用滑动窗口进行趋势分析"""
        df = pd.DataFrame({'value': data})
        df['rolling_mean'] = df['value'].rolling(window=window_size).mean()
        df['rolling_std'] = df['value'].rolling(window=window_size).std()

        return {
            'max_slope': np.max(np.abs(np.diff(df['rolling_mean'].dropna()))),
            'stability': df['rolling_std'].mean() / df['value'].mean(),
            'trend_stability': np.polyfit(
                df.index[window_size - 1:],
                df['rolling_mean'].dropna(),
                1)[0]  # 线性趋势斜率
        }

    def _frequency_domain_analysis(self, raw, calibrated):
        """频域分析"""
        fft_raw = np.abs(np.fft.fft(raw))
        fft_cal = np.abs(np.fft.fft(calibrated))
        freq = np.fft.fftfreq(len(raw))

        return {
            'dominant_freq_raw': freq[np.argmax(fft_raw)],
            'dominant_freq_cal': freq[np.argmax(fft_cal)],
            'noise_energy_ratio': (np.sum(fft_cal[fft_cal < np.median(fft_cal)]) /
                                   np.sum(fft_raw[fft_raw < np.median(fft_raw)]))
        }


class DataFetcherManager(QObject):
    data_source_changed = pyqtSignal(str)  # 定义信号

    def __init__(self, gui):
        super().__init__()  # 确保调用 QObject 的初始化
        self.gui = gui
        self.current_data_fetcher = None

    def switch_to_real_data(self):
        """切换到真实设备数据源"""
        try:
            real_fetcher = RealDataFetcher()
            self.current_data_fetcher = real_fetcher
            self.gui.processor.data_fetcher = real_fetcher
            self.gui.processor.data_source = "真实设备"
            self.data_source_changed.emit("真实设备")  # 发送信号
        except DeviceNotConnectedError:
            QtWidgets.QMessageBox.critical(self.gui, "错误", "无法连接到真实设备！")

    def switch_to_virtual_data(self):
        """切换到虚拟数据源"""
        virtual_fetcher = VirtualDataFetcher()
        self.current_data_fetcher = virtual_fetcher
        self.gui.processor.data_fetcher = virtual_fetcher
        self.gui.processor.data_source = "模拟数据"
        self.data_source_changed.emit("模拟数据")  # 发送信号


# ---------------------
# 模块2：增强GUI界面
# ---------------------
class EnhancedLaserGUI(QtWidgets.QMainWindow):
    update_gui_signal = QtCore.pyqtSignal(np.ndarray, np.ndarray, np.ndarray, dict)  # 新增信号
    calibration_success_signal = QtCore.pyqtSignal(str)
    parameter_updated = QtCore.pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.processor = LaserDataProcessor()
        self.init_ui()
        self.setup_connections()
        self.init_styles()
        self.history_data = []
        self.real_time_mode = False
        self.real_time_thread = None
        self.should_stop_real_time = False
        # 新增：硬件连接管理器
        self.data_fetcher_manager = DataFetcherManager(self)  # 确保此行存在
        self.data_fetcher_manager.data_source_changed.connect(self.on_data_source_changed)  # 确保已连接信号
        # 状态栏初始化
        self.statusBar().showMessage(f"数据来源: {self.processor.data_source} | 系统就绪")
        pdfmetrics.registerFont(TTFont('Simhei', 'simhei.ttf'))
        self.stop_event = threading.Event()  # 替换原来的布尔标志
        self.update_gui_signal.connect(self.handle_realtime_update)  # 连接信号到处理函数
        self.report_dir = "reports"
        os.makedirs(self.report_dir, exist_ok=True)

    def on_data_source_changed(self, data_source):
        """数据源变更时的回调函数"""
        self.processor.data_source = data_source
        self.statusBar().showMessage(f"数据来源: {data_source} | 系统就绪")

    def init_ui(self):
        # 主容器
        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QtWidgets.QHBoxLayout(main_widget)
        # 左侧面板
        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel
                                            )
        # 左侧3D视图
        self.gl_view = GLViewWidget()
        self.gl_view.setBackgroundColor('#1A1A1A')
        self.init_3d_grid()  # 初始化网格
        left_layout.addWidget(self.gl_view, 3)

        # 右侧面板
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)

        # 2D绘图区
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#1E1E1E')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel('left', '信号强度', color='white', font_size=5)
        self.plot_widget.setLabel('bottom', '时间索引', color='white', font_size=5)
        right_layout.addWidget(self.plot_widget, 3)

        # 诊断面板
        self.diagnosis_text = QtWidgets.QTextEdit()
        self.diagnosis_text.setReadOnly(True)
        right_layout.addWidget(self.diagnosis_text, 2)
        # 添加硬件连接功能模块
        self.hardware_control_panel = QtWidgets.QWidget()
        self.hardware_control_panel.setStyleSheet("background-color: #3C3C3C;")
        hardware_layout = QtWidgets.QVBoxLayout(self.hardware_control_panel)
        left_layout.addWidget(self.hardware_control_panel, 2)

        # 初始化硬件控制面板
        self.init_hardware_control_panel(hardware_layout)
        main_layout.addWidget(left_panel, 5)
        main_layout.addWidget(right_panel, 6)
        # 初始化控制面板
        self.init_control_dock()

    def init_hardware_control_panel(self, layout):
        """初始化硬件控制面板"""
        # 硬件参数显示区域
        self.temp_label = QtWidgets.QLabel("温度：25.0 ℃")
        self.humidity_label = QtWidgets.QLabel("湿度：50.0 %")
        white_color = QColor(255, 255, 255)
        new_palette = self.temp_label.palette()
        new_palette.setColor(QPalette.WindowText, white_color)
        self.temp_label.setPalette(new_palette)
        new_palette = self.humidity_label.palette()
        new_palette.setColor(QPalette.WindowText, white_color)
        self.humidity_label.setPalette(new_palette)
        layout.addWidget(self.create_group("环境参数", [self.temp_label, self.humidity_label]))
        # 添加参数显示标签
        self.formula_label = QtWidgets.QLabel(
            "计算公式：Δθ= f1×arctan(f2×(ti-tj)×ω×r/h0) - θ"
        )
        self.formula_label.setStyleSheet("font-size: 14px; color: white;")
        self.omega_label = QtWidgets.QLabel("旋转激光发射器旋转角频率 (ω): 0.0 rad/s")
        self.theta_label = QtWidgets.QLabel("预设标准夹角 (θ): 0.0°")
        self.r_label = QtWidgets.QLabel("横向距离 (r): 0.0 m")
        self.h0_label = QtWidgets.QLabel("竖向间距 (h0): 0.0 m")
        self.ti_label = QtWidgets.QLabel("上方光敏元件响应时间 (ti): 0.0 s")
        self.tj_label = QtWidgets.QLabel("下方光敏元件响应时间 (tj): 0.0 s")
        self.result_label = QtWidgets.QLabel("线激光校正角度 (Δθ): 0.0°")
        # 创建样式表来设置字体颜色为白色
        style_sheet = "color: white;"
        # 将样式表应用到每个标签
        self.omega_label.setStyleSheet(style_sheet)
        self.theta_label.setStyleSheet(style_sheet)
        self.r_label.setStyleSheet(style_sheet)
        self.h0_label.setStyleSheet(style_sheet)
        self.ti_label.setStyleSheet(style_sheet)
        self.tj_label.setStyleSheet(style_sheet)
        self.result_label.setStyleSheet(style_sheet)
        layout.addWidget(
            self.create_group("激光检测装置检测参数", [self.formula_label, self.omega_label, self.theta_label, self.r_label, self.h0_label,
                                                       self.ti_label, self.tj_label, self.result_label]))

        # 硬件连接控制按钮
        self.connect_device_btn = QtWidgets.QPushButton("连接真实设备")
        self.connect_device_btn.setStyleSheet("background-color: #4A4A4A; color: white;")
        self.connect_device_btn.clicked.connect(self.connect_device)
        layout.addWidget(self.connect_device_btn)

        self.disconnect_device_btn = QtWidgets.QPushButton("断开设备")
        self.disconnect_device_btn.setStyleSheet("background-color: #4A4A4A; color: white;")
        self.disconnect_device_btn.clicked.connect(self.disconnect_device)
        self.disconnect_device_btn.setEnabled(False)
        layout.addWidget(self.disconnect_device_btn)

        # 硬件状态显示
        self.hardware_status_label = QtWidgets.QLabel("硬件状态：未连接")
        self.hardware_status_label.setStyleSheet("color: white;")
        layout.addWidget(self.hardware_status_label)

    def connect_device(self):
        """连接真实设备"""
        try:
            # 模拟真实设备连接
            self.data_fetcher_manager.switch_to_real_data()
            self.hardware_status_label.setText("硬件状态：已连接")
            self.hardware_status_label.setStyleSheet("color: #4CAF50;")
            self.connect_device_btn.setEnabled(False)
            self.disconnect_device_btn.setEnabled(True)
            self.processor.data_source = "设备检测参数"

            # 更新硬件参数
            self.update_hardware_params()

            # 弹出提示信息
            QtWidgets.QMessageBox.information(self, "提示", "成功连接到真实设备！")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "错误", f"连接真实设备失败：{str(e)}")

    def update_hardware_params(self):
        """更新硬件参数"""
        # 模拟从硬件设备获取参数
        self.omega_label.setText(f"旋转角频率 (ω): {self.processor.env_factors['omega']:.2f} rad/s")
        self.theta_label.setText(f"预设夹角 (θ): {self.processor.env_factors['theta']:.2f}°")
        self.r_label.setText(f"横向距离 (r): {self.processor.env_factors['r']:.2f} m")
        self.h0_label.setText(f"竖向间距 (h0): {self.processor.env_factors['h0']:.2f} m")
        self.ti_label.setText(f"上方光敏元件响应时间 (ti): {self.processor.env_factors['ti']:.2f} s")
        self.tj_label.setText(f"下方光敏元件响应时间 (tj): {self.processor.env_factors['tj']:.2f} s")

    def disconnect_device(self):
        """断开硬件设备"""
        try:
            # 模拟硬件断开
            self.hardware_status_label.setText("硬件状态：未连接")
            self.hardware_status_label.setStyleSheet("color: white;")
            QtWidgets.QMessageBox.information(self, "提示", "硬件设备已成功断开！")
            self.processor.data_source = "模拟数据"
            self.connect_device_btn.setEnabled(True)
            self.disconnect_device_btn.setEnabled(False)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "错误", f"断开硬件设备失败：{str(e)}")

    def init_3d_grid(self):
        """初始化3D参考网格（正确维度版本）"""
        # 生成一维坐标
        x = np.linspace(-8, 8, 50)  # shape (50,)
        y = np.linspace(-8, 8, 50)  # shape (50,)

        # 生成二维Z数据（关键修正点）
        z = np.zeros((len(y), len(x)))  # shape (50,50)
        for i in range(len(x)):
            for j in range(len(y)):
                z[j, i] = 0.1 * (x[i] ** 2 + y[j] ** 2)  # 注意索引顺序

        # 正确初始化表面数据
        self.grid = GLSurfacePlotItem(
            x=x,  # 一维数组
            y=y,  # 一维数组
            z=z,  # 二维数组 (len(y), len(x))
            colors=(0.2, 0.5, 1.0, 0.3),  # RGBA
            drawEdges=False,
            shader='shaded'
        )
        self.grid.translate(0, 0, -5)
        self.gl_view.addItem(self.grid)

    def update_3d_view(self, data):
        """更新光束显示（兼容性修正）"""
        # 生成一维坐标
        x = np.linspace(-8, 8, len(data))  # shape (N,)
        z = data * 0.1  # shape (N,)
        y = np.zeros_like(x)  # shape (N,)

        # 转换为三维点云
        pts = np.column_stack((x, y, z))  # shape (N,3)

        # 移除旧的光束
        if hasattr(self, 'beam_plot'):
            self.gl_view.removeItem(self.beam_plot)

        # 创建新的光束对象
        self.beam_plot = pg.opengl.GLLinePlotItem(
            pos=pts,
            color=(0, 255, 173, 255),  # RGBA格式
            width=4.0,
            antialias=True,
            mode='line_strip'
        )
        self.gl_view.addItem(self.beam_plot)

    def init_control_dock(self):
        """初始化控制面板（带背景色设置）"""
        self.control_panel = QtWidgets.QWidget()

        # 设置主面板背景色
        self.control_panel.setStyleSheet("""
                QWidget {
                    background-color: #2D2D2D;
                    color: #FFFFFF;
                }
                QGroupBox {
                    border: 1px solid #4A4A4A;
                    margin-top: 3px;
                }
                QGroupBox::title {
                    color: #A0A0A0;
                }
                QTabWidget::pane {
                    border: 0;
                    background: #2D2D2D;
                }
                QTabBar::tab {
                    background: #2D2D2D;
                    color: white;
                    padding: 8px;
                    border-top-left-radius: 4px;
                    border-top-right-radius: 4px;
                }
                QTabBar::tab:selected {
                    background: #404040;
                    border-bottom: 2px solid #3E8AF0;
                }
            """)

        main_layout = QtWidgets.QVBoxLayout(self.control_panel)
        main_layout.setContentsMargins(1, 1, 1, 1)  # 减少主布局边距

        # 创建带滚动区域的容器（设置滚动区域背景）
        scroll = QtWidgets.QScrollArea()
        scroll.setStyleSheet("""
                QScrollArea {
                    background: transparent;
                    border: none;
                }
                QScrollBar:horizontal {  /* 隐藏横向滚动条 */
                    height: 0px;
                }
                QScrollBar:vertical {
                    background: #2D2D2D;
                    width: 2px;
                }
                QScrollBar::handle:vertical {
                    background: #404040;
                    min-height: 4px;
                }
            """)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)  # 禁用横向滚动
        scroll.setWidgetResizable(True)

        # 内容容器背景设置
        content = QtWidgets.QWidget()
        content.setStyleSheet("background-color: #2D2D2D;")  # 与主面板一致
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(3, 3, 3, 3)  # 减少边距
        # 选项卡容器
        tab_widget = QtWidgets.QTabWidget()
        # tab_widget.setStyleSheet("background-color: #2D2D2D;")  # 与主面板一致
        # 选项卡1：校准参数
        calib_tab = QtWidgets.QWidget()
        calib_layout = QtWidgets.QVBoxLayout(calib_tab)
        self.f1_slider = self.create_slider("环境系数 F1", 0.5, 2.0, 1.2)
        self.f2_slider = self.create_slider("校准系数 F2", 0.1, 1.5, 0.8)
        calib_layout.addWidget(self.create_group("核心参数", [self.f1_slider, self.f2_slider]))

        # 选项卡2：环境参数（新增折叠面板）
        env_tab = QtWidgets.QWidget()
        env_layout = QtWidgets.QVBoxLayout(env_tab)
        # 折叠面板 - 几何参数
        self.r_slider = self.create_slider("横向距离r/m ", 0.0, 150.0, 0.0)
        self.h0_slider = self.create_slider("竖向间距h0/cm", 0.1, 100.0, 1.0)
        env_layout.addWidget(self.create_group("几何参数", [self.r_slider, self.h0_slider]))
        # 折叠面板 - 运动参数
        self.omega_slider = self.create_slider("角频率ω(rad/s)", 0.0, 100.0, 0.0)
        self.theta_slider = self.create_slider("标准夹角θ/°", 0.0, 90.0, 0.0)
        env_layout.addWidget(self.create_group("运动参数", [self.omega_slider, self.theta_slider]))
        env_layout.addStretch()

        # 添加选项卡
        tab_widget.addTab(calib_tab, "校准设置")
        tab_widget.addTab(env_tab, "环境参数")

        layout.addWidget(tab_widget)
        # 操作按钮组
        self.start_btn = self.create_button("测试校准", "#2196F3", "play")
        self.export_btn = self.create_button("生成报告", "#2196F3", "report")
        self.real_time_check = QtWidgets.QCheckBox("实时模式")
        self.real_time_check.setStyleSheet("color: white;")
        self.view_reports_btn = self.create_button("查看历史报告", "#3E8AF0", "save")
        # 增加停止按钮和退出按钮
        self.stop_btn = self.create_button("停止预测", "#FF5722", "stop")
        self.exit_btn = self.create_button("退出系统", "#D32F2F", "exit")
        self.stop_btn.clicked.connect(self.stop_real_time)
        self.exit_btn.clicked.connect(self.close)
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.export_btn)
        layout.addLayout(btn_layout)
        # 测试按钮
        test_btn = self.create_button("测试3D系统", "#3E8AF0", "test")
        test_btn.clicked.connect(self.test_3d)
        btn1_layout = QtWidgets.QHBoxLayout()
        btn1_layout.addWidget(test_btn)
        btn1_layout.addWidget(self.view_reports_btn)
        layout.addLayout(btn1_layout)
        # 在控制面板初始化部分添加频率调节
        self.freq_slider = self.create_slider("采样频率(Hz)", 0.5, 30.0, 2.0)
        # layout.insertWidget(2, self.create_group("采样设置", [self.freq_slider]))
        layout.addWidget(self.create_group("实时模式", [self.real_time_check, self.freq_slider, self.stop_btn]))
        white_color = QColor(255, 255, 255)
        layout.addWidget(self.exit_btn)
        # 状态指示器
        self.status_indicator = QtWidgets.QLabel()
        new_palette = self.status_indicator.palette()
        new_palette.setColor(QPalette.WindowText, white_color)
        self.status_indicator.setPalette(new_palette)
        self.status_indicator.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.create_group("状态指示器(绿色|正常;红色|异常)", [self.status_indicator]))
        scroll.setWidget(content)
        main_layout.addWidget(scroll)
        # 确保子控件背景透明
        self.control_panel.setAutoFillBackground(True)
        content.setAutoFillBackground(True)
        # 创建停靠窗口
        dock = QtWidgets.QDockWidget("控制面板", self)
        dock.setStyleSheet("QDockWidget { color: white; }")
        dock.setWidget(self.control_panel)

        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
        # 新增状态栏初始化
        self.statusBar().setStyleSheet("background-color: #3C3C3C; color: white;")

    def create_slider(self, title, min_val, max_val, default):
        """创建带数字输入框的参数滑块"""
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)

        # 创建标签（保留原有样式设置）
        label = QtWidgets.QLabel(f"{title}: {default:.2f}")
        new_palette = label.palette()
        white_color = QColor(255, 255, 255)
        new_palette.setColor(QPalette.WindowText, white_color)
        label.setPalette(new_palette)

        # 创建水平布局容器用于滑块和输入框
        control_container = QtWidgets.QWidget()
        control_layout = QtWidgets.QHBoxLayout(control_container)
        control_layout.setContentsMargins(0, 0, 0, 0)  # 移除默认边距

        # 创建滑块
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setRange(int(min_val * 100), int(max_val * 100))
        slider.setValue(int(default * 100))

        # 创建数字输入框
        spinbox = QtWidgets.QDoubleSpinBox()
        spinbox.setMinimum(min_val)
        spinbox.setMaximum(max_val)
        spinbox.setValue(default)
        spinbox.setSingleStep(0.01)
        spinbox.setDecimals(2)
        spinbox.setFixedWidth(80)  # 设置固定宽度

        # 设置输入框文字颜色为白色
        spinbox.setStyleSheet("""
            QDoubleSpinBox {
                color: white;
                background-color: #404040;
            }
        """)

        # 信号连接
        def update_slider(value):
            """当输入框变化时更新滑块"""
            slider.blockSignals(True)
            slider.setValue(int(value * 100))
            label.setText(f"{title}: {value:.2f}")
            slider.blockSignals(False)

        def update_spinbox(value):
            """当滑块变化时更新输入框和标签"""
            spinbox.blockSignals(True)
            spin_value = value / 100
            spinbox.setValue(spin_value)
            label.setText(f"{title}: {spin_value:.2f}")
            spinbox.blockSignals(False)

        # 双向绑定信号
        slider.valueChanged.connect(update_spinbox)
        slider.valueChanged.connect(self.update_parameters)
        spinbox.valueChanged.connect(update_slider)
        spinbox.valueChanged.connect(self.update_parameters)

        # 将控件添加到布局
        control_layout.addWidget(slider)
        control_layout.addWidget(spinbox)
        layout.addWidget(label)
        layout.addWidget(control_container)

        return container

    def create_button(self, text, color, icon_name):
        """创建风格化按钮"""
        btn = QtWidgets.QPushButton(text)
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border-radius: 4px;
                padding: 8px;
                min-width: 100px;
            }}
            QPushButton:hover {{
                background-color: {self.lighten_color(color)};
            }}
        """)
        if icon_name:
            btn.setIcon(QtGui.QIcon.fromTheme(icon_name))
        return btn

    def lighten_color(self, hex_color, factor=0.2):
        """颜色亮度调整"""
        rgb = [int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]
        new_rgb = [min(255, c + int(255 * factor)) for c in rgb]
        return "#{:02x}{:02x}{:02x}".format(*new_rgb)

    def setup_connections(self):
        """只需要连接业务逻辑信号"""
        self.start_btn.clicked.connect(self.start_calibration)
        self.export_btn.clicked.connect(self.export_report)
        self.real_time_check.stateChanged.connect(self.toggle_real_time_mode)
        self.view_reports_btn.clicked.connect(self.view_reports)
        self.parameter_updated.connect(self.on_parameter_update)
        self.calibration_success_signal.connect(self.update_status)

    def test_3d(self):
        """3D系统功能测试"""
        test_data = np.sin(np.linspace(0, 8 * np.pi, 200)) * 3 + 5
        self.update_3d_view(test_data)
        self.statusBar().showMessage("3D测试数据已加载")

    def on_parameter_update(self, params):
        """参数更新处理"""
        self.processor.calibration_params['f1'] = params['f1']
        self.processor.calibration_params['f2'] = params['f2']
        self.processor.env_factors['omega'] = params['omega']
        self.processor.env_factors['theta'] = params['theta']
        self.processor.env_factors['r'] = params['r']
        self.processor.env_factors['h0'] = params['h0']
        self.statusBar().showMessage(
            f"参数已更新：F1={params['f1']:.2f}, F2={params['f2']:.2f},ω={params['omega']:.2f}rad"
            f"/s,θ={params['theta']:.2f}°,r={params['r']:.2f}m,h0={params['h0']:.2f}cm")

    def update_parameters(self):
        """更新处理参数"""
        params = {
            'f1': self.f1_slider.findChild(QtWidgets.QSlider).value() / 100,
            'f2': self.f2_slider.findChild(QtWidgets.QSlider).value() / 100,
            'omega': self.omega_slider.findChild(QtWidgets.QSlider).value() / 100,
            'r': self.r_slider.findChild(QtWidgets.QSlider).value() / 100,
            'theta': self.theta_slider.findChild(QtWidgets.QSlider).value() / 100,
            'h0': self.h0_slider.findChild(QtWidgets.QSlider).value() / 100,
        }
        self.parameter_updated.emit(params)

    def view_reports(self):
        """显示已生成的报告"""
        report_files = [f for f in os.listdir(self.report_dir) if
                        f.startswith("激光校准报告_") and f.endswith(".pdf")]
        if not report_files:
            QtWidgets.QMessageBox.information(self, "提示", "没有生成任何报告！")
            return

        # 弹出选择对话框
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("选择报告")
        layout = QtWidgets.QVBoxLayout()

        list_widget = QtWidgets.QListWidget()
        list_widget.addItems(report_files)
        list_widget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)  # 启用多选
        layout.addWidget(list_widget)

        # 删除选中报告
        delete_btn = QtWidgets.QPushButton("删除选中报告")
        delete_btn.clicked.connect(lambda: self.delete_selected_report(list_widget))
        layout.addWidget(delete_btn)

        # 清空历史
        clear_btn = QtWidgets.QPushButton("清空历史")
        clear_btn.clicked.connect(lambda: self.clear_all_reports(list_widget))
        layout.addWidget(clear_btn)

        # 查看按钮
        view_btn = QtWidgets.QPushButton("查看报告")
        view_btn.clicked.connect(lambda: self.open_report(list_widget.currentItem().text()))
        layout.addWidget(view_btn)

        close_btn = QtWidgets.QPushButton("关闭")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)

        dialog.setLayout(layout)
        dialog.exec_()

    def delete_selected_report(self, list_widget):
        """删除选中的报告"""
        selected_items = list_widget.selectedItems()
        if not selected_items:
            QtWidgets.QMessageBox.warning(self, "警告", "请选择要删除的报告！")
            return

        reply = QtWidgets.QMessageBox.question(
            self, "确认删除",
            f"确定要删除选中的 {len(selected_items)} 个报告吗？",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        if reply == QtWidgets.QMessageBox.Yes:
            for item in selected_items:
                filename = item.text()
                filepath = os.path.join(self.report_dir, filename)
                try:
                    os.remove(filepath)
                    row = list_widget.row(item)
                    list_widget.takeItem(row)
                except Exception as e:
                    QtWidgets.QMessageBox.critical(self, "错误", f"删除报告 '{filename}' 失败！\n错误信息：{str(e)}")
                    break
            QtWidgets.QMessageBox.information(self, "提示", "选中的报告已成功删除！")

    def clear_all_reports(self, list_widget):
        """清空所有历史报告"""
        reply = QtWidgets.QMessageBox.question(
            self, "清空历史",
            "确定要删除所有历史报告吗？",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        if reply == QtWidgets.QMessageBox.Yes:
            try:
                for filename in os.listdir(self.report_dir):
                    if filename.startswith("激光校准报告_") and filename.endswith(".pdf"):
                        filepath = os.path.join(self.report_dir, filename)
                        os.remove(filepath)
                list_widget.clear()  # 清空列表
                QtWidgets.QMessageBox.information(self, "提示", "所有历史报告已成功清除！")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "错误", f"清空历史失败！\n错误信息：{str(e)}")

    def open_report(self, filename):
        """打开报告文件"""
        filepath = os.path.join(self.report_dir, filename)
        if os.path.exists(filepath):
            webbrowser.open(filepath)
        else:
            QtWidgets.QMessageBox.warning(self, "警告", "报告文件不存在！")

    def start_calibration(self):
        """更新状态栏信息"""
        self.statusBar().showMessage(f"数据来源: {self.processor.data_source} | 正在处理数据...")
        try:
            if not self.real_time_mode:
                raw_data = self.processor.get_raw_data()
                filtered = self.processor.adaptive_filter(raw_data)
                calibrated = self.processor.dynamic_calibration(filtered)
                # 在获取数据前更新环境参数
                self.processor.env_factors['temp'] = np.random.normal(25, 0.5)  # 模拟温度变化
                self.processor.env_factors['humidity'] = np.random.normal(50, 2)  # 模拟湿度变化
                self.processor.env_factors['ti'] = np.random.normal(0.002, 0.0001)  # 模拟温度变化
                self.processor.env_factors['tj'] = np.random.normal(0.002, 0.0001)  # 模拟湿度变化

                # 更新界面显示
                self.temp_label.setText(f"温度：{self.processor.env_factors['temp']:.1f} ℃")
                self.humidity_label.setText(f"湿度：{self.processor.env_factors['humidity']:.1f} %")
                self.omega_label.setText(f"旋转激光发射器旋转角频率 (ω): {self.processor.env_factors['omega']:.2f} rad/s")
                self.theta_label.setText(f"预设标准夹角 (θ): {self.processor.env_factors['theta']:.2f}°")
                self.r_label.setText(f"横向距离 (r):  {self.processor.env_factors['r']:.2f} m")
                self.h0_label.setText(f"竖向间距 (h0):  {self.processor.env_factors['h0']:.2f} cm")
                self.ti_label.setText(f"上方光敏元件响应时间 (ti):  {self.processor.env_factors['ti']:.4f} s")
                self.tj_label.setText(f"下方光敏元件响应时间 (tj): {self.processor.env_factors['tj']:.4f} s")
                # self.result_label.setText(f"线激光校正角度 (Δθ): {self.processor.env_factors['omega']:.1f}°")
                # 计算线激光校正角度
                delta_theta = self.calculate_delta_theta()

                # 更新计算结果
                self.result_label.setText(f"线激光校正角度 (Δθ): {delta_theta:.4f}°")
                # 更新2D图表
                self.plot_widget.clear()  # 清除之前的绘图
                # 添加图例
                self.plot_widget.addLegend(offset=(0, 5))  # 直接添加图例并设置偏移
                self.plot_widget.plot(raw_data, pen='r', name="原始信号")  # 绘制原始数据
                self.plot_widget.plot(filtered, pen='g', name="滤波信号")  # 绘制滤波数据
                self.plot_widget.plot(calibrated, pen='b', name="校准信号")  # 绘制校准数据
                # 更新3D视图
                self.update_3d_view(calibrated)

                # 生成诊断信息
                analysis = self.processor.analyze_calibration(raw_data, filtered, calibrated)
                diagnosis = self.generate_diagnosis(analysis)

                # 显示诊断结果
                self.diagnosis_text.setHtml(
                    f"<h3>校准分析报告</h3>"
                    f"<p>数据来源：{self.processor.data_source}</p>"
                    f"<p>标准偏差：{analysis['calibrated_stats']['std']:.4f}</p>"
                    f"<p>信噪比：{analysis['calibrated_stats']['snr']:.2f} dB</p>"
                    f"<h4>诊断建议：</h4><ul>{''.join([f'<li>{d}</li>' for d in diagnosis])}</ul>"
                )

                self.calibration_success_signal.emit("校准成功！系统状态正常")
            else:
                self.start_real_time_calibration()

        except Exception as e:
            self.calibration_success_signal.emit(f"校准失败：{str(e)}")

    def calculate_delta_theta(self):
        """计算线激光校正角度"""
        f1 = self.processor.calibration_params['f1']
        f2 = self.processor.calibration_params['f2']
        omega = self.processor.env_factors['omega']
        theta = self.processor.env_factors['theta']
        r = self.processor.env_factors['r']*100
        h0 = self.processor.env_factors['h0']
        ti = self.processor.env_factors['ti']
        tj = self.processor.env_factors['tj']

        delta_theta = f1 * np.arctan(f2 * (ti - tj) * omega * r / h0) * 180/3.1415926 - theta
        return delta_theta

    def generate_diagnosis(self, analysis):
        """生成诊断建议"""
        diagnosis = []

        # 残差分析
        if analysis['residual_analysis']['norm_test'] < 0.05:
            diagnosis.append("残差分布异常（p=%.3f），建议检查环境干扰源" %
                             analysis['residual_analysis']['norm_test'])

        # 频域分析
        if abs(analysis['frequency_analysis']['dominant_freq_cal']) > 0.2:
            diagnosis.append("检测到高频噪声（%.2fHz），建议优化滤波参数" %
                             analysis['frequency_analysis']['dominant_freq_cal'])

        # 趋势稳定性
        if abs(analysis['trend_analysis']['trend_stability']) > 0.1:
            diagnosis.append("检测到显著线性趋势（斜率=%.3f），建议检查设备稳定性" %
                             analysis['trend_analysis']['trend_stability'])

        return diagnosis if diagnosis else ["系统状态正常"]

    def toggle_real_time_mode(self, state):
        """切换实时模式（改进版本）"""
        self.real_time_mode = state == QtCore.Qt.Checked
        if self.real_time_mode:
            self.start_btn.setEnabled(False)
            self.export_btn.setEnabled(False)
            self.start_real_time_calibration()
        else:
            self.stop_real_time()
            self.start_btn.setEnabled(True)
            self.export_btn.setEnabled(True)

    def stop_real_time(self):
        """停止实时模式（线程安全版本）"""
        self.stop_event.set()
        if self.real_time_thread and self.real_time_thread.is_alive():
            self.real_time_thread.join(timeout=2)
        self.real_time_thread = None
        self.statusBar().showMessage("实时模式已停止")
        self.real_time_check.setChecked(False)  # 取消复选框勾选

    def start_real_time_calibration(self):
        """启动实时数据处理（线程安全版本）"""
        if self.real_time_thread and self.real_time_thread.is_alive():
            return

        self.stop_event.clear()
        self.real_time_thread = threading.Thread(target=self.real_time_data_processing, daemon=True)
        self.real_time_thread.start()

    def real_time_data_processing(self):
        """实时数据处理逻辑（线程安全版本）"""
        while not self.stop_event.is_set():
            try:
                start_time = time.time()

                # 获取频率参数（主线程安全访问）
                freq = self.freq_slider.findChild(QtWidgets.QSlider).value() / 100
                interval = 1.0 / freq
                # 获取数据
                raw_data = self.processor.get_raw_data()

                # 数据处理
                filtered = self.processor.adaptive_filter(raw_data)
                calibrated = self.processor.dynamic_calibration(filtered)

                # 分析结果
                analysis = self.processor.analyze_calibration(raw_data, filtered, calibrated)

                # 使用信号发送数据到主线程
                self.update_gui_signal.emit(raw_data, filtered, calibrated, analysis)

                # 精确控制间隔
                elapsed = time.time() - start_time
                sleep_time = max(0, interval - elapsed)
                time.sleep(sleep_time)
            except Exception as e:
                print(f"Thread error: {str(e)}")
                break

    @QtCore.pyqtSlot(np.ndarray, np.ndarray, np.ndarray, dict)
    def handle_realtime_update(self, raw, filtered, calibrated, analysis):
        """处理实时更新（主线程执行）"""
        # 更新图表
        self.plot_widget.clear()
        # 计算线激光校正角度
        # 在获取数据前更新环境参数
        self.processor.env_factors['temp'] = np.random.normal(25, 0.5)  # 模拟温度变化
        self.processor.env_factors['humidity'] = np.random.normal(50, 2)  # 模拟湿度变化
        self.processor.env_factors['ti'] = np.random.normal(0.002, 0.0001)  # 模拟温度变化
        self.processor.env_factors['tj'] = np.random.normal(0.002, 0.0001)  # 模拟湿度变化

        # 更新界面显示
        self.temp_label.setText(f"温度：{self.processor.env_factors['temp']:.1f} ℃")
        self.humidity_label.setText(f"湿度：{self.processor.env_factors['humidity']:.1f} %")
        self.omega_label.setText(f"旋转激光发射器旋转角频率 (ω): {self.processor.env_factors['omega']:.2f} rad/s")
        self.theta_label.setText(f"预设标准夹角 (θ): {self.processor.env_factors['theta']:.2f}°")
        self.r_label.setText(f"横向距离 (r):  {self.processor.env_factors['r']:.2f} m")
        self.h0_label.setText(f"竖向间距 (h0):  {self.processor.env_factors['h0']:.2f} cm")
        self.ti_label.setText(f"上方光敏元件响应时间 (ti):  {self.processor.env_factors['ti']:.4f} s")
        self.tj_label.setText(f"下方光敏元件响应时间 (tj): {self.processor.env_factors['tj']:.4f} s")
        # self.result_label.setText(f"线激光校正角度 (Δθ): {self.processor.env_factors['omega']:.1f}°")
        # 计算线激光校正角度
        delta_theta = self.calculate_delta_theta()

        # 更新计算结果
        self.result_label.setText(f"线激光校正角度 (Δθ): {delta_theta:.4f}°")

        # 添加图例
        self.plot_widget.addLegend(offset=(0, 5))  # 直接添加图例并设置偏移

        self.plot_widget.plot(raw, pen='r', name="原始信号")  # 绘制原始数据
        self.plot_widget.plot(filtered, pen='g', name="滤波信号")  # 绘制滤波数据
        self.plot_widget.plot(calibrated, pen='b', name="校准信号")  # 绘制校准数据

        # 更新3D视图
        self.update_3d_view(calibrated)

        # 更新诊断信息
        diagnosis = self.generate_diagnosis(analysis)
        self.diagnosis_text.setHtml(
            f"<h3>实时校准分析</h3>"
            f"<p>数据来源：{self.processor.data_source}</p>"
            f"<p>标准偏差：{analysis['calibrated_stats']['std']:.4f}</p>"
            f"<p>信噪比：{analysis['calibrated_stats']['snr']:.2f} dB</p>"
            f"<h4>诊断建议：</h4><ul>{''.join([f'<li>{d}</li>' for d in diagnosis])}</ul>"
        )

        # 更新状态
        std = analysis['calibrated_stats']['std']
        status = f"实时模式运行中 | 当前偏差：{std:.3f}" + (" ✓" if std < 0.5 else " ⚠")
        self.statusBar().showMessage(status)
        self.calibration_success_signal.emit("实时校准成功！系统状态正常")

    def create_group(self, title, widgets):
        """创建带标题的分组框"""
        group = QtWidgets.QGroupBox(title)
        group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #5C5C5C;
                border-radius: 5px;
                margin-top: 10px;
                color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
            }
        """)
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(10, 15, 10, 10)
        for widget in widgets:
            layout.addWidget(widget)
        group.setLayout(layout)
        return group

    def export_report(self):
        """导出报告为PDF文件（包含图表和说明）到 'reports' 目录"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"激光校准报告_{timestamp}.pdf"
            filepath = os.path.join(self.report_dir, filename)

            # 获取当前数据（使用虚拟设备数据）
            raw_data = self.processor.get_raw_data()
            filtered = self.processor.adaptive_filter(raw_data)
            calibrated = self.processor.dynamic_calibration(filtered)

            # 执行分析获取诊断信息
            analysis = self.processor.analyze_calibration(raw_data, filtered, calibrated)
            diagnosis = self.generate_diagnosis(analysis)

            # 生成图表
            plt.figure(figsize=(11, 8.5))
            ax = plt.subplot(111)
            ax.plot(raw_data, label="原始信号", color='r')
            ax.plot(filtered, label="滤波信号", color='g')
            ax.plot(calibrated, label="校准信号", color='b')

            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
            plt.rcParams['axes.unicode_minus'] = False

            ax.legend()
            ax.set_title("激光校准数据展示")
            ax.set_xlabel("时间（索引）")
            ax.set_ylabel("信号强度")

            # 创建临时图表文件
            fd, temp_img_path = tempfile.mkstemp(suffix='.png')
            os.close(fd)
            plt.savefig(temp_img_path)
            plt.close()

            # 生成PDF报告
            c = canvas.Canvas(filepath, pagesize=letter)
            width, height = letter
            y_position = height - 40  # 初始Y坐标

            # 注册中文字体
            try:
                pdfmetrics.registerFont(TTFont('SimHei', 'simhei.ttf'))
            except:
                print("字体注册失败，使用默认字体")

            # 标题部分
            c.setFont("SimHei", 16)
            c.drawString(50, y_position, "激光校准系统报告")
            y_position -= 30

            # 基本信息
            c.setFont("SimHei", 12)
            info_lines = [
                f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"数据来源: {self.processor.data_source}",
                f"环境温度: {self.processor.env_factors['temp']:.1f} ℃",
                f"环境湿度: {self.processor.env_factors['humidity']:.1f} %",
                f"环境系数 F1: {self.processor.calibration_params['f1']:.2f}",
                f"(F1为环境系数，根据环境中的温度和湿度参数来调节，温度越高，取值越小，湿度越高，取值越大)",
                f"校准系数 F2: {self.processor.calibration_params['f2']:.2f}",
                f"(F2为测时精度修正系数，根据光敏元件2的时间响应误差来调节，误差越大，取值范围越小)"
            ]

            for line in info_lines:
                c.drawString(50, y_position, line)
                y_position -= 20

            # 添加间隔
            y_position -= 20

            # 数据分析结果
            stats_lines = [
                "数据分析结果:",
                f"- 原始信号均值: {analysis['raw_stats']['mean']:.2f}",
                f"- 校准后标准差: {analysis['calibrated_stats']['std']:.2f}",
                f"- 动态范围改进: {analysis['improvement_ratio']['dynamic_range'] * 100:.1f}%",
                f"- 信噪比 (SNR): {analysis['calibrated_stats']['snr']:.1f} dB",
                f"- 均方误差 (MSE): {np.mean((raw_data - calibrated) ** 2):.4f}"
            ]

            for line in stats_lines:
                if line.startswith("-"):
                    c.drawString(70, y_position, line)
                else:
                    c.drawString(50, y_position, line)
                y_position -= 20
            c.setFont("SimHei", 9)
            # 添加参数说明
            discribe_line = [
                "（参数说明：",
                "- 原始信号均值: 表示未经处理的原始信号平均值，反映信号基准电平或中心位置（单位与输入信号一致）",
                "- 校准后标准差: 校准后信号数据的离散程度指标，值越小说明信号稳定性越高（单位与输入信号一致）",
                "- 动态范围改进: 显示校准后信号最大/最小幅值比率的提升幅度，百分比越高表示系统处理能力提升越显著",
                "- 信噪比 (SNR): 有效信号功率与噪声功率的比值，",
                "数值越大说明信号质量越好（dB为对数单位，每增加10dB表示信噪比提高10倍）",
                "- 均方误差 (MSE): 原始信号与校准信号差异的平方期望值，",
                "数值越小说明校准精度越高（无量纲指标，适用于系统间的横向对比）"]
            for line in discribe_line:
                if line.startswith("-"):
                    c.drawString(70, y_position, line)
                else:
                    c.drawString(50, y_position, line)
                y_position -= 20
            y_position -= 25
            # 诊断信息
            c.setFont("SimHei", 14)
            c.drawString(50, y_position, "诊断建议:")
            y_position -= 25
            c.setFont("SimHei", 12)

            for i, advice in enumerate(diagnosis, 1):
                text = f"{i}. {advice}"
                # 自动换行处理
                while len(text) > 0:
                    c.drawString(70, y_position, text[:80])  # 每行最多80字符
                    text = text[80:]
                    y_position -= 20
                y_position -= 5  # 条目间距

            # 添加图表（自动换页）
            c.showPage()
            c.setFont("SimHei", 14)
            c.drawString(50, height - 40, "信号对比图表")
            c.drawImage(temp_img_path, 50, height - 500, width=500, height=400)
            c.setFont("Simhei", 12)
            c.drawString(50, height - 650, "数据说明：")
            c.drawString(70, height - 670, "- 原始信号：直接从激光传感器采集到的数据，未经任何处理。")
            c.drawString(70, height - 690, "- 滤波信号：经过滤波处理后的数据，用于去除信号中的噪声。")
            c.drawString(70, height - 710, "- 校准信号：经过动态校准后的最终数据，用于分析激光系统的性能。")
            # 保存PDF
            c.save()

            # 清理临时文件
            try:
                os.remove(temp_img_path)
            except Exception as e:
                print(f"无法删除临时文件: {str(e)}")

            QtWidgets.QMessageBox.information(self, "提示", f"报告已生成成功并保存到：\n{filepath}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "错误", f"报告生成失败！\n错误信息：{str(e)}")

    def update_status(self, message):
        """更新状态显示"""
        if "实时" not in message:
            self.statusBar().showMessage(message)
        color = "#4CAF50" if "成功" in message else "#F44336"
        self.status_indicator.setStyleSheet(f"""
            background-color: {color};
            border-radius: 8px;
            min-height: 16px;
        """)
        QtCore.QTimer.singleShot(3000, lambda: self.status_indicator.setStyleSheet(""))

    def init_styles(self):
        """初始化样式表"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2E2E2E;
            }
            QDockWidget {
                background: #3C3C3C;
                titlebar-close-icon: url(close.png);
                titlebar-normal-icon: url(undock.png);
            }
            QDockWidget::title {
                background: #3C3C3C;
                padding: 6px;
                color: white;
            }
            QTextEdit {
                background-color: #404040;
                color: #FFFFFF;
                border: 1px solid #505050;
                border-radius: 4px;
                padding: 8px;
            QGroupBox {
                    border: 1px solid #5C5C5C;
                    border-radius: 5px;
                    margin-top: 10px;
                    color: white;
                }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
            }
            QLabel {
                color: #FFFFFF;
                font: 12px 'Simhei';
            }
        """)


# ---------------------
# 主程序入口
# ---------------------
if __name__ == "__main__":
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QtWidgets.QApplication([])
    app.setFont(QtGui.QFont("Segoe UI", 10))

    window = EnhancedLaserGUI()
    window.setWindowTitle("激光校准系统专业版 v1.0")
    window.resize(1360, 800)
    window.show()

    app.exec_()
