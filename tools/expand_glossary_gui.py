import os
import sys
from pathlib import Path

from core.engine.glossary_expander import expand_glossary_csv, load_spacy_model


def default_output_path(input_file):
    path = Path(input_file)
    return str(path.with_name(f"{path.stem}_expanded.csv"))


def run_expansion(input_file, output_file=None, src_lang="en", model_name="en_core_web_sm", model_path=None, use_spacy=True):
    output = output_file or default_output_path(input_file)
    nlp = load_spacy_model(model_name=model_name, model_path=model_path) if use_spacy else None
    return expand_glossary_csv(input_file, output, src_lang=src_lang, nlp=nlp)


APP_STYLESHEET = """
QWidget {
    background-color: #FDF6E3;
    color: #333333;
    font-family: "Segoe UI", "Microsoft YaHei", "Arial", sans-serif;
}
#HeaderLabel {
    font-size: 26px;
    font-weight: bold;
    color: #B85C00;
    padding: 12px 0;
}
QLabel {
    font-size: 14px;
}
QLineEdit, QComboBox, QPlainTextEdit {
    border: 1px solid #DCDCDC;
    border-radius: 6px;
    padding: 8px 10px;
    background-color: #FFFFFF;
    font-size: 13px;
}
QPushButton {
    background-color: #FFFFFF;
    color: #333333;
    border: 1px solid #DCDCDC;
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 14px;
}
QPushButton:hover {
    border-color: #B85C00;
}
#RunButton {
    background-color: #B85C00;
    color: #FFFFFF;
    font-size: 17px;
    font-weight: bold;
    border: 1px solid #8F4500;
    border-radius: 8px;
    padding: 12px 20px;
}
#RunButton:disabled {
    background-color: #BDC3C7;
    color: #6F7A7D;
}
QProgressBar {
    border: 1px solid #BDC3C7;
    border-radius: 6px;
    text-align: center;
    height: 20px;
}
QProgressBar::chunk {
    background-color: #2ECC71;
    border-radius: 5px;
}
"""


def launch_gui():
    from PyQt6.QtCore import Qt, QThread, pyqtSignal
    from PyQt6.QtWidgets import (
        QApplication,
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QFileDialog,
        QPlainTextEdit,
        QComboBox,
        QMessageBox,
        QProgressBar,
        QLineEdit,
        QCheckBox,
    )

    class DropLabel(QLabel):
        filePathChanged = pyqtSignal(str)

        def __init__(self):
            super().__init__()
            self.setAcceptDrops(True)
            self.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setText("拖放 CSV / XLSX 文件于此\n或点击选择文件")
            self.setMinimumHeight(110)
            self.setStyleSheet(
                "QLabel { border: 2px dashed #BDC3C7; border-radius: 8px; "
                "background-color: #FFFFFF; color: #7F8C8D; padding: 18px; }"
            )

        def dragEnterEvent(self, event):
            if event.mimeData().hasUrls() and len(event.mimeData().urls()) == 1:
                file_path = event.mimeData().urls()[0].toLocalFile()
                if file_path.lower().endswith((".csv", ".xlsx", ".xls")):
                    event.acceptProposedAction()

        def dropEvent(self, event):
            self.filePathChanged.emit(event.mimeData().urls()[0].toLocalFile())

        def mousePressEvent(self, event):
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "选择术语表文件",
                "",
                "Glossary Files (*.csv *.xlsx *.xls);;CSV Files (*.csv);;Excel Files (*.xlsx *.xls)",
            )
            if file_path:
                self.filePathChanged.emit(file_path)

        def set_file_path(self, path):
            self.setText(f"已选择文件:\n{os.path.basename(path)}\n\n可再次拖放或点击更换")

    class Worker(QThread):
        finished = pyqtSignal(int)
        error = pyqtSignal(str)

        def __init__(self, input_file, output_file, src_lang, model_name, model_path, use_spacy):
            super().__init__()
            self.input_file = input_file
            self.output_file = output_file
            self.src_lang = src_lang
            self.model_name = model_name
            self.model_path = model_path
            self.use_spacy = use_spacy

        def run(self):
            try:
                count = run_expansion(
                    self.input_file,
                    self.output_file,
                    src_lang=self.src_lang,
                    model_name=self.model_name,
                    model_path=self.model_path or None,
                    use_spacy=self.use_spacy,
                )
                self.finished.emit(count)
            except Exception as exc:
                self.error.emit(str(exc))

    class App(QWidget):
        def __init__(self):
            super().__init__()
            self.input_file = ""
            self.output_file = ""
            self.worker = None
            self.init_ui()

        def init_ui(self):
            self.setWindowTitle("术语匹配扩展工具")
            self.setMinimumSize(760, 620)

            layout = QVBoxLayout(self)
            layout.setContentsMargins(24, 18, 24, 18)
            layout.setSpacing(14)

            header = QLabel("术语匹配扩展工具")
            header.setObjectName("HeaderLabel")
            header.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(header)

            layout.addWidget(QLabel("第一步：选择术语表文件"))
            self.drop_label = DropLabel()
            self.drop_label.filePathChanged.connect(self.on_input_selected)
            layout.addWidget(self.drop_label)

            layout.addWidget(QLabel("第二步：选择输出文件"))
            output_row = QHBoxLayout()
            self.output_display = QLineEdit()
            self.output_display.setReadOnly(True)
            output_btn = QPushButton("浏览...")
            output_btn.clicked.connect(self.choose_output_file)
            output_row.addWidget(self.output_display)
            output_row.addWidget(output_btn)
            layout.addLayout(output_row)

            config_row = QHBoxLayout()
            self.src_lang = QLineEdit("en")
            self.src_lang.setToolTip("需要扩展的源语言列名")
            self.model_combo = QComboBox()
            self.model_combo.addItems(["en_core_web_sm", "en_core_web_md", "en_core_web_lg", "en_core_web_trf"])
            config_row.addWidget(QLabel("源列"))
            config_row.addWidget(self.src_lang)
            config_row.addWidget(QLabel("spaCy 模型"))
            config_row.addWidget(self.model_combo)
            layout.addLayout(config_row)

            model_path_row = QHBoxLayout()
            self.use_spacy = QCheckBox("使用 spaCy 词形分析")
            self.use_spacy.setChecked(True)
            self.model_path = QLineEdit()
            self.model_path.setPlaceholderText("可选：本地 spaCy 模型目录，用于离线 exe 打包")
            model_path_btn = QPushButton("模型目录...")
            model_path_btn.clicked.connect(self.choose_model_path)
            model_path_row.addWidget(self.use_spacy)
            model_path_row.addWidget(self.model_path)
            model_path_row.addWidget(model_path_btn)
            layout.addLayout(model_path_row)

            self.start_btn = QPushButton("开始扩展")
            self.start_btn.setObjectName("RunButton")
            self.start_btn.clicked.connect(self.start)
            layout.addWidget(self.start_btn, alignment=Qt.AlignmentFlag.AlignCenter)

            self.progress = QProgressBar()
            self.progress.setVisible(False)
            layout.addWidget(self.progress)

            layout.addWidget(QLabel("日志"))
            self.log_output = QPlainTextEdit()
            self.log_output.setReadOnly(True)
            layout.addWidget(self.log_output, stretch=1)

        def log(self, message):
            self.log_output.appendPlainText(message)

        def on_input_selected(self, file_path):
            if file_path.lower().endswith((".xlsx", ".xls")):
                QMessageBox.warning(self, "暂不支持", "当前离线扩展工具入口支持 CSV。请先将 Excel 另存为 CSV。")
                return
            self.input_file = file_path
            self.output_file = default_output_path(file_path)
            self.drop_label.set_file_path(file_path)
            self.output_display.setText(self.output_file)
            self.log(f"已选择输入文件: {file_path}")
            self.log(f"输出文件: {self.output_file}")

        def choose_output_file(self):
            if not self.input_file:
                QMessageBox.warning(self, "提示", "请先选择输入文件。")
                return
            file_path, _ = QFileDialog.getSaveFileName(self, "选择输出 CSV", self.output_file, "CSV Files (*.csv)")
            if file_path:
                self.output_file = file_path
                self.output_display.setText(file_path)

        def choose_model_path(self):
            directory = QFileDialog.getExistingDirectory(self, "选择本地 spaCy 模型目录", "")
            if directory:
                self.model_path.setText(directory)

        def start(self):
            if not self.input_file:
                QMessageBox.warning(self, "提示", "请先选择输入文件。")
                return
            self.start_btn.setEnabled(False)
            self.progress.setVisible(True)
            self.progress.setRange(0, 0)
            self.log("--- 任务开始 ---")

            self.worker = Worker(
                self.input_file,
                self.output_file,
                self.src_lang.text().strip() or "en",
                self.model_combo.currentText(),
                self.model_path.text().strip(),
                self.use_spacy.isChecked(),
            )
            self.worker.finished.connect(self.on_finished)
            self.worker.error.connect(self.on_error)
            self.worker.start()

        def on_finished(self, count):
            self.progress.setRange(0, 100)
            self.progress.setValue(100)
            self.start_btn.setEnabled(True)
            self.log(f"任务完成：已生成 {count} 条记录。")
            self.log(f"保存至: {self.output_file}")
            self.log("--- 任务结束 ---")
            QMessageBox.information(self, "完成", f"已生成 {count} 条记录。")

        def on_error(self, message):
            self.progress.setVisible(False)
            self.start_btn.setEnabled(True)
            self.log(f"错误: {message}")
            self.log("--- 任务中断 ---")
            QMessageBox.critical(self, "错误", message)

    app = QApplication(sys.argv)
    app.setStyleSheet(APP_STYLESHEET)
    window = App()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(launch_gui())
