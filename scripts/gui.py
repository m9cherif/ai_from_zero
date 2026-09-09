"""PyQt6 desktop interface for talking to the model.

    python scripts/gui.py
    python scripts/gui.py --checkpoint output/checkpoints/checkpoint_latest.pt

Generation runs in a worker thread and streams token by token, so the window
stays responsive on CPU where the model produces only tens of tokens a second.
"""

import argparse
import glob
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QFont, QTextCursor, QTextCharFormat, QColor, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
    QLineEdit, QPushButton, QLabel, QSpinBox, QDoubleSpinBox, QGroupBox,
    QFormLayout, QStatusBar, QSplitter, QCheckBox, QMessageBox,
)

# --- palette ---------------------------------------------------------------
BG = "#12141a"
PANEL = "#1a1d26"
BORDER = "#2a2f3d"
TEXT = "#e4e6eb"
MUTED = "#8b93a7"
USER = "#7aa2f7"
AI = "#9ece6a"
ACCENT = "#bb9af7"


def find_latest_checkpoint(directory: str = "output/checkpoints") -> str:
    """Prefer checkpoint_latest.pt, else the highest-numbered step file."""
    latest = os.path.join(directory, "checkpoint_latest.pt")
    if os.path.exists(latest):
        return latest

    def step_of(path: str) -> int:
        match = re.search(r"checkpoint_step_(\d+)\.pt$", path.replace("\\", "/"))
        return int(match.group(1)) if match else -1

    candidates = sorted(glob.glob(os.path.join(directory, "checkpoint_step_*.pt")), key=step_of)
    return candidates[-1] if candidates else ""


class ModelLoader(QThread):
    """Loads the checkpoint off the UI thread so the window paints immediately."""

    loaded = pyqtSignal(object, str)
    failed = pyqtSignal(str)

    def __init__(self, checkpoint: str, tokenizer: str, device: str):
        super().__init__()
        self._checkpoint = checkpoint
        self._tokenizer = tokenizer
        self._device = device

    def run(self) -> None:
        try:
            from myai.inference import InferenceEngine

            engine = InferenceEngine(device=self._device)
            engine.load_checkpoint(self._checkpoint, self._tokenizer)
            config = engine.model.config
            summary = (
                f"{engine.model.num_parameters():,} params · "
                f"{config.n_layers} layers · d_model {config.d_model} · "
                f"{config.n_heads} heads ({config.n_kv_heads or config.n_heads} kv) · "
                f"{config.position_encoding} · vocab {config.vocab_size} · "
                f"ctx {config.max_seq_len}"
            )
            self.loaded.emit(engine, summary)
        except Exception as exc:  # surfaced in the UI rather than the console
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class GenerationWorker(QThread):
    """Streams generated text, emitting each decoded chunk as it arrives."""

    chunk = pyqtSignal(str)
    done = pyqtSignal(int, float)
    failed = pyqtSignal(str)

    def __init__(self, engine, prompt: str, params: dict):
        super().__init__()
        self._engine = engine
        self._prompt = prompt
        self._params = params
        self._stop = False

    def stop(self) -> None:
        """Ask the loop to end after the current token."""
        self._stop = True

    def run(self) -> None:
        start = time.perf_counter()
        chars = 0
        try:
            for piece in self._engine.generate_stream(prompt=self._prompt, **self._params):
                if self._stop:
                    break
                chars += len(piece)
                self.chunk.emit(piece)
            self.done.emit(chars, time.perf_counter() - start)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class ChatWindow(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self._args = args
        self._engine = None
        self._worker = None

        self.setWindowTitle("myai — a language model built from scratch")
        self.resize(1040, 720)
        self._build_ui()
        self._apply_style()
        self._start_loading()

    # --- construction ----------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        outer = QHBoxLayout(central)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(12)

        # Left: transcript + composer
        left = QVBoxLayout()
        left.setSpacing(10)

        self.banner = QLabel(
            "This is a <b>base language model</b>: it continues your text, it does not "
            "answer questions. Give it the start of a sentence."
        )
        self.banner.setWordWrap(True)
        self.banner.setObjectName("banner")

        self.transcript = QTextEdit()
        self.transcript.setReadOnly(True)
        self.transcript.setFont(QFont("Cascadia Mono, Consolas, monospace", 11))

        composer = QHBoxLayout()
        composer.setSpacing(8)
        self.input = QLineEdit()
        self.input.setPlaceholderText("To be, or not to be…")
        self.input.returnPressed.connect(self._on_send)
        self.send_button = QPushButton("Send")
        self.send_button.setObjectName("send")
        self.send_button.clicked.connect(self._on_send)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("stop")
        self.stop_button.clicked.connect(self._on_stop)
        self.stop_button.setEnabled(False)
        composer.addWidget(self.input, 1)
        composer.addWidget(self.send_button)
        composer.addWidget(self.stop_button)

        left.addWidget(self.banner)
        left.addWidget(self.transcript, 1)
        left.addLayout(composer)

        left_widget = QWidget()
        left_widget.setLayout(left)

        # Right: sampling controls
        panel = QGroupBox("Sampling")
        form = QFormLayout(panel)
        form.setSpacing(9)

        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(0.0, 2.0)
        self.temperature.setSingleStep(0.05)
        self.temperature.setValue(0.7)
        self.temperature.setToolTip("0 = greedy/deterministic. Higher = more random.")

        self.top_k = QSpinBox()
        self.top_k.setRange(0, 2048)
        self.top_k.setValue(40)
        self.top_k.setToolTip("Keep only the k highest-scoring tokens. 0 disables.")

        self.top_p = QDoubleSpinBox()
        self.top_p.setRange(0.0, 1.0)
        self.top_p.setSingleStep(0.05)
        self.top_p.setValue(0.95)
        self.top_p.setToolTip("Nucleus sampling. 1.0 disables.")

        self.min_p = QDoubleSpinBox()
        self.min_p.setRange(0.0, 1.0)
        self.min_p.setSingleStep(0.01)
        self.min_p.setValue(0.05)
        self.min_p.setToolTip("Keep tokens with probability >= min_p x p_max. 0 disables.")

        self.repetition = QDoubleSpinBox()
        self.repetition.setRange(1.0, 2.0)
        self.repetition.setSingleStep(0.05)
        self.repetition.setValue(1.15)
        self.repetition.setToolTip("Above 1.0 discourages repeating tokens.")

        self.max_tokens = QSpinBox()
        self.max_tokens.setRange(1, 512)
        self.max_tokens.setValue(60)

        form.addRow("Temperature", self.temperature)
        form.addRow("Top-k", self.top_k)
        form.addRow("Top-p", self.top_p)
        form.addRow("Min-p", self.min_p)
        form.addRow("Repetition", self.repetition)
        form.addRow("Max tokens", self.max_tokens)

        self.keep_context = QCheckBox("Feed transcript back as context")
        self.keep_context.setToolTip(
            "Continues from the whole conversation instead of just the latest prompt."
        )

        right = QVBoxLayout()
        right.setSpacing(10)
        right.addWidget(panel)
        right.addWidget(self.keep_context)

        self.model_label = QLabel("Loading model…")
        self.model_label.setWordWrap(True)
        self.model_label.setObjectName("model")
        right.addWidget(self.model_label)

        clear = QPushButton("Clear transcript")
        clear.clicked.connect(self._on_clear)
        right.addWidget(clear)
        right.addStretch(1)

        right_widget = QWidget()
        right_widget.setLayout(right)
        right_widget.setFixedWidth(280)

        outer.addWidget(left_widget, 1)
        outer.addWidget(right_widget)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Loading model…")

        QShortcut(QKeySequence("Ctrl+L"), self, self._on_clear)
        QShortcut(QKeySequence("Esc"), self, self._on_stop)

    def _apply_style(self) -> None:
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{ background: {BG}; color: {TEXT}; }}
            QTextEdit {{
                background: {PANEL}; border: 1px solid {BORDER};
                border-radius: 10px; padding: 12px; selection-background-color: {USER};
            }}
            QLineEdit {{
                background: {PANEL}; border: 1px solid {BORDER}; border-radius: 8px;
                padding: 10px 12px; font-size: 14px;
            }}
            QLineEdit:focus {{ border: 1px solid {USER}; }}
            QPushButton {{
                background: {PANEL}; border: 1px solid {BORDER}; border-radius: 8px;
                padding: 10px 18px; font-weight: 600;
            }}
            QPushButton:hover {{ border: 1px solid {MUTED}; }}
            QPushButton:disabled {{ color: {MUTED}; }}
            QPushButton#send {{ background: {USER}; color: #10131a; border: none; }}
            QPushButton#stop {{ background: #f7768e; color: #10131a; border: none; }}
            QPushButton#stop:disabled {{ background: {PANEL}; color: {MUTED}; }}
            QGroupBox {{
                border: 1px solid {BORDER}; border-radius: 10px;
                margin-top: 10px; padding: 14px 10px 10px 10px; font-weight: 600;
            }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 12px; color: {ACCENT}; }}
            QLabel#banner {{ color: {MUTED}; padding: 2px 4px; }}
            QLabel#model {{ color: {MUTED}; font-size: 11px; padding: 6px 2px; }}
            QSpinBox, QDoubleSpinBox {{
                background: {BG}; border: 1px solid {BORDER};
                border-radius: 6px; padding: 5px;
            }}
            QStatusBar {{ color: {MUTED}; }}
        """)

    # --- model loading ---------------------------------------------------
    def _start_loading(self) -> None:
        checkpoint = self._args.checkpoint or find_latest_checkpoint()
        if not checkpoint or not os.path.exists(checkpoint):
            self._fail(
                "No checkpoint found.\n\nTrain one first:\n"
                "  python scripts/build_tokenizer.py --type bpe --vocab-size 2048\n"
                "  python scripts/train.py --d-model 320 --n-layers 6 --steps 6500"
            )
            return

        self._append_system(f"checkpoint: {checkpoint}")
        self._loader = ModelLoader(checkpoint, self._args.tokenizer, self._args.device)
        self._loader.loaded.connect(self._on_loaded)
        self._loader.failed.connect(self._fail)
        self._loader.start()

    def _on_loaded(self, engine, summary: str) -> None:
        self._engine = engine
        self.model_label.setText(summary.replace(" · ", "\n"))
        self.statusBar().showMessage("Ready")
        self._append_system(summary)
        self.input.setFocus()

    def _fail(self, message: str) -> None:
        self.statusBar().showMessage("Failed to load model")
        self._append_system(message)
        QMessageBox.critical(self, "Could not load model", message)

    # --- transcript ------------------------------------------------------
    def _append_block(self, label: str, color: str, text: str = "") -> None:
        cursor = self.transcript.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        header = QTextCharFormat()
        header.setForeground(QColor(color))
        header.setFontWeight(QFont.Weight.Bold)

        body = QTextCharFormat()
        body.setForeground(QColor(TEXT))

        if not self.transcript.document().isEmpty():
            cursor.insertText("\n\n", body)
        cursor.insertText(f"{label}\n", header)
        if text:
            cursor.insertText(text, body)

        self.transcript.setTextCursor(cursor)
        self.transcript.ensureCursorVisible()

    def _append_system(self, text: str) -> None:
        self._append_block("system", MUTED, text)

    def _stream(self, piece: str) -> None:
        cursor = self.transcript.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(TEXT))
        cursor.insertText(piece, fmt)
        self.transcript.setTextCursor(cursor)
        self.transcript.ensureCursorVisible()

    def _on_clear(self) -> None:
        self.transcript.clear()

    # --- generation ------------------------------------------------------
    def _on_send(self) -> None:
        prompt = self.input.text().strip()
        if not prompt or self._engine is None or self._worker is not None:
            return

        self._append_block("you", USER, prompt)
        self.input.clear()

        if self.keep_context.isChecked():
            prompt = self.transcript.toPlainText()[-2000:]

        params = dict(
            max_new_tokens=self.max_tokens.value(),
            temperature=self.temperature.value(),
            top_k=self.top_k.value() or None,
            top_p=self.top_p.value() if self.top_p.value() < 1.0 else None,
            min_p=self.min_p.value() or None,
            repetition_penalty=self.repetition.value(),
        )

        self._append_block("myai", AI)
        self._set_busy(True)
        self.statusBar().showMessage("Generating…")

        self._worker = GenerationWorker(self._engine, prompt, params)
        self._worker.chunk.connect(self._stream)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_gen_failed)
        self._worker.start()

    def _on_stop(self) -> None:
        if self._worker is not None:
            self._worker.stop()
            self.statusBar().showMessage("Stopping…")

    def _on_done(self, chars: int, elapsed: float) -> None:
        rate = chars / elapsed if elapsed > 0 else 0.0
        self.statusBar().showMessage(f"{chars} chars in {elapsed:.1f}s ({rate:.0f} chars/s)")
        self._finish()

    def _on_gen_failed(self, message: str) -> None:
        self._append_system(message)
        self.statusBar().showMessage("Generation failed")
        self._finish()

    def _finish(self) -> None:
        self._worker = None
        self._set_busy(False)
        self.input.setFocus()

    def _set_busy(self, busy: bool) -> None:
        self.send_button.setEnabled(not busy)
        self.input.setEnabled(not busy)
        self.stop_button.setEnabled(busy)

    def closeEvent(self, event) -> None:
        if self._worker is not None:
            self._worker.stop()
            self._worker.wait(3000)
        event.accept()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None, help="Defaults to the newest in output/checkpoints")
    parser.add_argument("--tokenizer", default="output/tokenizer.json")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setApplicationName("myai")
    window = ChatWindow(args)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
