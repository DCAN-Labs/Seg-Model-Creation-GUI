import sys
import os
import subprocess
import psutil
from pathlib import Path

from main_window import Ui_MainWindow
from main_window_v2 import Ui_MainWindowV2
from login_window import Ui_LoginWindow
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QObject, QThread, pyqtSignal, QTimer, Qt
import PyQt5_stylesheets
from custom_widgets import *

# region ### CONSTANTS ###
PRESETS_DIR_V1 = "automation_presets"
PRESETS_DIR_V2 = "automation_presets_v2"
PRESET_EXTENSION = ".config"
GRAY_BACKGROUND = "background-color: rgb(137, 137, 137)"
# endregion


# region ### WORKER THREAD CLASSES ###

class PipelineWorkerThread(QtCore.QThread):
    """Thread for running the v1 training pipeline without blocking the GUI"""
    finished = pyqtSignal()

    def __init__(self, dcan_path, task_path, synth_path, raw_path, results_path, trained_path,
                 modality, task_num, distribution, synth_amt, script_dir, step_selections):
        QtCore.QThread.__init__(self)
        self.dcan_path = dcan_path
        self.task_path = task_path
        self.synth_path = synth_path
        self.raw_path = raw_path
        self.results_path = results_path
        self.trained_path = trained_path
        self.modality = modality
        self.task_num = task_num
        self.distribution = distribution
        self.synth_amt = synth_amt
        self.script_dir = script_dir
        self.step_selections = step_selections
        self.processes = []
        self.quit_program = False

    def cancel_jobs(self):
        """Uses active jobs file to cancel all job ids listed"""
        active_jobs_path = Path(self.script_dir) / "logs" / f"Task{self.task_num}" / "active_jobs.txt"
        if not active_jobs_path.exists():
            return
        with open(active_jobs_path, 'r') as f:
            job_lines = f.readlines()
        for line in job_lines:
            job_id = line.strip()
            if job_id:
                subprocess.run(["scancel", job_id])
        active_jobs_path.unlink()

    def run(self):
        """Start subprocess and wait for it to finish"""
        pipeline_script = Path(self.script_dir) / "trainer_pipeline.py"
        cmd = [
            "python", str(pipeline_script),
            self.dcan_path, self.task_path, self.synth_path,
            self.raw_path, self.results_path, self.trained_path,
            self.modality, self.task_num, self.distribution,
            self.synth_amt, self.step_selections
        ]
        process = subprocess.Popen(cmd, stdout=None, stderr=None)
        self.processes.append(process)
        process.wait()
        self.cancel_jobs()
        if process.returncode == 0:
            pass
        elif not self.quit_program:
            print("AN ERROR HAS OCCURRED")
        elif self.quit_program:
            print("PROCESS STOPPED")
        self.finished.emit()

    def stop_program(self):
        """Cancel subprocesses when user stops program manually"""
        if len(self.processes) > 0:
            self.quit_program = True
            print("Stopping Process...")
            parent = psutil.Process(self.processes[-1].pid)
            try:
                for child in parent.children(recursive=True):
                    child.kill()
            except:
                pass
            parent.kill()


class PipelineWorkerThreadV2(QtCore.QThread):
    """Thread for running the v2 training pipeline without blocking the GUI"""
    finished = pyqtSignal()

    def __init__(self, dcan_path, task_path, synth_path, raw_path, results_path, trained_path,
                 modality, task_num, distribution, synth_amt, dataset_name, model_type,
                 script_dir, step_selections):
        QtCore.QThread.__init__(self)
        self.dcan_path = dcan_path
        self.task_path = task_path
        self.synth_path = synth_path
        self.raw_path = raw_path
        self.results_path = results_path
        self.trained_path = trained_path
        self.modality = modality
        self.task_num = task_num
        self.distribution = distribution
        self.synth_amt = synth_amt
        self.dataset_name = dataset_name
        self.model_type = model_type
        self.script_dir = script_dir
        self.step_selections = step_selections
        self.processes = []
        self.quit_program = False

    def cancel_jobs(self):
        """Uses active jobs file to cancel all job ids listed"""
        dataset_folder = f"Dataset{self.task_num}_{self.dataset_name}"
        active_jobs_path = Path(self.script_dir) / "logs" / dataset_folder / "active_jobs.txt"
        if not active_jobs_path.exists():
            return
        with open(active_jobs_path, 'r') as f:
            job_lines = f.readlines()
        for line in job_lines:
            job_id = line.strip()
            if job_id:
                subprocess.run(["scancel", job_id])
        active_jobs_path.unlink()

    def run(self):
        """Start subprocess and wait for it to finish"""
        pipeline_script = Path(self.script_dir) / "trainer_pipeline_v2.py"
        cmd = [
            "python", str(pipeline_script),
            self.dcan_path, self.task_path, self.synth_path,
            self.raw_path, self.results_path, self.trained_path,
            self.modality, self.task_num, self.distribution,
            self.synth_amt, self.dataset_name, self.model_type,
            self.step_selections
        ]
        process = subprocess.Popen(cmd, stdout=None, stderr=None)
        self.processes.append(process)
        process.wait()
        self.cancel_jobs()
        if process.returncode == 0:
            pass
        elif not self.quit_program:
            print("AN ERROR HAS OCCURRED")
        elif self.quit_program:
            print("PROCESS STOPPED")
        self.finished.emit()

    def stop_program(self):
        """Cancel subprocesses when user stops program manually"""
        if len(self.processes) > 0:
            self.quit_program = True
            print("Stopping Process...")
            parent = psutil.Process(self.processes[-1].pid)
            try:
                for child in parent.children(recursive=True):
                    child.kill()
            except:
                pass
            parent.kill()

# endregion


# region ### MAIN WINDOW CLASS ###

class Window(QtWidgets.QMainWindow):
    """Main window — works for both v1 and v2 depending on the pipeline_version passed in."""

    def __init__(self, pipeline_version=1):
        super().__init__()
        self.pipeline_version = pipeline_version

        # Pick the right UI class and preset directory
        if pipeline_version == 2:
            self.ui = Ui_MainWindowV2()
            self.presets_dir_name = PRESETS_DIR_V2
        else:
            self.ui = Ui_MainWindow()
            self.presets_dir_name = PRESETS_DIR_V1

        self.ui.setupUi(self)

        self.script_dir = Path(__file__).resolve().parent
        os.chdir(self.script_dir)

        self.worker_thread = None
        self.is_running = False
        self.step_selections = []

        # Core input fields shared by v1 and v2
        self.input_fields = {
            'dcan_path': self.ui.line_dcan_path,
            'synth_path': self.ui.line_synth_path,
            'task_path': self.ui.line_task_path,
            'raw_data_base_path': self.ui.line_raw_data_base_path,
            'modality': self.ui.line_modality,
            'task_number': self.ui.line_task_number,
            'distribution': self.ui.line_distribution,
            'synth_img_amt': self.ui.line_synth_img_amt,
            'results_path': self.ui.line_results_path,
            'trained_models_path': self.ui.line_trained_models_path,
        }

        # V2-only extra fields
        if pipeline_version == 2:
            self.input_fields['dataset_name'] = self.ui.line_dataset_name
            self.input_fields['model_type'] = self.ui.line_model_type

        self._initialize_preset_comboboxes()

        # Wire up buttons (attributes exist on both UI classes)
        self.ui.pushButton.setText('Run')
        self.ui.pushButton.clicked.connect(self.run_program)
        self.ui.pushButton_2.setText('Populate Preset')
        self.ui.pushButton_2.clicked.connect(self.populate_inputs)
        self.ui.button_clear.clicked.connect(self.clear_inputs)
        self.ui.button_save.clicked.connect(self.save_preset)
        self.ui.button_remove.clicked.connect(self.remove_preset)
        self.ui.button_select_all.clicked.connect(self.toggle_all_checkboxes)
        self.ui.button_browse_1.clicked.connect(lambda: self.browse_path('dcan_path', str(Path.home())))
        self.ui.button_browse_2.clicked.connect(lambda: self.browse_path('synth_path', str(Path.home())))
        self.ui.button_browse_3.clicked.connect(lambda: self.browse_path('task_path', "/"))
        self.ui.button_browse_4.clicked.connect(lambda: self.browse_path('raw_data_base_path', "/"))

    # ------------------------------------------------------------------
    # Preset helpers
    # ------------------------------------------------------------------

    def _initialize_preset_comboboxes(self):
        """Load and populate preset comboboxes from the version-appropriate folder"""
        presets_dir = self.script_dir / self.presets_dir_name

        for file in (presets_dir.iterdir() if presets_dir.exists() else []):
            if file.suffix == PRESET_EXTENSION:
                name = file.stem
                self.ui.comboBox_preset.insertItem(
                    self._find_alphabetical_index(self.ui.comboBox_preset, name), name)
                self.ui.comboBox_remove_preset.insertItem(
                    self._find_alphabetical_index(self.ui.comboBox_remove_preset, name), name)

        if self.ui.comboBox_preset.count() < 1:
            self._setup_empty_combobox(self.ui.comboBox_preset, '-- No Presets --')
            self._setup_empty_combobox(self.ui.comboBox_remove_preset, '-- No Presets --')
        else:
            self._setup_searchable_combobox(self.ui.comboBox_preset, '-- Select Preset --')
            self._setup_searchable_combobox(self.ui.comboBox_remove_preset, '-- Select Preset --')

    def _find_alphabetical_index(self, combo_box, item):
        items = [combo_box.itemText(i) for i in range(combo_box.count())]
        items.append(item)
        items.sort(key=str.upper)
        return items.index(item)

    def _setup_searchable_combobox(self, combo_box, placeholder):
        combo_box.setEditable(True)
        combo_box.lineEdit().setPlaceholderText(placeholder)
        combo_box.completer().setCompletionMode(QtWidgets.QCompleter.PopupCompletion)
        combo_box.setInsertPolicy(QComboBox.NoInsert)
        combo_box.setCurrentIndex(-1)

    def _setup_empty_combobox(self, combo_box, placeholder):
        combo_box.setEditable(False)
        combo_box.setPlaceholderText(placeholder)
        combo_box.setStyleSheet(GRAY_BACKGROUND)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_inputs(self):
        """Validate all inputs; v2 adds dataset_name and model_type checks"""
        path_fields = ['dcan_path', 'synth_path', 'task_path', 'raw_data_base_path',
                       'results_path', 'trained_models_path']
        paths_valid = all(
            Path(self.input_fields[f].text().strip()).exists() for f in path_fields
        )

        modality = self.input_fields['modality'].text().strip().lower()
        modality_valid = modality in ["t1", "t2", "t1t2"]

        task_number_valid = self.input_fields['task_number'].text().isdigit()

        distribution = self.input_fields['distribution'].text().strip().lower()
        distribution_valid = distribution in ["uniform", "normal"]

        synth_amt_valid = self.input_fields['synth_img_amt'].text().strip().isdigit()

        # Task-path / task-number consistency differs between v1 and v2
        tasks_match = True
        if task_number_valid and Path(self.input_fields['task_path'].text().strip()).exists():
            task_path = Path(self.input_fields['task_path'].text().strip())
            task_num = self.input_fields['task_number'].text().strip()
            if self.pipeline_version == 2:
                dataset_name = self.input_fields['dataset_name'].text().strip()
                # v2 folder is Dataset###_NAME
                tasks_match = task_path.name == f'Dataset{task_num}_{dataset_name}'
            else:
                tasks_match = task_path.name == f'Task{task_num}'

        # V2-specific field validation
        v2_valid = True
        if self.pipeline_version == 2:
            dataset_name_valid = bool(self.input_fields['dataset_name'].text().strip())
            model_type_valid = self.input_fields['model_type'].text().strip().lower() in ["infant", "lifespan"]
            v2_valid = dataset_name_valid and model_type_valid

        return all([paths_valid, modality_valid, task_number_valid,
                    distribution_valid, synth_amt_valid, tasks_match, v2_valid])

    # ------------------------------------------------------------------
    # Running the pipeline
    # ------------------------------------------------------------------

    def _get_step_selections(self):
        selections = []
        for checkbox in self.ui.checkBoxes:
            selections.append(1 if checkbox.isChecked() else 0)
        return str(selections)

    def _update_status(self, message):
        print(message)
        self.ui.menuiuhwuaibfa.setTitle(message)

    def run_program(self):
        """Handle run / cancel button click"""
        if not self.is_running:
            if any(w.text() == "" for w in self.input_fields.values()):
                self._update_status("Please fill out all input fields")
                return
            if not self._validate_inputs():
                self._update_status("Make sure all inputs are valid")
                return

            self._update_status("Running...")
            self.step_selections = self._get_step_selections()

            if self.pipeline_version == 2:
                self.worker_thread = PipelineWorkerThreadV2(
                    Path(self.input_fields['dcan_path'].text().strip()),
                    Path(self.input_fields['task_path'].text().strip()),
                    Path(self.input_fields['synth_path'].text().strip()),
                    Path(self.input_fields['raw_data_base_path'].text().strip()),
                    Path(self.input_fields['results_path'].text().strip()),
                    Path(self.input_fields['trained_models_path'].text().strip()),
                    self.input_fields['modality'].text().strip().lower(),
                    self.input_fields['task_number'].text().strip(),
                    self.input_fields['distribution'].text().strip().lower(),
                    self.input_fields['synth_img_amt'].text().strip(),
                    self.input_fields['dataset_name'].text().strip(),
                    self.input_fields['model_type'].text().strip().lower(),
                    self.script_dir,
                    self.step_selections
                )
            else:
                self.worker_thread = PipelineWorkerThread(
                    Path(self.input_fields['dcan_path'].text().strip()),
                    Path(self.input_fields['task_path'].text().strip()),
                    Path(self.input_fields['synth_path'].text().strip()),
                    Path(self.input_fields['raw_data_base_path'].text().strip()),
                    Path(self.input_fields['results_path'].text().strip()),
                    Path(self.input_fields['trained_models_path'].text().strip()),
                    self.input_fields['modality'].text().strip().lower(),
                    self.input_fields['task_number'].text().strip(),
                    self.input_fields['distribution'].text().strip().lower(),
                    self.input_fields['synth_img_amt'].text().strip(),
                    self.script_dir,
                    self.step_selections
                )

            self.worker_thread.finished.connect(self.on_pipeline_finished)
            self.worker_thread.start()
            self.is_running = True
            self.ui.pushButton.setText('Cancel')

        else:
            self._update_status("Program Stopped")
            self.worker_thread.stop_program()

    def on_pipeline_finished(self):
        self.is_running = False
        self.step_selections = []
        self.ui.pushButton.setText('Run')

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    def browse_path(self, field_name, default_path):
        field_widget = self.input_fields[field_name]
        current_path = field_widget.text() or default_path
        selected_path = QFileDialog.getExistingDirectory(self, "Select Directory", current_path)
        if selected_path:
            field_widget.setText(str(selected_path))

    def toggle_all_checkboxes(self):
        all_checked = all(cb.isChecked() for cb in self.ui.checkBoxes)
        for cb in self.ui.checkBoxes:
            cb.setChecked(not all_checked)

    def populate_inputs(self):
        if self.ui.comboBox_preset.currentIndex() < 0:
            return
        preset_name = self.ui.comboBox_preset.currentText().strip()
        preset_path = self.script_dir / self.presets_dir_name / f"{preset_name}{PRESET_EXTENSION}"
        if not preset_path.exists():
            self._update_status("File Does Not Exist")
            return
        with open(preset_path) as f:
            lines = [line for line in f.readlines() if line.strip()]
        for line in lines:
            parts = line.strip().split('=', 1)
            if parts[0] in self.input_fields:
                if len(parts) == 1:
                    self.input_fields[parts[0]].clear()
                elif len(parts) == 2:
                    self.input_fields[parts[0]].setText(parts[1])
        self._update_status("Preset Loaded")

    def save_preset(self):
        preset_name = self.ui.line_save_preset.text().strip()
        if not preset_name:
            return
        if all(w.text().strip() == "" for w in self.input_fields.values()):
            self._update_status("Please fill out at least one input")
            return

        presets_dir = self.script_dir / self.presets_dir_name
        presets_dir.mkdir(parents=True, exist_ok=True)
        preset_path = presets_dir / f"{preset_name}{PRESET_EXTENSION}"

        if self.ui.check_overwrite.isChecked() and preset_path.exists():
            preset_path.unlink()
            self.ui.comboBox_preset.removeItem(self.ui.comboBox_preset.findText(preset_name))
            self.ui.comboBox_remove_preset.removeItem(self.ui.comboBox_remove_preset.findText(preset_name))

        if preset_path.exists():
            self._update_status("File Already Exists")
            return

        with open(preset_path, "w") as f:
            for key, widget in self.input_fields.items():
                f.write(f"{key}={widget.text().strip()}\n")

        self.ui.comboBox_preset.setStyleSheet("")
        self.ui.comboBox_preset.insertItem(
            self._find_alphabetical_index(self.ui.comboBox_preset, preset_name), preset_name)
        self.ui.comboBox_preset.setCurrentIndex(self.ui.comboBox_preset.findText(preset_name))

        self.ui.comboBox_remove_preset.setStyleSheet("")
        self.ui.comboBox_remove_preset.insertItem(
            self._find_alphabetical_index(self.ui.comboBox_remove_preset, preset_name), preset_name)

        if self.ui.comboBox_preset.count() == 1:
            self._setup_searchable_combobox(self.ui.comboBox_preset, '-- Select Preset --')
            self._setup_searchable_combobox(self.ui.comboBox_remove_preset, '-- Select Preset --')

        if self.ui.comboBox_remove_preset.currentText().strip():
            self.ui.comboBox_remove_preset.setCurrentIndex(-1)

        self.ui.comboBox_preset.setStyleSheet(PyQt5_stylesheets.load_stylesheet_pyqt5(style="style_Dark"))
        self.ui.comboBox_remove_preset.setStyleSheet(PyQt5_stylesheets.load_stylesheet_pyqt5(style="style_Dark"))

        self._update_status("Preset Saved")

    def remove_preset(self):
        if self.ui.comboBox_remove_preset.currentIndex() < 0:
            return
        preset_name = self.ui.comboBox_remove_preset.currentText().strip()
        preset_path = self.script_dir / self.presets_dir_name / f"{preset_name}{PRESET_EXTENSION}"
        if not preset_path.exists():
            self._update_status("File Does Not Exist")
            return

        dialog = CustomDialog()
        if not dialog.exec():
            return

        preset_path.unlink()
        current_selection = self.ui.comboBox_preset.currentText().strip()
        if current_selection == preset_name:
            self.ui.comboBox_preset.setCurrentIndex(-1)

        self.ui.comboBox_preset.removeItem(self.ui.comboBox_preset.findText(preset_name))
        self.ui.comboBox_remove_preset.removeItem(self.ui.comboBox_remove_preset.findText(preset_name))

        if current_selection and current_selection != preset_name:
            self.ui.comboBox_preset.setCurrentIndex(self.ui.comboBox_preset.findText(current_selection))

        self.ui.comboBox_remove_preset.setCurrentIndex(-1)

        if self.ui.comboBox_preset.count() < 1:
            self._setup_empty_combobox(self.ui.comboBox_preset, '-- No Presets --')
            self._setup_empty_combobox(self.ui.comboBox_remove_preset, '-- No Presets --')

        self._update_status("Preset Removed")

    def clear_inputs(self):
        for widget in self.input_fields.values():
            widget.clear()

    def closeEvent(self, event):
        print("CLOSING")
        if not self.is_running:
            event.accept()
            return
        reply = QMessageBox.question(
            self, 'Close Confirmation',
            "A program is currently running. Quitting now will cause it to stop at its current step, "
            "you will be able to start from here again if you wish to continue later. "
            "Are you sure you want to quit?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.run_program()
            event.accept()
        else:
            event.ignore()

# endregion


# region ### LOGIN WINDOW CLASS ###

class LoginWindow(QtWidgets.QMainWindow, Ui_LoginWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.script_dir = Path(__file__).resolve().parent

        # Populate combobox — show presets from whichever version is currently selected
        self._load_presets_for_version(self._selected_version())

        # Update the preset list whenever the version radio changes
        self.radio_v1.toggled.connect(self._on_version_toggled)
        self.radio_v2.toggled.connect(self._on_version_toggled)

        self.button_launch_ui.setText('Launch UI')
        self.button_launch_ui.clicked.connect(self.launch_main_ui)

    def _selected_version(self):
        return 2 if self.radio_v2.isChecked() else 1

    def _on_version_toggled(self):
        """Reload preset combobox when the user switches version"""
        self.comboBox.clear()
        self.comboBox.setCurrentIndex(-1)
        self._load_presets_for_version(self._selected_version())

    def _load_presets_for_version(self, version):
        presets_dir_name = PRESETS_DIR_V2 if version == 2 else PRESETS_DIR_V1
        presets_dir = self.script_dir / presets_dir_name

        for file in (presets_dir.iterdir() if presets_dir.exists() else []):
            if file.suffix == PRESET_EXTENSION:
                name = file.stem
                self.comboBox.insertItem(self._find_alphabetical_index(self.comboBox, name), name)

        if self.comboBox.count() < 1:
            self.comboBox.setEditable(False)
            self.comboBox.setPlaceholderText('-- No Presets --')
            self.comboBox.setStyleSheet(GRAY_BACKGROUND)
        else:
            self.comboBox.setEditable(True)
            self.comboBox.completer().setCompletionMode(QtWidgets.QCompleter.PopupCompletion)
            self.comboBox.setInsertPolicy(QComboBox.NoInsert)
            self.comboBox.lineEdit().setPlaceholderText('-- Select Preset --')

        self.comboBox.setCurrentIndex(-1)

    def _find_alphabetical_index(self, combo_box, item):
        items = [combo_box.itemText(i) for i in range(combo_box.count())]
        items.append(item)
        items.sort(key=str.upper)
        return items.index(item)

    def launch_main_ui(self):
        """Open the main window with the correct pipeline version"""
        selected_preset = self.comboBox.currentText().strip()
        if selected_preset and self.comboBox.findText(selected_preset) == -1:
            return

        version = self._selected_version()
        self.main_window = Window(pipeline_version=version)
        self.main_window.show()

        if selected_preset:
            idx = self.main_window.ui.comboBox_preset.findText(selected_preset)
            self.main_window.ui.comboBox_preset.setCurrentIndex(idx)
            self.main_window.populate_inputs()

        self.close()

# endregion


# region ### MAIN ###
def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Windows')
    login_window = LoginWindow()
    login_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
# endregion