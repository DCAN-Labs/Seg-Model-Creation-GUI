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
    '''Thread for running the training pipeline without blocking the GUI's functionality. This runs the orinigal nnUnet v1-based pipeline'''
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
        # Uses active jobs file to cancel all job ids listed
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
        # Start subprocess running the training pipeline and wait for it to finish, then cancel any remaining jobs if the process was stopped manually from the GUI
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
        
        # If the process finished on its own, do nothing. If it was stopped by the user, print a message. If it ended with an error, print a different message
        if process.returncode == 0:
            pass
        elif not self.quit_program:
            print("AN ERROR HAS OCCURRED")
        elif self.quit_program:
            print("PROCESS STOPPED")
        self.finished.emit()

    def stop_program(self):
        # Cancel subprocesses when user stops program manually from the GUI, then set a flag so that when the process finishes it knows it was stopped by the user and doesn't print an error message
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
    # Thread for running the training pipeline without blocking the GUI's functionality. This runs the new nnUnet v2-based pipeline, which has some differences in how it handles tasks and datasets so it required a separate thread class
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
        # Uses active jobs file to cancel all job ids listed
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
        # Start subprocess running the training pipeline and wait for it to finish, then cancel any remaining jobs if the process was stopped manually from the GUI
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
        # If the process finished on its own, do nothing. If it was stopped by the user, print a message. If it ended with an error, print a different message
        if process.returncode == 0:
            pass
        elif not self.quit_program:
            print("AN ERROR HAS OCCURRED")
        elif self.quit_program:
            print("PROCESS STOPPED")
        self.finished.emit()

    def stop_program(self):
        # Cancel subprocesses when user stops program manually from the GUI, then set a flag so that when the process finishes it knows it was stopped by the user and doesn't print an error message
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
    '''Main window class for the training pipeline GUI. This class handles both the original nnUnet v1-based pipeline and the new nnUnet v2-based pipeline'''

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

    ## Preset helpers ##

    def _initialize_preset_comboboxes(self):
        # Load and populate preset comboboxes from the version-appropriate folder
        presets_dir = self.script_dir / self.presets_dir_name

        # Populate preset selection and removal comboboxes with presets from the appropriate folder based on the selected pipeline version, in alphabetical order
        for file in (presets_dir.iterdir() if presets_dir.exists() else []):
            if file.suffix == PRESET_EXTENSION:
                name = file.stem
                self.ui.comboBox_preset.insertItem(
                    self._find_alphabetical_index(self.ui.comboBox_preset, name), name)
                self.ui.comboBox_remove_preset.insertItem(
                    self._find_alphabetical_index(self.ui.comboBox_remove_preset, name), name)
        # If there are no presets, set combobox to non-editable and show "No Presets" placeholder. If there are presets, set up combobox to be searchable and show "Select Preset" placeholder
        if self.ui.comboBox_preset.count() < 1:
            self._setup_empty_combobox(self.ui.comboBox_preset, '-- No Presets --')
            self._setup_empty_combobox(self.ui.comboBox_remove_preset, '-- No Presets --')
        else:
            self._setup_searchable_combobox(self.ui.comboBox_preset, '-- Select Preset --')
            self._setup_searchable_combobox(self.ui.comboBox_remove_preset, '-- Select Preset --')

    def _find_alphabetical_index(self, combo_box, item):
        # Helper function for inserting items into the preset comboboxes in alphabetical order
        items = [combo_box.itemText(i) for i in range(combo_box.count())]
        items.append(item)
        items.sort(key=str.upper)
        return items.index(item)

    def _setup_searchable_combobox(self, combo_box, placeholder):
        # Formats a combobox to be searchable with a placeholder
        combo_box.setEditable(True)
        combo_box.lineEdit().setPlaceholderText(placeholder)
        combo_box.completer().setCompletionMode(QtWidgets.QCompleter.PopupCompletion)
        combo_box.setInsertPolicy(QComboBox.NoInsert)
        combo_box.setCurrentIndex(-1)

    def _setup_empty_combobox(self, combo_box, placeholder):
        # Formats a combobox to show a placeholder and not be editable when there are no items to show
        combo_box.setEditable(False)
        combo_box.setPlaceholderText(placeholder)
        combo_box.setStyleSheet(GRAY_BACKGROUND)

    ## Validation ##

    def _validate_inputs(self):
        # Validate all inputs; v2 adds dataset_name and model_type checks. Just making sure all inputs make sense before starting the pipeline
        
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

        # Making sure that the task folder name matches the task number (and dataset name for v2)
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

    ## Running the pipeline ##

    def _get_step_selections(self):
        # Encode which steps the user has selected to run as a list of 1s and 0s, which will be passed to the pipeline and decoded there to determine which steps to run
        selections = []
        for checkbox in self.ui.checkBoxes:
            selections.append(1 if checkbox.isChecked() else 0)
        return str(selections)

    def _update_status(self, message):
        # Update the status message shown in the UI
        print(message)
        self.ui.menuiuhwuaibfa.setTitle(message)

    def run_program(self):
        #Handles run / cancel button click, starting the pipeline in a new thread if not currently running, or stopping the pipeline if it is currently running
        
        # If program is not currently running, validate inputs and start the pipeline in a new thread
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
        # If program is currently running, stop the pipeline and any active jobs
        else:
            self._update_status("Program Stopped")
            self.worker_thread.stop_program()

    def on_pipeline_finished(self):
        #Pipeline finished behavior
        self.is_running = False
        self.step_selections = []
        self.ui.pushButton.setText('Run')


    ## UI helpers ##

    def browse_path(self, field_name, default_path):
        #Some path input fields have a browse button that opens a file explorer to select the path instead of typing it out, this handles those button clicks
        
        field_widget = self.input_fields[field_name]
        current_path = field_widget.text() or default_path
        selected_path = QFileDialog.getExistingDirectory(self, "Select Directory", current_path)
        if selected_path:
            field_widget.setText(str(selected_path))

    def toggle_all_checkboxes(self):
        # If any checkbox is unchecked, check them all. If they are all checked, uncheck them all
        all_checked = all(cb.isChecked() for cb in self.ui.checkBoxes)
        for cb in self.ui.checkBoxes:
            cb.setChecked(not all_checked)

    def populate_inputs(self):
        # Populate input fields with values from the selected preset, if there is one. This looks for a preset file with the same name as the selected preset in the appropriate presets folder for the pipeline version, and populates fields based on the key=value pairs listed in that file
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
        # Save the current input field values as a preset with the name given in the preset name field. This creates a file in the appropriate presets folder for the pipeline version with key=value pairs for each input field
        preset_name = self.ui.line_save_preset.text().strip()
        if not preset_name:
            return
        if all(w.text().strip() == "" for w in self.input_fields.values()):
            self._update_status("Please fill out at least one input")
            return

        presets_dir = self.script_dir / self.presets_dir_name
        presets_dir.mkdir(parents=True, exist_ok=True)
        preset_path = presets_dir / f"{preset_name}{PRESET_EXTENSION}"

        # If the preset already exists and the overwrite checkbox is checked, delete the existing preset file and remove it from the comboboxes so that it can be replaced with the new one. If the preset already exists and the overwrite checkbox is not checked, show an error message and don't save
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

        # Select preset and remove preset combobox visual updates
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
        # Remove the preset file corresponding to the selected preset in the remove preset combobox
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

        preset_path.unlink() # Delete the preset file
        
        # Update preset selection and removal comboboxes to remove the deleted preset
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
        # Clear all input fields
        for widget in self.input_fields.values():
            widget.clear()

    def closeEvent(self, event):
        # Override the default close behavior to show a confirmation dialog if the user tries to close the window while the pipeline is running, since closing will stop the pipeline and any active jobs
        print("CLOSING")
        if not self.is_running: # If program is not running, just close the window
            event.accept()
            return
        
        reply = QMessageBox.question(
            self, 'Close Confirmation',
            "A program is currently running. Quitting now will cause it to stop at its current step, "
            "you will be able to start from here again if you wish to continue later. "
            "Are you sure you want to quit?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes: # If user confirms they want to quit, stop the pipeline and close the window
            self.run_program()
            event.accept()
        else: # If user cancels quitting, ignore the close event and keep the window open
            event.ignore()

# endregion


# region ### LOGIN WINDOW CLASS ###

class LoginWindow(QtWidgets.QMainWindow, Ui_LoginWindow):
    ''''Login window class, which is the first thing the user sees when they open the program. This just allows the user to select which version of the pipeline they want to run (v1 or v2), and then opens the main window with the appropriate pipeline version when they click the "Launch UI" button'''
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.script_dir = Path(__file__).resolve().parent

        # Populate combobox — show presets from whichever version is currently selected
        self._load_presets_for_version(self._selected_version())

        # Update the preset list whenever the version radio changes, since v1 and v2 have different presets folders
        self.radio_v1.toggled.connect(self._on_version_toggled)
        self.radio_v2.toggled.connect(self._on_version_toggled)

        self.button_launch_ui.setText('Launch UI')
        self.button_launch_ui.clicked.connect(self.launch_main_ui)

    def _selected_version(self):
        # Helper function to determine which pipeline version is currently selected based on the radio buttons
        if (self.radio_v1.isChecked()):
            return 1
        else:
            return 2

    def _on_version_toggled(self):
        # Reload preset combobox to reflect the presets available for the currently selected pipeline version
        self.comboBox.clear()
        self.comboBox.setCurrentIndex(-1)
        self._load_presets_for_version(self._selected_version())

    def _load_presets_for_version(self, version):
        # Load presets from the appropriate folder based on the selected pipeline version and populate the preset selection combobox, in alphabetical order
        presets_dir_name = PRESETS_DIR_V2 if version == 2 else PRESETS_DIR_V1
        presets_dir = self.script_dir / presets_dir_name

        for file in (presets_dir.iterdir() if presets_dir.exists() else []):
            if file.suffix == PRESET_EXTENSION:
                name = file.stem
                self.comboBox.insertItem(self._find_alphabetical_index(self.comboBox, name), name)

        # If there are no presets, set combobox to non-editable and show "No Presets" placeholder. If there are presets, set up combobox to be searchable and show "Select Preset" placeholder
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
        # Helper function for inserting items into the preset combobox in alphabetical order
        items = [combo_box.itemText(i) for i in range(combo_box.count())]
        items.append(item)
        items.sort(key=str.upper)
        return items.index(item)

    def launch_main_ui(self):
        # Open the main window with the correct pipeline version
        selected_preset = self.comboBox.currentText().strip()
        if selected_preset and self.comboBox.findText(selected_preset) == -1:
            return

        version = self._selected_version()
        self.main_window = Window(pipeline_version=version)
        self.main_window.show()

        # If a preset was selected on the login screen, automatically populate the main window input fields with that preset's values when it launches
        if selected_preset:
            idx = self.main_window.ui.comboBox_preset.findText(selected_preset)
            self.main_window.ui.comboBox_preset.setCurrentIndex(idx)
            self.main_window.populate_inputs()

        self.close()

# endregion


# region ### MAIN ###
def main():
    # Create and show the login window, which will then open the main window when the user clicks the "Launch UI" button
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Windows')
    login_window = LoginWindow()
    login_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
# endregion