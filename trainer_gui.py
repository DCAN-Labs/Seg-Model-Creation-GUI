import sys
import os
import subprocess
import psutil
from pathlib import Path

from main_window import Ui_MainWindow
from login_window import Ui_LoginWindow
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QObject, QThread, pyqtSignal, QTimer, Qt
import PyQt5_stylesheets
from custom_widgets import *

# region ### CONSTANTS ###
PRESETS_DIR = "automation_presets"
PRESET_EXTENSION = ".config"
GRAY_BACKGROUND = "background-color: rgb(137, 137, 137)"
# endregion


# region ### WORKER THREAD CLASS ###
class PipelineWorkerThread(QtCore.QThread):
    """Thread for running the training pipeline without blocking the GUI"""
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
        
        # Build command arguments
        cmd = [
            "python", str(pipeline_script),
            self.dcan_path, self.task_path, self.synth_path,
            self.raw_path, self.results_path, self.trained_path,
            self.modality, self.task_num, self.distribution,
            self.synth_amt, self.step_selections
        ]
        
        # Start subprocess
        process = subprocess.Popen(cmd, stdout=None, stderr=None)
        self.processes.append(process)
        process.wait()
              
        # Do certain things depending on how the program stopped
        self.cancel_jobs()
        if process.returncode == 0:  # Complete normally
            pass
        elif not self.quit_program:
            print("AN ERROR HAS OCCURRED")  # There was an error
        elif self.quit_program:
            print("PROCESS STOPPED")  # User stopped program manually
  
        self.finished.emit()  # Tells the program that the thread has finished
           
    def stop_program(self):
        """Specifically for when user stops program manually
        Cancels all subprocesses that are currently running"""
        if len(self.processes) > 0:
            self.quit_program = True
            
            print("Stopping Process...")
            parent = psutil.Process(self.processes[-1].pid)
            try:
                for child in parent.children(recursive=True):  # Kill current subprocess and all child subprocesses
                    child.kill()
            except:
                pass
            parent.kill()
# endregion


# region ### MAIN WINDOW CLASS ###
class Window(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
                
        # Get the directory of this file
        self.script_dir = Path(__file__).resolve().parent
        os.chdir(self.script_dir)
        
        # Initialize state variables
        self.worker_thread = None
        self.is_running = False
        self.step_selections = []
        
        # Put all input fields in a dictionary, used for presets
        self.input_fields = {
            'dcan_path': self.line_dcan_path,
            'synth_path': self.line_synth_path,
            'task_path': self.line_task_path,
            'raw_data_base_path': self.line_raw_data_base_path,
            'modality': self.line_modality,
            'task_number': self.line_task_number,
            'distribution': self.line_distribution,
            'synth_img_amt': self.line_synth_img_amt,
            'results_path': self.line_results_path,
            'trained_models_path': self.line_trained_models_path,
        }
        
        # Set up presets
        self._initialize_preset_comboboxes()
        
        # Connect button signals
        self.pushButton.setText('Run')
        self.pushButton.clicked.connect(self.run_program)
        self.pushButton_2.setText('Populate Preset')
        self.pushButton_2.clicked.connect(self.populate_inputs)
        self.button_clear.clicked.connect(self.clear_inputs)
        self.button_save.clicked.connect(self.save_preset)
        self.button_remove.clicked.connect(self.remove_preset)
        self.button_select_all.clicked.connect(self.toggle_all_checkboxes)
        self.button_browse_1.clicked.connect(lambda: self.browse_path('dcan_path', str(Path.home())))
        self.button_browse_2.clicked.connect(lambda: self.browse_path('synth_path', str(Path.home())))
        self.button_browse_3.clicked.connect(lambda: self.browse_path('task_path', "/"))
        self.button_browse_4.clicked.connect(lambda: self.browse_path('raw_data_base_path', "/"))
    
    def _initialize_preset_comboboxes(self):
        """Load and populate preset comboboxes"""
        presets_dir = self.script_dir / PRESETS_DIR
        
        for file in presets_dir.iterdir() if presets_dir.exists() else []:
            if file.suffix == PRESET_EXTENSION:
                preset_name = file.stem
                self.comboBox_preset.insertItem(self._find_alphabetical_index(self.comboBox_preset, preset_name), preset_name)
                self.comboBox_remove_preset.insertItem(self._find_alphabetical_index(self.comboBox_remove_preset, preset_name), preset_name)
        
        # Configure comboboxes based on whether presets exist
        if self.comboBox_preset.count() < 1:
            self._setup_empty_combobox(self.comboBox_preset, '-- No Presets --')
            self._setup_empty_combobox(self.comboBox_remove_preset, '-- No Presets --')
        else:
            self._setup_searchable_combobox(self.comboBox_preset, '-- Select Preset --')
            self._setup_searchable_combobox(self.comboBox_remove_preset, '-- Select Preset --')
    
    def _find_alphabetical_index(self, combo_box, item):
        """Used to help add items to comboboxes in alphabetical order"""
        items = [combo_box.itemText(i) for i in range(combo_box.count())]
        items.append(item)
        items.sort(key=str.upper)
        return items.index(item)
    
    def _setup_searchable_combobox(self, combo_box, placeholder):
        """Configure a combobox to be searchable"""
        combo_box.setEditable(True)
        combo_box.lineEdit().setPlaceholderText(placeholder)
        combo_box.completer().setCompletionMode(QtWidgets.QCompleter.PopupCompletion)
        combo_box.setInsertPolicy(QComboBox.NoInsert)
        combo_box.setCurrentIndex(-1)
    
    def _setup_empty_combobox(self, combo_box, placeholder):
        """Configure an empty disabled combobox"""
        combo_box.setEditable(False)
        combo_box.setPlaceholderText(placeholder)
        combo_box.setStyleSheet(GRAY_BACKGROUND)
    
    def _validate_inputs(self):
        """Makes sure all inputs are valid: paths exist, option inputs are valid, etc"""
        # Validate paths
        path_fields = ['dcan_path', 'synth_path', 'task_path', 'raw_data_base_path', 'results_path', 'trained_models_path']
        paths_valid = all(Path(self.input_fields[field].text().strip()).exists() for field in path_fields)
        
        # Validate modality
        modality = self.input_fields['modality'].text().strip().lower()
        modality_valid = modality in ["t1", "t2", "t1t2"]
        
        # Validate task number
        task_number_valid = self.input_fields['task_number'].text().isdigit()
        
        # Validate distribution
        distribution = self.input_fields['distribution'].text().strip().lower()
        distribution_valid = distribution in ["uniform", "normal"]
        
        # Validate synth image amount
        synth_amt_valid = self.input_fields['synth_img_amt'].text().strip().isdigit()
        
        # Validate task path matches task number
        tasks_match = True
        if task_number_valid and Path(self.input_fields['task_path'].text().strip()).exists():
            task_path = Path(self.input_fields['task_path'].text().strip())
            task_num = self.input_fields['task_number'].text().strip()
            tasks_match = task_path.name == f'Task{task_num}'
        
        return all([paths_valid, modality_valid, task_number_valid, distribution_valid, synth_amt_valid, tasks_match])
    
    def _get_step_selections(self):
        """Updates a list showing which steps the user wants to run"""
        selections = []
        for checkbox in self.checkBoxes:
            selections.append(1 if checkbox.isChecked() else 0)
        return str(selections)
    
    def _update_status(self, message):
        """Update status message in menu bar and console"""
        print(message)
        self.menuiuhwuaibfa.setTitle(message)
    
    def run_program(self):
        """Handle run/cancel button click"""
        # If process isn't currently running
        if not self.is_running:
            # Make sure all inputs are filled
            if any(widget.text() == "" for widget in self.input_fields.values()):
                self._update_status("Please fill out all input fields")
                return
            
            # Validate inputs
            if not self._validate_inputs():
                self._update_status("Make sure all inputs are valid")
                return
            
            # Start the pipeline
            self._update_status("Running...")
            self.step_selections = self._get_step_selections()
            
            # Start new worker thread to run main program. Allows UI to continue working along with it
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
            self.worker_thread.finished.connect(self.on_pipeline_finished)  # Listen for when process finishes
            self.worker_thread.start()
            self.is_running = True
            self.pushButton.setText('Cancel')
            
        # If process is currently running
        else:
            self._update_status("Program Stopped")
            self.worker_thread.stop_program()  # Stops subprocesses within thread. This will cause the finish signal to be sent
    
    def on_pipeline_finished(self):
        """Runs when the class receives the finished signal from the thread"""
        self.is_running = False
        self.step_selections = []
        self.pushButton.setText('Run')
    
    def browse_path(self, field_name, default_path):
        """Files browser"""
        field_widget = self.input_fields[field_name]
        current_path = field_widget.text() or default_path
        
        selected_path = QFileDialog.getExistingDirectory(self, "Select Directory", current_path)
        
        if selected_path:
            field_widget.setText(str(selected_path))
    
    def toggle_all_checkboxes(self):
        """Select or deselect all checkboxes"""
        all_checked = all(checkbox.isChecked() for checkbox in self.checkBoxes)
        
        for checkbox in self.checkBoxes:
            checkbox.setChecked(not all_checked)
    
    def populate_inputs(self):
        """Fills input boxes after reading from file"""
        if self.comboBox_preset.currentIndex() < 0:
            return
        
        preset_name = self.comboBox_preset.currentText().strip()
        preset_path = self.script_dir / PRESETS_DIR / f"{preset_name}{PRESET_EXTENSION}"
        
        if not preset_path.exists():
            self._update_status("File Does Not Exist")
            return
        
        with open(preset_path) as f:
            lines = [line for line in f.readlines() if line.strip()]  # Ignore blank lines
        
        for line in lines:
            parts = line.strip().split('=', 1)
            if parts[0] in self.input_fields:
                # If there is no info associated with a certain input, clear the input line
                if len(parts) == 1:
                    self.input_fields[parts[0]].clear()
                elif len(parts) == 2:
                    self.input_fields[parts[0]].setText(parts[1])
        
        self._update_status("Preset Loaded")
    
    def save_preset(self):
        """Saves preset data to a file"""
        preset_name = self.line_save_preset.text().strip()
        
        if not preset_name:
            return
        
        if all(widget.text().strip() == "" for widget in self.input_fields.values()):
            self._update_status("Please fill out at least one input")
            return
        
        preset_path = self.script_dir / PRESETS_DIR / f"{preset_name}{PRESET_EXTENSION}"
        
        # If overwrite is checked, delete the file if it exists already
        if self.check_overwrite.isChecked() and preset_path.exists():
            preset_path.unlink()
            self.comboBox_preset.removeItem(self.comboBox_preset.findText(preset_name))
            self.comboBox_remove_preset.removeItem(self.comboBox_remove_preset.findText(preset_name))
        
        # Make sure file doesn't exist yet and create presets data
        if preset_path.exists():
            self._update_status("File Already Exists")
            return
        
        with open(preset_path, "w") as f:
            for key, widget in self.input_fields.items():
                f.write(f"{key}={widget.text().strip()}\n")
        
        # Add to comboboxes
        self.comboBox_preset.setStyleSheet("")
        self.comboBox_preset.insertItem(self._find_alphabetical_index(self.comboBox_preset, preset_name), preset_name)
        self.comboBox_preset.setCurrentIndex(self.comboBox_preset.findText(preset_name))
        
        self.comboBox_remove_preset.setStyleSheet("")
        self.comboBox_remove_preset.insertItem(self._find_alphabetical_index(self.comboBox_remove_preset, preset_name), preset_name)
        
        # Enable comboboxes if this was the first preset
        if self.comboBox_preset.count() == 1:
            self._setup_searchable_combobox(self.comboBox_preset, '-- Select Preset --')
            self._setup_searchable_combobox(self.comboBox_remove_preset, '-- Select Preset --')
        
        if self.comboBox_remove_preset.currentText().strip():
            self.comboBox_remove_preset.setCurrentIndex(-1)
        
        self.comboBox_preset.setStyleSheet(PyQt5_stylesheets.load_stylesheet_pyqt5(style="style_Dark"))
        self.comboBox_remove_preset.setStyleSheet(PyQt5_stylesheets.load_stylesheet_pyqt5(style="style_Dark"))
        
        self._update_status("Preset Saved")
    
    def remove_preset(self):
        """Delete preset file if it exists"""
        if self.comboBox_remove_preset.currentIndex() < 0:
            return
        
        preset_name = self.comboBox_remove_preset.currentText().strip()
        preset_path = self.script_dir / PRESETS_DIR / f"{preset_name}{PRESET_EXTENSION}"
        
        if not preset_path.exists():
            self._update_status("File Does Not Exist")
            return
        
        # Creates popup asking user if they are sure they want to delete their preset
        dialog = CustomDialog()
        if not dialog.exec():
            return
        
        preset_path.unlink()
        
        # Remember current selection
        current_selection = self.comboBox_preset.currentText().strip()
        if current_selection == preset_name:
            self.comboBox_preset.setCurrentIndex(-1)
        
        # Remove from both comboboxes
        self.comboBox_preset.removeItem(self.comboBox_preset.findText(preset_name))
        self.comboBox_remove_preset.removeItem(self.comboBox_remove_preset.findText(preset_name))
        
        # Restore selection if it wasn't the deleted preset
        if current_selection and current_selection != preset_name:
            self.comboBox_preset.setCurrentIndex(self.comboBox_preset.findText(current_selection))
        
        self.comboBox_remove_preset.setCurrentIndex(-1)
        
        # If you deleted your only preset
        if self.comboBox_preset.count() < 1:
            self._setup_empty_combobox(self.comboBox_preset, '-- No Presets --')
            self._setup_empty_combobox(self.comboBox_remove_preset, '-- No Presets --')
        
        self._update_status("Preset Removed")
    
    def clear_inputs(self):
        """Clear all input fields"""
        for widget in self.input_fields.values():
            widget.clear()
    
    def closeEvent(self, event):
        """Override the close event to execute a function first"""
        print("CLOSING")
        
        if not self.is_running:
            event.accept()
            return
        
        reply = QMessageBox.question(
            self,
            'Close Confirmation',
            "A program is currently running. Quitting now will cause it to stop at its current step, "
            "you will be able to start from here again if you wish to continue later. "
            "Are you sure you want to quit?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.run_program()
            event.accept()  # Accept the event to close the window
        else:
            event.ignore()  # Ignore the event to prevent the window from closing
# endregion


# region ### LOGIN WINDOW CLASS ###
class LoginWindow(QtWidgets.QMainWindow, Ui_LoginWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
        # Get the directory of this file
        self.script_dir = Path(__file__).resolve().parent
        
        # Populate inputs based on the preset you selected
        presets_dir = self.script_dir / PRESETS_DIR
        for file in presets_dir.iterdir() if presets_dir.exists() else []:
            if file.suffix == PRESET_EXTENSION:
                preset_name = file.stem
                self.comboBox.insertItem(self._find_alphabetical_index(self.comboBox, preset_name), preset_name)
        
        # Configure combobox
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
        
        # Connect signals
        self.button_launch_ui.setText('Launch UI')
        self.button_launch_ui.clicked.connect(self.launch_main_ui)
    
    def _find_alphabetical_index(self, combo_box, item):
        """Used to sort combobox alphabetically"""
        items = [combo_box.itemText(i) for i in range(combo_box.count())]
        items.append(item)
        items.sort(key=str.upper)
        return items.index(item)
    
    def launch_main_ui(self):
        """Starts up main UI screen"""
        selected_preset = self.comboBox.currentText().strip()
        
        # Only launch if no preset selected or valid preset selected
        if selected_preset and self.comboBox.findText(selected_preset) == -1:
            return
        
        self.main_window = Window()
        self.main_window.show()
        
        # Load preset if one was selected
        if selected_preset:
            self.main_window.comboBox_preset.setCurrentIndex(
                self.main_window.comboBox_preset.findText(selected_preset)
            )
            self.main_window.populate_inputs()
        
        self.close()
# endregion


# region ### MAIN ###
def main():
    """Application entry point"""
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Windows')
    
    login_window = LoginWindow()
    login_window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
# endregion