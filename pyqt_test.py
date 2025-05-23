import sys
import os
import signal
import subprocess
import psutil
from pathlib import Path

from test1 import Ui_MainWindow
from k_login import Ui_LoginWindow
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QObject, QThread, pyqtSignal, QTimer, Qt
import PyQt5_stylesheets
from custom_widgets import *

class Thread(QtCore.QThread):
    finished = pyqtSignal()
    dcan_path = ''
    task_path = ''
    synth_path = ''
    raw_path = ''
    results_path = ''
    trained_path = ''
    modality = ''
    task_num = ''
    distribution = ''
    synth_amt = ''
    processes = []
    script_dir = ""
    check_list=[]

    quit_program = False

    def __init__(self, dcan_path, task_path, synth_path, raw_path, results_path, trained_path, modality, task_num, distribution, synth_amt, script_dir, check_list):
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
        self.check_list = check_list

    def cancel_jobs(self):
        # Uses active jobs file to cancel al job ids listedF
        with open(os.path.join(self.script_dir, "scripts", "slurm_scripts", self.task_num, "active_jobs.txt"), 'r') as f:
            lines = f.readlines()
                    
        for i in lines:
            if i.strip() != '':
                subprocess.run(["scancel", i.strip()])
                
        os.remove(os.path.join(self.script_dir, "scripts", "slurm_scripts", self.task_num, "active_jobs.txt"))

    def run(self):
        # Start subprocess and wait for it to finish
        p = subprocess.Popen(["python", os.path.join(self.script_dir, "automation_test.py"), self.dcan_path, self.task_path, self.synth_path, self.raw_path, self.results_path, self.trained_path, self.modality, self.task_num, self.distribution, self.synth_amt, self.check_list], stdout=None, stderr=None) 
        self.processes.append(p)
        p.wait()
              
        # Do certain things depending on how the program stopped
        self.cancel_jobs()
        if p.returncode == 0: # Complete normally
            pass
        elif not self.quit_program:      
            print("AN ERROR HAS OCCURED") # There was an erro
        elif self.quit_program:
            print("PROCESS STOPPED") # User stopped program manually
  
        self.finished.emit() # Tells the program that the thread has finished
           
    def stop_program(self):  
        # Specifically for when user stops program manually
        # Cancels all subprocesses that are currently running
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

class Window(QtWidgets.QMainWindow, Ui_MainWindow):
    temp_thread = None
    inputDict = {}
    script_dir = ''
    t_num = ''
    running = False
    
    def __init__(self):
        super().__init__()
        self.setupUi(self)
                
        # Get the directory of this file
        self.script_dir = os.path.abspath(os.path.dirname(__file__))
        os.chdir(self.script_dir)  
        
        # Set up presets
        for file in os.listdir(os.path.join(self.script_dir, "automation_presets")):
            if ".config" in file:
                file = file[:-7]
                self.comboBox_preset.insertItem(self.findAlphabeticalIndex(self.comboBox_preset, file), file)
                self.comboBox_remove_preset.insertItem(self.findAlphabeticalIndex(self.comboBox_remove_preset, file), file)
            
        if self.comboBox_preset.count() < 1:
            self.comboBox_preset.setEditable(False)
            self.comboBox_remove_preset.setEditable(False)
            self.comboBox_preset.setPlaceholderText('No Presets')
            self.comboBox_remove_preset.setPlaceholderText('No Presets')
            self.comboBox_preset.setStyleSheet("background-color: rgb(137, 137, 137)")
            self.comboBox_remove_preset.setStyleSheet("background-color: rgb(137, 137, 137)")
            
        self.comboBox_preset.setCurrentIndex(-1)
        self.comboBox_remove_preset.setCurrentIndex(-1)
        
        # Put all input fields in a dictionary, used for presets
        self.inputDict['dcan_path'] = self.line_dcan_path
        self.inputDict['synth_path'] = self.line_synth_path
        self.inputDict['task_path'] = self.line_task_path
        self.inputDict['raw_data_base_path'] = self.line_raw_data_base_path
        self.inputDict['modality'] = self.line_modality
        self.inputDict['task_number'] = self.line_task_number
        self.inputDict['distribution'] = self.line_distribution
        self.inputDict['synth_img_amt'] = self.line_synth_img_amt
        self.inputDict['results_path'] = self.line_results_path
        self.inputDict['trained_models_path'] = self.line_trained_models_path
        
        self.check_list = []
        
        # Some setup stuff
        self.menuiuhwuaibfa.setTitle("TEST PROGRAM")
        self.pushButton.setText('run')
        self.pushButton_2.setText('Populate Preset')
        self.pushButton.clicked.connect(self.run_program)
        self.pushButton_2.clicked.connect(self.populate_inputs)
        self.button_clear.clicked.connect(self.clear_inputs)
        self.button_save.clicked.connect(self.save_preset)
        self.button_remove.clicked.connect(self.remove_preset)
        self.button_select_all.clicked.connect(self.select_all_presets)
        self.button_browse_1.clicked.connect(lambda: self.browse(1))
        self.button_browse_2.clicked.connect(lambda: self.browse(2))
        self.button_browse_3.clicked.connect(lambda: self.browse(3))
        self.button_browse_4.clicked.connect(lambda: self.browse(4))
    
    def findAlphabeticalIndex(self, combo, item):
        # Used to help add items to comboboxes in alphabetical order
        combo_list = [combo.itemText(i) for i in range(combo.count())]
        combo_list.append(item)
        combo_list.sort(key=lambda i: i.upper())
        return combo_list.index(item)
    
    def check_inputs(self):
        # Makes sure all inputs are vald: paths exist, option inputs are valid, etc
        inp_dcan_path = os.path.exists(Path(self.line_dcan_path.text().strip()))
        inp_synth_path = os.path.exists(Path(self.line_synth_path.text().strip()))
        inp_task_path = os.path.exists(Path(self.line_task_path.text().strip()))
        inp_raw_data_base_path = os.path.exists(Path(self.line_raw_data_base_path.text().strip()))
        inp_results_path= os.path.exists(Path(self.line_results_path.text().strip()))
        inp_trained_models_path = os.path.exists(Path(self.line_trained_models_path.text().strip()))
        inp_modality = self.line_modality.text().strip().lower() == "t1" or self.line_modality.text().strip().lower() == "t2" or self.line_modality.text().strip().lower() == "t1t2"
        # TODO: update task number check
        inp_task_number = self.line_task_number.text().isdigit()
        inp_distribution = self.line_distribution.text().strip().lower() == "uniform" or self.line_distribution.text().strip().lower() == "normal"
        inp_synth_img_amt = self.line_synth_img_amt.text().strip().isdigit()
        
        tasks_match = True
        if inp_task_number and inp_task_path:
            tasks_match = os.path.split(Path(self.line_task_path.text().strip()))[-1] == f'Task{self.line_task_number.text().strip()}'
        
        arguments = [inp_dcan_path, inp_synth_path, inp_task_path, inp_raw_data_base_path, inp_modality, inp_distribution, inp_synth_img_amt, inp_results_path, inp_trained_models_path, tasks_match] 
        
        if all(i == True for i in arguments):
            return True
        return False
    
    def run_program(self):
        
        # If process isn't currently running
        if self.running == False:
            # Make sure all inputs are filled
            if any(inp.text() == "" for inp in self.inputDict.values()):
                print("Please fill out all input fields")
                self.menuiuhwuaibfa.setTitle("Please fill out all input fields")
            else:
                if self.check_inputs():
                    self.menuiuhwuaibfa.setTitle("Running...")
                    self.check_status()
                    #running_tasks.append*
                    # Start new worker thread to run main program. Allows UI to continue working along with it
                    self.temp_thread = Thread(Path(self.line_dcan_path.text().strip()), Path(self.line_task_path.text().strip()), Path(self.line_synth_path.text().strip()), Path(self.line_raw_data_base_path.text().strip()), Path(self.line_results_path.text().strip()), Path(self.line_trained_models_path.text().strip()),
                                            self.line_modality.text().strip().lower(), self.line_task_number.text().strip(), self.line_distribution.text().strip().lower(), self.line_synth_img_amt.text().strip(), self.script_dir, str(self.check_list))
                    #self.temp_thread.finished.connect(lambda: self.pushButton.setText('run')) # Listen for when process finishes
                    self.temp_thread.finished.connect(self.on_finish_thread) # Listen for when process finishes
                    self.temp_thread.start()
                    self.running = True
                    self.pushButton.setText('cancel')
                else:
                    print("Make sure all inputs are valid")
                    self.menuiuhwuaibfa.setTitle("Make sure all inputs are valid")
        # If process is currently running
        elif self.running == True:
            self.menuiuhwuaibfa.setTitle("Program Stopped")
            self.temp_thread.stop_program() # Stops subrocesses within thread. This will cause the finish signal to be sent
            
    def browse(self,num):
    # Files browser
        path_dic = {
    1: (self.line_dcan_path, os.path.expanduser("~")),
    2: (self.line_task_path, "/"),
    3: (self.line_synth_path, os.path.expanduser("~")),
    4: (self.line_raw_data_base_path, "/")
        }

        line_edit,default_path=path_dic.get(num)

        dialog = QFileDialog.getExistingDirectory(self, "Select Directory", line_edit.text() or default_path)

        if dialog:
            line_edit.setText(str(dialog))
        
    def check_status(self):  
        # Updates a list showing which steps the user wants to run  
        for checkBox in self.checkBoxes:
            self.check_list.append(1 if checkBox.isChecked() else 0)
        
        print(str(self.check_list))  # You can print or use this list as needed
        
    def select_all_presets(self):
        temp =True
        for checkBox in self.checkBoxes: #checks to see if all of the boxes are already selected
            if not checkBox.isChecked():
                temp=False
        if temp == False:
            for checkBox in self.checkBoxes: #selects all boxes
                checkBox.setChecked(True)
        else:
            for checkBox in self.checkBoxes: #deselects all boxes
                checkBox.setChecked(False)

    def on_finish_thread(self):
        # Runs when the class recieves the finished signal from the thread
        self.running = False
        self.check_list = []
        self.pushButton.setText('run')
            
    def populate_inputs(self):
        # Fills input boxes after reading from file
        if self.comboBox_preset.currentIndex()>=0:
            if os.path.isfile(os.path.join(self.script_dir, "automation_presets", f"{self.comboBox_preset.currentText().strip()}.config")):
                f = open(os.path.join(self.script_dir, "automation_presets", f"{self.comboBox_preset.currentText().strip()}.config"))
                lines = [line for line in f.readlines() if line.strip()] # Ignore blank lines

                for line in lines:
                    line = line.strip().split('=')
                    if line[0] in self.inputDict.keys():
                        # If there is no info associated with a certain input, clear the input line
                        if len(line) == 1:
                            self.inputDict[line[0]].clear() 
                        elif len(line) == 2:
                            self.inputDict[line[0]].setText(line[1])
                            
                f.close()
                print("Preset Loaded")
                self.menuiuhwuaibfa.setTitle("Preset Loaded")
            else:
                print("File Does Not Exist")
                self.menuiuhwuaibfa.setTitle("File Does Not Exist")
            
    def save_preset(self):
        # Saves preset data to a file
        if self.line_save_preset.text().strip() == "":
            return
        if all(inp.text().strip() == "" for inp in self.inputDict.values()):
            print("Please fill out at least one input")
            self.menuiuhwuaibfa.setTitle("Please fill out at least one input")
            return
        # If overwrite is checked, delete the file if it exists already
        if self.check_overwrite.isChecked(): 
            if os.path.isfile(os.path.join(self.script_dir, "automation_presets", f"{self.line_save_preset.text().strip()}.config")):
                os.remove(os.path.join(self.script_dir, "automation_presets", f"{self.line_save_preset.text().strip()}.config"))
                self.comboBox_preset.removeItem(self.comboBox_preset.findText(self.line_save_preset.text().strip()))
                self.comboBox_remove_preset.removeItem(self.comboBox_remove_preset.findText(self.line_save_preset.text().strip()))
        # Make sure file doesn't exist yet and create presets data
        if not os.path.isfile(os.path.join(self.script_dir, "automation_presets", f"{self.line_save_preset.text().strip()}.config")):
            f = open(os.path.join(self.script_dir, "automation_presets", f"{self.line_save_preset.text().strip()}.config"), "w")
            for key, val in self.inputDict.items():
                f.write(f"{key}={val.text().strip()}\n")
            f.close()
            
            self.comboBox_preset.setStyleSheet("")
            self.comboBox_preset.insertItem(self.findAlphabeticalIndex(self.comboBox_preset, self.line_save_preset.text().strip()), self.line_save_preset.text().strip())
            self.comboBox_preset.setCurrentIndex(self.comboBox_preset.findText(self.line_save_preset.text().strip()))
            
            self.comboBox_preset.setEditable(True)
            self.comboBox_preset.lineEdit().setPlaceholderText('-- Select Preset --')
                        
            self.comboBox_preset.completer().setCompletionMode(QtWidgets.QCompleter.PopupCompletion) 
            self.comboBox_preset.setInsertPolicy(QComboBox.NoInsert) 
            
            self.comboBox_remove_preset.setStyleSheet("")
            self.comboBox_remove_preset.insertItem(self.findAlphabeticalIndex(self.comboBox_remove_preset, self.line_save_preset.text().strip()), self.line_save_preset.text().strip())
            #self.comboBox_remove_preset.setCurrentIndex(self.comboBox_remove_preset.findText(self.line_save_preset.text().strip()))
            self.comboBox_remove_preset.setEditable(True)
            self.comboBox_remove_preset.lineEdit().setPlaceholderText('-- Select Preset --')
            self.comboBox_remove_preset.completer().setCompletionMode(QtWidgets.QCompleter.PopupCompletion) 
            self.comboBox_remove_preset.setInsertPolicy(QComboBox.NoInsert)
            #self.comboBox_preset.setCurrentIndex(-1)
            if self.comboBox_remove_preset.currentText().strip() != '':
                self.comboBox_remove_preset.setCurrentIndex(-1)
            
            self.comboBox_preset.setStyleSheet(PyQt5_stylesheets.load_stylesheet_pyqt5(style="style_Dark"))
            self.comboBox_remove_preset.setStyleSheet(PyQt5_stylesheets.load_stylesheet_pyqt5(style="style_Dark"))
            
            print("Preset Saved")
            self.menuiuhwuaibfa.setTitle("Preset Saved")
        else:
            print("File Already Exists")
            self.menuiuhwuaibfa.setTitle("File Already Exists")
            
    def remove_preset(self, event):
        # Delete file if it exists
        if self.comboBox_remove_preset.currentIndex()>=0:
            if os.path.isfile(os.path.join(self.script_dir, "automation_presets", f"{self.comboBox_remove_preset.currentText().strip()}.config")):
                
                # Creates popup asking user if they are sure they want to delete their preset
                dlg = CustomDialog()
                if dlg.exec():
                    os.remove(os.path.join(self.script_dir, "automation_presets", f"{self.comboBox_remove_preset.currentText().strip()}.config"))
                    temp_text = ''
                    if self.comboBox_preset.currentText().strip() == self.comboBox_remove_preset.currentText().strip():
                        self.comboBox_preset.setCurrentIndex(-1)
                    else:
                        temp_text = self.comboBox_preset.currentText().strip()
                    self.comboBox_preset.removeItem(self.comboBox_remove_preset.findText(self.comboBox_remove_preset.currentText().strip()))
                    self.comboBox_remove_preset.removeItem(self.comboBox_remove_preset.findText(self.comboBox_remove_preset.currentText().strip()))
                    
                    if temp_text != '':
                        self.comboBox_preset.setCurrentIndex(self.comboBox_preset.findText(temp_text))
                        
                    self.comboBox_remove_preset.setCurrentIndex(-1)
                
                    # If you deleted your only preset
                    if self.comboBox_preset.count() < 1:
                        self.comboBox_preset.setEditable(False)
                        self.comboBox_remove_preset.setEditable(False)
                        self.comboBox_preset.setPlaceholderText('-- No Presets --')
                        self.comboBox_remove_preset.setPlaceholderText('-- No Presets --')
                        # self.comboBox_remove_preset.setItemText(0, 'You do not have any presets')
                        self.comboBox_remove_preset.setStyleSheet("background-color: rgb(137, 137, 137)")
                        
                    
                        self.comboBox_preset.setEditable(False) 
                        #self.comboBox_preset.setItemText(0, 'You do not have any presets')
                        self.comboBox_preset.setStyleSheet("background-color: rgb(137, 137, 137)")
            
                    print("Preset Removed")
                    self.menuiuhwuaibfa.setTitle("Preset Removed")
                              
            else:
                print("File Does Not Exist")
                self.menuiuhwuaibfa.setTitle("File Does Not Exist")
        
    def clear_inputs(self):
        # Clear all input fields
        for key in self.inputDict.keys():
            self.inputDict[key].clear()
        
    def sleepSec(self, sec):
        # Disable buttons if needed
        self.pushButton.setEnabled(False)
        QTimer.singleShot(sec * 1000, lambda: self.pushButton.setEnabled(True))
        
    def closeEvent(self, event):

        print("CLOSING")
        # Override the close event to execute a function first
        if self.running == True:
            reply = QMessageBox.question(self, 'Close Confirmation', 
                                        "A program is currently running. Quitting now will cause it to stop at its current step, you will be able to start from here again if you wish to continue later. Are you sure you want to quit?", 
                                        QMessageBox.Yes | QMessageBox.No, 
                                        QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.run_program()
                event.accept()  # Accept the event to close the window
            else:
                event.ignore()  # Ignore the event to prevent the window from closing
    
class LoginWindow(QtWidgets.QMainWindow, Ui_LoginWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
        #self.comboBox_preset = SearchableComboBox(self)
        #self.gridLayout.addWidget(self.comboBox_preset, 5, 1, 1, 2)
        
        self.button_launch_ui.setText('Launch UI')
        self.button_launch_ui.clicked.connect(self.run_uiScript)
        
        # Get the directory of this file
        self.script_dir = os.path.abspath(os.path.dirname(__file__))  
    
        # Populate inputs based on the preset you selected
        for file in os.listdir(os.path.join(self.script_dir, "automation_presets")):
            if ".config" in file:
                file = file[:-7]
                self.comboBox.insertItem(self.findAlphabeticalIndex(self.comboBox, file), file)
        
        if self.comboBox.count() < 1:
            self.comboBox.setEditable(False)
            self.comboBox.setPlaceholderText('-- No Presets --')
            #self.comboBox.setItemText(0, )
            self.comboBox.setStyleSheet("background-color: rgb(137, 137, 137)")
            
        self.comboBox.setCurrentIndex(-1)
        
    def findAlphabeticalIndex(self, combo, item):
        # Used to sort combobox alphabetically
        combo_list = [combo.itemText(i) for i in range(combo.count())]
        combo_list.append(item)
        combo_list.sort(key=lambda i: i.upper())
        return combo_list.index(item)
    
    def run_uiScript(self):
        # Starts up main UI screen
        if self.comboBox.currentText().strip() == '' or self.comboBox.findText(self.comboBox.currentText()) != -1:
            self.new_ui = Window()
            self.new_ui.show() 
            if self.comboBox.currentText() != '':
                self.new_ui.comboBox_preset.setCurrentIndex(self.new_ui.comboBox_preset.findText(self.comboBox.currentText().strip()))
                self.new_ui.populate_inputs()
            self.close()
         
    def set_background_image(self, image_path):
        self.setStyleSheet(f"QMainWindow {{background-image: url({image_path}); background-repeat: no-repeat; background-position: center;}}")

def main(): 
    
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Windows')
    # app.setStyleSheet(PyQt5_stylesheets.load_stylesheet_pyqt5(style="style_Dark"))
    ui = LoginWindow()
    ui.show()
    
    sys.exit(app.exec_())
    
if __name__ == "__main__":
    main()