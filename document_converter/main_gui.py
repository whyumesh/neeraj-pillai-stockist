"""
Main GUI interface for document classification and conversion
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import threading
import logging

from .document_processor import DocumentProcessor
from .file_scanner import FileScanner
from .config_loader import ConfigLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentConverterGUI:
    """GUI application for document conversion"""
    
    def __init__(self):
        """Initialize GUI"""
        self.root = tk.Tk()
        self.root.title("Document Classification & Conversion System")
        self.root.geometry("1000x700")
        
        self.config = ConfigLoader()
        self.processor = None
        self.files = []
        self.processing = False
        
        self.setup_ui()
        
        # Set default folder if EmailAttachments exists
        default_folder = Path("EmailAttachments")
        if default_folder.exists():
            self.folder_var.set(str(default_folder))
    
    def setup_ui(self):
        """Create user interface"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Folder selection
        folder_frame = ttk.LabelFrame(main_frame, text="Input Folder", padding="10")
        folder_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        folder_frame.columnconfigure(1, weight=1)
        
        ttk.Label(folder_frame, text="EmailAttachments Folder:").grid(row=0, column=0, padx=5, sticky=tk.W)
        
        self.folder_var = tk.StringVar()
        folder_entry = ttk.Entry(folder_frame, textvariable=self.folder_var, width=50)
        folder_entry.grid(row=0, column=1, padx=5, sticky=(tk.W, tk.E))
        
        ttk.Button(folder_frame, text="Browse...", command=self.browse_folder).grid(row=0, column=2, padx=5)
        ttk.Button(folder_frame, text="Scan Files", command=self.scan_files).grid(row=0, column=3, padx=5)
        
        # File list
        list_frame = ttk.LabelFrame(main_frame, text="Files to Process", padding="10")
        list_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        
        # Treeview for files
        columns = ("Filename", "Type", "Classification", "Status")
        self.files_tree = ttk.Treeview(list_frame, columns=columns, show="tree headings", height=15)
        self.files_tree.heading("#0", text="")
        self.files_tree.column("#0", width=30)
        
        for col in columns:
            self.files_tree.heading(col, text=col)
            self.files_tree.column(col, width=200)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.files_tree.yview)
        self.files_tree.configure(yscrollcommand=scrollbar.set)
        
        self.files_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        self.process_button = ttk.Button(button_frame, text="Process All Files", 
                                         command=self.process_all, state=tk.DISABLED)
        self.process_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Clear List", command=self.clear_list).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="View Log", command=self.view_log).pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        progress_frame = ttk.Frame(main_frame)
        progress_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        progress_frame.columnconfigure(0, weight=1)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=5)
        
        self.status_label = ttk.Label(progress_frame, text="Ready")
        self.status_label.grid(row=1, column=0, pady=5)
    
    def browse_folder(self):
        """Browse for input folder"""
        folder = filedialog.askdirectory(title="Select EmailAttachments Folder")
        if folder:
            self.folder_var.set(folder)
    
    def scan_files(self):
        """Scan files in selected folder"""
        folder_path = self.folder_var.get()
        if not folder_path:
            messagebox.showwarning("No Folder", "Please select a folder first")
            return
        
        folder = Path(folder_path)
        if not folder.exists():
            messagebox.showerror("Error", f"Folder does not exist: {folder_path}")
            return
        
        self.status_label.config(text="Scanning files...")
        self.root.update()
        
        try:
            scanner = FileScanner(folder)
            self.files = scanner.scan_files()
            
            # Clear existing items
            for item in self.files_tree.get_children():
                self.files_tree.delete(item)
            
            # Add files to tree
            for file_info in self.files:
                self.files_tree.insert("", tk.END, values=(
                    file_info['filename'],
                    file_info['file_type'],
                    "Not classified",
                    "Pending"
                ))
            
            self.process_button.config(state=tk.NORMAL if self.files else tk.DISABLED)
            self.status_label.config(text=f"Found {len(self.files)} files")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error scanning files: {e}")
            self.status_label.config(text="Error scanning files")
    
    def process_all(self):
        """Process all files"""
        if self.processing:
            return
        
        if not self.files:
            messagebox.showwarning("No Files", "Please scan files first")
            return
        
        # Run in separate thread to avoid blocking UI
        thread = threading.Thread(target=self._process_files_thread)
        thread.daemon = True
        thread.start()
    
    def _process_files_thread(self):
        """Process files in background thread"""
        self.processing = True
        self.process_button.config(state=tk.DISABLED)
        
        folder_path = Path(self.folder_var.get())
        
        try:
            self.processor = DocumentProcessor(self.config)
            
            total = len(self.files)
            for i, file_info in enumerate(self.files):
                # Update progress
                progress = (i / total) * 100
                self.progress_var.set(progress)
                self.status_label.config(text=f"Processing {i+1}/{total}: {file_info['filename']}")
                self.root.update()
                
                # Process file
                result = self.processor.process_file(file_info)
                
                # Update treeview
                classification = result.get('classification', 'unknown')
                status = result.get('status', 'error')
                
                # Find and update the item in tree
                for item_id in self.files_tree.get_children():
                    values = self.files_tree.item(item_id, 'values')
                    if values[0] == file_info['filename']:
                        self.files_tree.item(item_id, values=(
                            values[0],
                            values[1],
                            classification,
                            status.capitalize()
                        ))
                        break
            
            self.progress_var.set(100)
            self.status_label.config(text=f"Processing complete! Processed {total} files.")
            
            # Save logs
            self.processor.processing_logger.save_logs()
            
            # Show statistics
            stats = self.processor.processing_logger.get_statistics()
            messagebox.showinfo("Complete", 
                              f"Processing complete!\n\n"
                              f"Total: {stats['total_files']}\n"
                              f"Successful: {stats['successful']}\n"
                              f"Errors: {stats['errors']}\n"
                              f"Purchase Orders: {stats['purchase_orders']}\n"
                              f"Stock & Sales: {stats['stock_sales_reports']}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Error during processing: {e}")
            self.status_label.config(text="Error during processing")
        
        finally:
            self.processing = False
            self.process_button.config(state=tk.NORMAL)
    
    def clear_list(self):
        """Clear file list"""
        for item in self.files_tree.get_children():
            self.files_tree.delete(item)
        self.files = []
        self.process_button.config(state=tk.DISABLED)
    
    def view_log(self):
        """View processing log"""
        log_file = Path(self.config.get('paths.log_file', 'Output/ProcessingLog.json'))
        if log_file.exists():
            import os
            os.startfile(log_file)
        else:
            messagebox.showinfo("Log", "No log file found yet. Process some files first.")
    
    def run(self):
        """Start GUI application"""
        self.root.mainloop()


def main():
    """Main entry point"""
    app = DocumentConverterGUI()
    app.run()


if __name__ == '__main__':
    main()

