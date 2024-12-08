import tkinter as tk
from proctor_gui import ProctorGUI

def main():
    try:
        root = tk.Tk()
        app = ProctorGUI(root)
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        root.mainloop()
    except Exception as e:
        print(f"Error starting application: {e}")

if __name__ == "__main__":
    main() 