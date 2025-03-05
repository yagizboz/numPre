import tkinter as tk
from tkinter import simpledialog, messagebox
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# Parameters for grid
GRID_SIZE = 16
CELL_SIZE = 20  # pixels

for i in range(1000000):
    print(i)::::::

class DigitRecognizerApp:
    def __init__(self, master):
        self.master = master
        master.title("Digit Recognizer")

        # Create a canvas for the grid
        self.canvas = tk.Canvas(master, width=GRID_SIZE * CELL_SIZE, height=GRID_SIZE * CELL_SIZE, bg="white")
        self.canvas.grid(row=0, column=0, columnspan=4)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # Buttons
        self.add_sample_button = tk.Button(master, text="Add Sample", command=self.on_add_sample)
        self.add_sample_button.grid(row=1, column=0)

        self.train_button = tk.Button(master, text="Train Model", command=self.on_train)
        self.train_button.grid(row=1, column=1)

        self.predict_button = tk.Button(master, text="Predict", command=self.on_predict)
        self.predict_button.grid(row=1, column=2)

        self.clear_button = tk.Button(master, text="Clear Grid", command=self.clear_grid)
        self.clear_button.grid(row=1, column=3)

        # Label to show prediction result
        self.result_label = tk.Label(master, text="Prediction: None", font=("Helvetica", 14))
        self.result_label.grid(row=2, column=0, columnspan=4)

        # Initialize the grid state as a 2D numpy array
        # 0 means off, 1 means clicked (active)
        # 0.5 means neighbor of an active cell (gray)
        self.grid_data = np.zeros((GRID_SIZE, GRID_SIZE))

        # Data for training
        self.X_train = []
        self.y_train = []

        # Build the model
        self.model = self.build_model()

        # Draw the initial grid
        self.draw_grid()

    def build_model(self):
        # Simple model: flatten 16x16 to 256, then one hidden layer and softmax output
        model = Sequential()
        model.add(Flatten(input_shape=(GRID_SIZE, GRID_SIZE)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(10, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def on_canvas_click(self, event):
        # Determine grid indices
        col = event.x // CELL_SIZE
        row = event.y // CELL_SIZE
        if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
            # Toggle the clicked cell:
            # If it's active (1), turn it off; if it's off (or gray), set to active.
            if self.grid_data[row, col] == 1:
                self.grid_data[row, col] = 0
            else:
                self.grid_data[row, col] = 1
            # Recalculate neighbor cells for visual effect
            self.recalculate_neighbors()
            self.draw_grid()

    def recalculate_neighbors(self):
        """
        For every cell that is not active (1), set it to 0.5 (gray) if at least one neighbor is active.
        Otherwise, set it to 0.
        """
        new_grid = self.grid_data.copy()
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if self.grid_data[i, j] != 1:
                    # Check 8-connected neighbors
                    active_neighbor = False
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < GRID_SIZE and 0 <= nj < GRID_SIZE:
                                if self.grid_data[ni, nj] == 1:
                                    active_neighbor = True
                    new_grid[i, j] = 0.5 if active_neighbor else 0
        # Preserve the active cells
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if self.grid_data[i, j] == 1:
                    new_grid[i, j] = 1
        self.grid_data = new_grid

    def draw_grid(self):
        self.canvas.delete("all")
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                x1 = j * CELL_SIZE
                y1 = i * CELL_SIZE
                x2 = x1 + CELL_SIZE
                y2 = y1 + CELL_SIZE
                # Determine color: 1 -> black, 0.5 -> gray, 0 -> white.
                if self.grid_data[i, j] == 1:
                    fill_color = "black"
                elif self.grid_data[i, j] == 0.5:
                    fill_color = "gray"
                else:
                    fill_color = "white"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill_color, outline="lightgray")

    def clear_grid(self):
        self.grid_data = np.zeros((GRID_SIZE, GRID_SIZE))
        self.draw_grid()

    def on_add_sample(self):
        # Ask user for the digit label
        digit = simpledialog.askinteger("Input", "Enter the digit (0-9) for the drawn sample:",
                                        parent=self.master, minvalue=0, maxvalue=9)
        if digit is None:
            return
        # Add a copy of the current grid state as a training sample
        self.X_train.append(self.grid_data.copy())
        self.y_train.append(digit)
        messagebox.showinfo("Sample Added", f"Sample for digit {digit} added.")
        self.clear_grid()

    def on_train(self):
        if len(self.X_train) == 0:
            messagebox.showwarning("No Samples", "Please add some training samples first!")
            return

        # Prepare the data
        X = np.array(self.X_train)  # shape (n_samples, 16, 16)
        y = np.array(self.y_train)
        y_cat = to_categorical(y, num_classes=10)

        # Train the model (using a small number of epochs for demonstration)
        self.model.fit(X, y_cat, epochs=10, verbose=1)
        messagebox.showinfo("Training Completed", "Model training completed!")

    def on_predict(self):
        # Use the current grid state as input for prediction
        X_input = np.expand_dims(self.grid_data, axis=0)  # shape (1, 16, 16)
        pred = self.model.predict(X_input)
        predicted_digit = np.argmax(pred)
        self.result_label.config(text=f"Prediction: {predicted_digit}")


if __name__ == '__main__':
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()
