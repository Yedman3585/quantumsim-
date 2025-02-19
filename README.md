# quantumsim-
quantum concepts simulation 

Here is the python based program which simulates various different concepts of quantum physics through execution of definite commands. 
It allows users to simulate quantum computations using a set of predefined quantum gates and operations. The simulator reads quantum circuits from a file or generates random circuits, executes them, and provides output in the form of probability distributions and state vectors. It also features visualization capabilities through Matplotlib and Tkinter.

Custom Quantum Circuit Execution: Reads quantum circuits from a file and simulates the operations.

Random Quantum Circuit Generation: Generates a random test circuit with quantum gates.

Gate Operations:

Hadamard (h)

Pauli Gates (x, y, z)

Phase Gates (s, sdg)

T Gates (t, tdg)

Controlled Gates (cx, csk)

Quantum Fourier Transform (QFT, IQFT)

Measurement (measure)

State Vector Initialization: Allows different initialization methods, including predefined and random superposition states.

Simulation Output:

Final state vector representation.

Measurement probabilities for basis states.

Probability threshold filtering.

Graphical Visualization:

Plots probability distributions of quantum states.

Uses Tkinter to display interactive graphs.

Requirements

To run the simulator, you need the following Python libraries:

numpy

matplotlib

tkinter
