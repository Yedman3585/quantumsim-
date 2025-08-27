![QuantumSimulator Banner](assets/v3epY.png)


[//]: # (![QuantumSimulator Banner]&#40;assets.png&#41;)

<br />
<div align="center">
 
  <h1 align="center">Quantum Simulator</h1>

  <p align="center">
    Simulation of foundational concepts of quantum physics on Python 
</p>
</div>

## About The Project
> **Note** :  The picture above represents quantum circuit diagram with 8 qubits (Hadamard Gate at a0 and CNOT Gates on all others ) 

Quantum Simulator, in the given interpretation,
is a program that executes text-based sequences of commands—our custom 
“instructions”—parsed line by line; each instruction names a gate or operation 
(optionally with a qubit range or parameter), and the interpreter applies 
them in order to an n-qubit state vector using little-endian indexing.

# Features
A lightweight, didactic quantum **state-vector** simulator written in pure Python/NumPy.  
It prioritizes clarity and paper-friendly reproducibility: every amplitude move is explicit, 
range semantics are preserved, and printing/plotting make intermediate states easy to inspect.

- **Single-qubit gates:** `X, Y, Z, S, S† (sdg), T, T† (tdg), S_k (sk)`
- **Hadamard:** `h` (per-qubit) and `hn` (`H^{⊗n}` fast sweep)
- **Two-qubit gates:** `cx` (CNOT), `csk` (controlled phase with angle π/k)
- **Utilities:** `reverse` (bit-reversal / endianness swap), `Sign` (selective phase inversion)
- **QFT blocks:** `QFT`, `IQFT` over a contiguous qubit range
- **Measurement:** mark qubits to measure; post-processing prints marginals and conditional states
- **Visualization:** static Matplotlib figure (`plot.png`) and an optional Tkinter window

# Example 
console output corresponds to Grover’s search on 5 qubits (so 𝑁=2^5=32) with the marked state
index 21 (which prints as |10101> in the simulator’s little-endian display).
1) Hadamard gate geta applied to qubits from 0 to 4
Creates the uniform superposition ∣𝑠⟩=𝐻⊗5 ∣00000⟩
2) Then this pattern repeats several times:
Sign flip on index 21 — oracle 𝑂: multiply only ∣10101⟩ by −1
Hadamard gate applied ... → Sign flip on index 0 → Hadamard gate applied ...
So each Grover iteration is O followed by D.
Your log shows this exactly four times, which is optimal for N=32:
3) measure 0..4
Marks all five qubits to be measured, then the post-processing prints the marginal probabilities and
conditional states.

![GroverExample](graph/graphick.png)

Initial state:
[1.+0.j]
plot = 1

Number of qubits: 5
Initial state: |psi> = 
(1.000+0.000j)|00000> 
Hadamard gate applied to qubits from 0 to 4
Sign flip on index 21
Hadamard gate applied to qubits from 0 to 4
Sign flip on index 0
Hadamard gate applied to qubits from 0 to 4
Sign flip on index 21
Hadamard gate applied to qubits from 0 to 4
Sign flip on index 0
Hadamard gate applied to qubits from 0 to 4
Sign flip on index 21
Hadamard gate applied to qubits from 0 to 4
Sign flip on index 0
Hadamard gate applied to qubits from 0 to 4
Sign flip on index 21
Hadamard gate applied to qubits from 0 to 4
Sign flip on index 0
Hadamard gate applied to qubits from 0 to 4
Measure qubit 0
Measure qubit 1
Measure qubit 2
Measure qubit 3
Measure qubit 4

Final basis states with P > 0.0
P(|00000>) = 2.64e-05	 Amplitude: -5.14e-03+0.00e+00j
P(|10000>) = 2.64e-05	 Amplitude: -5.14e-03+0.00e+00j
P(|01000>) = 2.64e-05	 Amplitude: -5.14e-03+0.00e+00j
P(|11000>) = 2.64e-05	 Amplitude: -5.14e-03+0.00e+00j
P(|00100>) = 2.64e-05	 Amplitude: -5.14e-03+0.00e+00j
P(|10100>) = 2.64e-05	 Amplitude: -5.14e-03+0.00e+00j
P(|01100>) = 2.64e-05	 Amplitude: -5.14e-03+0.00e+00j
P(|11100>) = 2.64e-05	 Amplitude: -5.14e-03+0.00e+00j
P(|00010>) = 2.64e-05	 Amplitude: -5.14e-03+0.00e+00j
P(|10010>) = 2.64e-05	 Amplitude: -5.14e-03+0.00e+00j
P(|01010>) = 2.64e-05	 Amplitude: -5.14e-03+0.00e+00j
P(|11010>) = 2.64e-05	 Amplitude: -5.14e-03+0.00e+00j
P(|00110>) = 2.64e-05	 Amplitude: -5.14e-03+0.00e+00j
P(|10110>) = 2.64e-05	 Amplitude: -5.14e-03+0.00e+00j
P(|01110>) = 2.64e-05	 Amplitude: -5.14e-03+0.00e+00j
P(|11110>) = 2.64e-05	 Amplitude: -5.14e-03+0.00e+00j
P(|00001>) = 2.64e-05	 Amplitude: -5.14e-03+0.00e+00j
P(|10001>) = 2.64e-05	 Amplitude: -5.14e-03+0.00e+00j
P(|01001>) = 2.64e-05	 Amplitude: -5.14e-03+0.00e+00j
P(|11001>) = 2.64e-05	 Amplitude: -5.14e-03+0.00e+00j
P(|00101>) = 2.64e-05	 Amplitude: -5.14e-03+0.00e+00j
P(|10101>) = 9.99e-01	 Amplitude: 1.00e+00+0.00e+00j
P(|01101>) = 2.64e-05	 Amplitude: -5.14e-03+0.00e+00j
P(|11101>) = 2.64e-05	 Amplitude: -5.14e-03+0.00e+00j
P(|00011>) = 2.64e-05	 Amplitude: -5.14e-03+0.00e+00j
P(|10011>) = 2.64e-05	 Amplitude: -5.14e-03+0.00e+00j
P(|01011>) = 2.64e-05	 Amplitude: -5.14e-03+0.00e+00j
P(|11011>) = 2.64e-05	 Amplitude: -5.14e-03+0.00e+00j
P(|00111>) = 2.64e-05	 Amplitude: -5.14e-03+0.00e+00j
P(|10111>) = 2.64e-05	 Amplitude: -5.14e-03+0.00e+00j
P(|01111>) = 2.64e-05	 Amplitude: -5.14e-03+0.00e+00j
P(|11111>) = 2.64e-05	 Amplitude: -5.14e-03+0.00e+00j

Probabilities for measurements of qubits: 
0 
1 
2 
3 
4 

P(00000) = 
2.6376917958259447e-05
|psi> = 
(-1.000+0.000j)|00000>

P(10000) = 
2.6376917958258932e-05
|psi> = 
(-1.000+0.000j)|10000>

P(01000) = 
2.6376917958259315e-05
|psi> = 
(-1.000+0.000j)|01000>

P(11000) = 
2.637691795825886e-05
|psi> = 
(-1.000+0.000j)|11000>

P(00100) = 
2.6376917958259447e-05
|psi> = 
(-1.000+0.000j)|00100>

P(10100) = 
2.6376917958258796e-05
|psi> = 
(-1.000+0.000j)|10100>

P(01100) = 
2.6376917958259315e-05
|psi> = 
(-1.000+0.000j)|01100>

P(11100) = 
2.637691795825886e-05
|psi> = 
(-1.000+0.000j)|11100>

P(00010) = 
2.6376917958259447e-05
|psi> = 
(-1.000+0.000j)|00010>

P(10010) = 
2.6376917958258932e-05
|psi> = 
(-1.000+0.000j)|10010>

P(01010) = 
2.6376917958259315e-05
|psi> = 
(-1.000+0.000j)|01010>

P(11010) = 
2.637691795825886e-05
|psi> = 
(-1.000+0.000j)|11010>

P(00110) = 
2.6376917958259447e-05
|psi> = 
(-1.000+0.000j)|00110>

P(10110) = 
2.637691795825877e-05
|psi> = 
(-1.000+0.000j)|10110>

P(01110) = 
2.6376917958259315e-05
|psi> = 
(-1.000+0.000j)|01110>

P(11110) = 
2.637691795825886e-05
|psi> = 
(-1.000+0.000j)|11110>

P(00001) = 
2.6376917958259305e-05
|psi> = 
(-1.000+0.000j)|00001>

P(10001) = 
2.6376917958258932e-05
|psi> = 
(-1.000+0.000j)|10001>

P(01001) = 
2.6376917958259457e-05
|psi> = 
(-1.000+0.000j)|01001>

P(11001) = 
2.6376917958259874e-05
|psi> = 
(-1.000+0.000j)|11001>

P(00101) = 
2.6376917958261016e-05
|psi> = 
(-1.000+0.000j)|00101>

P(10101) = 
0.9991823155432863
|psi> = 
(1.000+0.000j)|10101>

P(01101) = 
2.6376917958260026e-05
|psi> = 
(-1.000+0.000j)|01101>

P(11101) = 
2.6376917958259874e-05
|psi> = 
(-1.000+0.000j)|11101>

P(00011) = 
2.6376917958259305e-05
|psi> = 
(-1.000+0.000j)|00011>

P(10011) = 
2.6376917958258932e-05
|psi> = 
(-1.000+0.000j)|10011>

P(01011) = 
2.6376917958259457e-05
|psi> = 
(-1.000+0.000j)|01011>

P(11011) = 
2.6376917958258664e-05
|psi> = 
(-1.000+0.000j)|11011>

P(00111) = 
2.6376917958259305e-05
|psi> = 
(-1.000+0.000j)|00111>

P(10111) = 
2.637691795825877e-05
|psi> = 
(-1.000+0.000j)|10111>

P(01111) = 
2.6376917958259457e-05
|psi> = 
(-1.000+0.000j)|01111>

P(11111) = 
2.6376917958258664e-05
|psi> = 
(-1.000+0.000j)|11111>
Plotting the graph...
Graph saved as plot.png!
Opening Tkinter window...
Graph successfully displayed in Tkinter!
Closing Tkinter window...
Finished

Process finished with exit code 0



# History 
This quantum simulator began as a student research project for the Scientific-Research Work of Students (НИРС).  
The competition’s documentation template emphasized formal reporting more than reproducible code, so the original
deliverables were not ideal for sharing or extending. 😺😺😺

> **NOTE** : The complete technical explanation of project and some theoretical foundations in pdf file attached.