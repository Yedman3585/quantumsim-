from __future__ import print_function
import numpy as np
import sys
import string
import random
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog,simpledialog

from openai import OpenAI


def choose_option():
    root = tk.Tk()
    root.withdraw()

    option = simpledialog.askstring(
        "choose mode ",
        "1 to choose file,\n"
        "2 to random generating"

    )

    if option == "1":
        return choose_file()
    elif option == "2":
        return generate_random_test()

    else:
        print("Некорректный ввод! Выбран режим загрузки файла.")
        return choose_file()

def choose_file():

    file_path = filedialog.askopenfilename(
        title="choose file ",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
    )
    return file_path


def generate_random_test():
    num_qubits = random.randint(2, 5)
    num_commands = random.randint(3, 7)

    commands = [
        "h q[{}];",
        "x q[{}];",
        "y q[{}];",
        "z q[{}];",
        "cx q[{}], q[{}];",
        "measure q[{}];"
    ]

    test_filename = "random_test.txt"

    with open(test_filename, "w") as f:
        f.write("verbose 1;\n")
        f.write(f"init q[0:{num_qubits-1}];\n")  # Инициализация всех кубитов

        for _ in range(num_commands):
            command = random.choice(commands)
            if "cx" in command:
                q1, q2 = random.sample(range(num_qubits), 2)
                f.write(command.format(q1, q2) + "\n")
            else:
                q = random.randint(0, num_qubits - 1)
                f.write(command.format(q) + "\n")

        f.write(f"measure q[0:{num_qubits-1}];\n")  # Добавляем измерение всех кубитов

    print(f"✅ random test generated : {test_filename}")
    return test_filename

input_file = choose_option()

if not input_file:
    input_file = "example_scenario.txt"

try:
    with open(input_file, "r", encoding="utf-8") as file:
        command_list = [line.strip() for line in file]
except UnicodeDecodeError:
    print(f"Encoding error in '{input_file}'. Ensure it's UTF-8 encoded.")
    sys.exit(1)
except FileNotFoundError:
    print(f"File '{input_file}' not found.")
    sys.exit(1)


num_qubits = 0
initial_state = -1
qubit_start = -1
qubit_end = -1
verbose_mode = 0
plot_enabled = 0
print_enabled = 1
probability_threshold = 0.0

state_vector = np.zeros(2 ** num_qubits, dtype=np.complex128)
measurement_array = np.zeros(num_qubits, dtype=np.complex128)

if initial_state == -1:
    state_vector[0] = 1
elif initial_state == -2:  # Initialize to random superposition
    for k in range(2 ** num_qubits):
        if qubit_start <= k <= qubit_end:
            state_vector[k] = random.uniform(-1, 1) + 1j * random.uniform(-1, 1)

if initial_state != -1:
    state_vector /= np.sqrt(np.sum(np.abs(state_vector) ** 2))

print('Initial state:')
print(state_vector)

def extract_qubits(command):
    before, sep, after = command.rpartition(";")
    gate = before.split()[0]
    if gate not in ['cx', 'sk', 'csk', 'N&m']:
        before1, sep1, after1 = before.rpartition(":")
        if sep1 == ':':
            digits = [int(s) for s in before1 if s.isdigit()]
            qubit_start = digits[0] if len(digits) == 1 else 10 * digits[0] + digits[1]
            digits = [int(s) for s in after1 if s.isdigit()]
            qubit_end = digits[0] if len(digits) == 1 else 10 * digits[0] + digits[1]
        else:
            digits = [int(s) for s in before if s.isdigit()]
            qubit_start = sum(d * 10 ** (len(digits) - i - 1) for i, d in enumerate(digits))
            qubit_end = qubit_start
        control_start = qubit_start
        control_end = qubit_end
        target_start = -1
        target_end = -1
    elif gate == 'sk':
        before2, sep2, after2 = before.rpartition(",")
        before1, sep1, after1 = before2.rpartition(":")
        if sep1 == ':':
            digits = [int(s) for s in before1 if s.isdigit()]
            qubit_start = digits[0] if len(digits) == 1 else 10 * digits[0] + digits[1]
            digits = [int(s) for s in after1 if s.isdigit()]
            qubit_end = digits[0] if len(digits) == 1 else 10 * digits[0] + digits[1]
        else:
            digits = [int(s) for s in before2 if s.isdigit()]
            qubit_start = digits[0] if len(digits) == 1 else 10 * digits[0] + digits[1]
            qubit_end = qubit_start
        digits = [int(s) for s in after2 if s.isdigit()]
        k = sum(d * 10 ** (len(digits) - i - 1) for i, d in enumerate(digits))
        control_start = qubit_start
        control_end = qubit_end
        target_start = k
        target_end = -1
    elif gate == 'cx':
        before2, sep2, after2 = before.rpartition(",")
        before1, sep1, after1 = before2.rpartition(":")
        if sep1 == ':':
            digits = [int(s) for s in before1 if s.isdigit()]
            control_start = digits[0] if len(digits) == 1 else 10 * digits[0] + digits[1]
            digits = [int(s) for s in after1 if s.isdigit()]
            control_end = digits[0] if len(digits) == 1 else 10 * digits[0] + digits[1]
        else:
            digits = [int(s) for s in before2 if s.isdigit()]
            control_start = digits[0] if len(digits) == 1 else 10 * digits[0] + digits[1]
            control_end = control_start
        before1, sep1, after1 = after2.rpartition(":")
        if sep1 == ':':
            digits = [int(s) for s in before1 if s.isdigit()]
            target_start = digits[0] if len(digits) == 1 else 10 * digits[0] + digits[1]
            digits = [int(s) for s in after1 if s.isdigit()]
            target_end = digits[0] if len(digits) == 1 else 10 * digits[0] + digits[1]
        else:
            digits = [int(s) for s in after2 if s.isdigit()]
            target_start = digits[0] if len(digits) == 1 else 10 * digits[0] + digits[1]
            target_end = target_start
    elif gate == 'csk':
        before1, sep1, after1 = before.rpartition(":")
        if sep1 == ':':
            sys.exit('The csk gate does not allow expansion of range of qubits')
        before2, sep2, after2 = before.rpartition(",")
        before3, sep3, after3 = before2.rpartition(",")
        digits = [int(s) for s in before3 if s.isdigit()]
        control_qubit = digits[0] if len(digits) == 1 else 10 * digits[0] + digits[1]
        digits = [int(s) for s in after3 if s.isdigit()]
        target_qubit = digits[0] if len(digits) == 1 else 10 * digits[0] + digits[1]
        digits = [int(s) for s in after2 if s.isdigit()]
        k = sum(d * 10 ** (len(digits) - i - 1) for i, d in enumerate(digits))
        control_start = control_qubit
        control_end = target_qubit
        target_start = k
        target_end = -1
    elif gate == 'N&m':
        before1, sep1, after1 = before.rpartition(":")
        before2, sep2, after2 = before.rpartition(",")
        digits = [int(s) for s in before2 if s.isdigit()]
        N = sum(d * 10 ** (len(digits) - i - 1) for i, d in enumerate(digits))
        digits = [int(s) for s in after2 if s.isdigit()]
        m = sum(d * 10 ** (len(digits) - i - 1) for i, d in enumerate(digits))
        control_start = N
        control_end = m
        target_start = -1
        target_end = -1
    return control_start, control_end, target_start, target_end

if len(sys.argv) > 1:
    input_file = sys.argv[1]
with open(input_file, "r") as file:
    command_list = [line.strip() for line in file]

for i in range(len(command_list)):
    command = command_list[i]
    before, sep, after = command.rpartition(";")
    if before.split() != []:
        gate = before.split()[0]
    else:
        gate = ''

    if gate in ['id', 'h', 'x', 'y', 'z', 's', 'sdg', 't', 'tdg', 'measure', 'QFT', 'IQFT']:
        qubit_start, qubit_end, _, _ = extract_qubits(command)
        num_qubits = max(num_qubits, qubit_start + 1)
        num_qubits = max(num_qubits, qubit_end + 1)
    elif gate in ['init', 'verbose', 'plot', 'printout', 'Inverse_P_threshold', 'N&m']:
        qubit_start, qubit_end, _, _ = extract_qubits(command)
        if gate == 'init':
            initial_state = -2
            qubit_start = qubit_start
            qubit_end = qubit_end
        elif gate == 'verbose':
            verbose_mode = qubit_start
        elif gate == 'plot':
            plot_enabled = qubit_start
            print('plot =', plot_enabled)
        elif gate == 'printout':
            print_enabled = qubit_start
            print('printout =', print_enabled)
        elif gate == 'Inverse_P_threshold':
            if qubit_start > 0: probability_threshold = float(1.0 / qubit_start)
            print('Inverse_P_threshold =', qubit_start)
            print('P_threshold =', probability_threshold)
        elif gate == 'N&m':
            initial_state = -3
            N = float(qubit_start)
            m = float(qubit_end)
    elif gate == 'sk':
        qubit_start, qubit_end, k, _ = extract_qubits(command)
        num_qubits = max(num_qubits, qubit_end + 1)
        num_qubits = max(num_qubits, qubit_start + 1)
    elif gate == 'cx':
        control_start, control_end, target_start, target_end = extract_qubits(command)
        num_qubits = max(num_qubits, control_start + 1)
        num_qubits = max(num_qubits, control_end + 1)
        num_qubits = max(num_qubits, target_start + 1)
        num_qubits = max(num_qubits, target_end + 1)
    elif gate == 'csk':
        control_qubit, target_qubit, k, _ = extract_qubits(command)
        num_qubits = max(num_qubits, control_qubit + 1)
        num_qubits = max(num_qubits, target_qubit + 1)

def set_bit(value, bit_index):

    return value | (1 << bit_index)

def clear_bit(value, bit_index):

    return value & ~(1 << bit_index)

def print_state(gate, num_qubits, verbose_mode, state):
    if gate not in ['cx', 'sk', 'csk', 'Sign', 'QFT', 'IQFT', 'h']:
        print(f'Gate {gate} on qubit {qubit}'),
    if verbose_mode == 1:
        print('  resulted in state |psi> = '),
        k1 = 0
        psi = ''
        for k in range(2 ** num_qubits):
            binary_str = ("{:0%db}" % num_qubits).format(k)[::-1]
            if state[k] != 0:
                k1 += 1
                if k1 == 1:
                    psi += f'({state[k]:.3f})|{binary_str}> '
                else:
                    psi += f'+ ({state[k]:.3f})|{binary_str}> '
        psi = psi.replace('+ -', '- ')
        print(psi)
        print()
    return state, state

def dft_j(type, N, j):
    dft_coeff = np.zeros(N, dtype=np.complex_)
    for k in range(N):
        dft_coeff[k] = np.exp(type * 2 * np.pi * 1j * j * k / N)
    return dft_coeff / np.sqrt(N)

def dft(num_qubits, qubit_start, qubit_end, type, state):
    result_state = np.zeros(2 ** num_qubits, dtype=np.complex128)
    N1 = 2 ** qubit_start  # Qubits below QFT
    N2 = 2 ** (qubit_end - qubit_start + 1)  # Qubits at QFT
    N3 = 2 ** (num_qubits - qubit_end - 1)  # Qubits above QFT
    for j3 in range(N3):
        for j2 in range(N2):
            for j1 in range(N1):
                j = (j3 << qubit_end + 1) + (j2 << qubit_start) + j1
                if np.absolute(state[j]) > 0:
                    dft_coeff = dft_j(type, N2, j2)
                    for jj in range(len(dft_coeff)):
                        j4 = (j3 << qubit_end + 1) + (jj << qubit_start) + j1
                        result_state[j4] += dft_coeff[jj] * state[j]
    return result_state

def identity_gate(num_qubits, qubit, state):
    return state

def hadamard_gate(num_qubits, qubit, state):
    print(f'Hadamard gate applied to qubit {qubit}')
    result_state = np.zeros(2 ** num_qubits, dtype=np.complex128)
    isq2 = 1 / np.sqrt(2)
    for j in range(2 ** num_qubits):
        if state[j] != 0:
            bit_parity = (j >> qubit) % 2
            if bit_parity == 0:
                result_state[j] += isq2 * state[j]
                result_state[set_bit(j, qubit)] += isq2 * state[j]
            elif bit_parity == 1:
                result_state[clear_bit(j, qubit)] += isq2 * state[j]
                result_state[j] += -isq2 * state[j]
    return result_state

def hadamard_n(num_qubits, state):
    print(f'Hadamard gate applied to qubits from 0 to {num_qubits - 1}')
    isq2 = 1 / np.sqrt(2)
    for qubit in range(num_qubits):
        result_state = np.zeros(2 ** num_qubits, dtype=np.complex128)
        for j in range(2 ** num_qubits):
            if state[j] != 0:
                bit_parity = (j >> qubit) % 2
                if bit_parity == 0:
                    result_state[j] += isq2 * state[j]
                    result_state[set_bit(j, qubit)] += isq2 * state[j]
                elif bit_parity == 1:
                    result_state[clear_bit(j, qubit)] += isq2 * state[j]
                    result_state[j] += -isq2 * state[j]
        state = result_state
    return result_state

def pauli_x(num_qubits, qubit, state):
    result_state = np.zeros(2 ** num_qubits, dtype=np.complex128)
    for j in range(2 ** num_qubits):
        if state[j] != 0:
            bit_parity = (j >> qubit) % 2
            if bit_parity == 0:
                result_state[set_bit(j, qubit)] += state[j]
            if bit_parity == 1:
                result_state[clear_bit(j, qubit)] += state[j]
    return result_state

def pauli_y(num_qubits, qubit, state):
    result_state = np.zeros(2 ** num_qubits, dtype=np.complex128)
    for j in range(2 ** num_qubits):
        if state[j] != 0:
            bit_parity = (j >> qubit) % 2
            if bit_parity == 0:
                result_state[set_bit(j, qubit)] += 1j * state[j]
            if bit_parity == 1:
                result_state[clear_bit(j, qubit)] += -1j * state[j]
    return result_state

def pauli_z(num_qubits, qubit, state):
    result_state = np.zeros(2 ** num_qubits, dtype=np.complex128)
    for j in range(2 ** num_qubits):
        if state[j] != 0:
            bit_parity = (j >> qubit) % 2
            if bit_parity == 0:
                result_state[j] += state[j]
            if bit_parity == 1:
                result_state[j] += -state[j]
    return result_state

def phase_gate(num_qubits, qubit, state):
    result_state = np.zeros(2 ** num_qubits, dtype=np.complex128)
    for j in range(2 ** num_qubits):
        if state[j] != 0:
            bit_parity = (j >> qubit) % 2
            if bit_parity == 0:
                result_state[j] += state[j]
            if bit_parity == 1:
                result_state[j] += 1j * state[j]
    return result_state

def phase_dagger_gate(num_qubits, qubit, state):
    result_state = np.zeros(2 ** num_qubits, dtype=np.complex128)
    for j in range(2 ** num_qubits):
        if state[j] != 0:
            bit_parity = (j >> qubit) % 2
            if bit_parity == 0:
                result_state[j] += state[j]
            if bit_parity == 1:
                result_state[j] += -1j * state[j]
    return result_state

def t_gate(num_qubits, qubit, state):
    result_state = np.zeros(2 ** num_qubits, dtype=np.complex128)
    for j in range(2 ** num_qubits):
        if state[j] != 0:
            bit_parity = (j >> qubit) % 2
            if bit_parity == 0:
                result_state[j] += state[j]
            if bit_parity == 1:
                result_state[j] += (1 + 1j) / np.sqrt(2) * state[j]
    return result_state

def t_dagger_gate(num_qubits, qubit, state):
    result_state = np.zeros(2 ** num_qubits, dtype=np.complex128)
    for j in range(2 ** num_qubits):
        if state[j] != 0:
            bit_parity = (j >> qubit) % 2
            if bit_parity == 0:
                result_state[j] += state[j]
            if bit_parity == 1:
                result_state[j] += (1 - 1j) / np.sqrt(2) * state[j]
    return result_state

def sk_gate(num_qubits, qubit, k, state):
    result_state = np.zeros(2 ** num_qubits, dtype=np.complex128)
    phase_factor = np.exp(np.pi * 1j / k)
    for j in range(2 ** num_qubits):
        if state[j] != 0:
            bit_parity = (j >> qubit) % 2
            if bit_parity == 0:
                result_state[j] += state[j]
            if bit_parity == 1:
                result_state[j] += phase_factor * state[j]
    print(f'Gate sk on qubit {qubit} with k = {k}'),
    return result_state

def controlled_x(num_qubits, control_qubit, target_qubit, state):
    result_state = np.zeros(2 ** num_qubits, dtype=np.complex128)
    for j in range(2 ** num_qubits):
        if state[j] != 0:
            control_parity = (j >> control_qubit) % 2
            target_parity = (j >> target_qubit) % 2
            if control_parity == 0:
                result_state[j] += state[j]
            else:
                if target_parity == 0:
                    result_state[set_bit(j, target_qubit)] += state[j]
                else:
                    result_state[clear_bit(j, target_qubit)] += state[j]
    print(f'Gate cx on control qubit {control_qubit} and target qubit {target_qubit}'),
    return result_state

def controlled_sk(num_qubits, control_qubit, target_qubit, k, state):
    result_state = np.zeros(2 ** num_qubits, dtype=np.complex128)
    phase_factor = np.exp(np.pi * 1j / k)
    for j in range(2 ** num_qubits):
        if state[j] != 0:
            control_parity = (j >> control_qubit) % 2
            target_parity = (j >> target_qubit) % 2
            if control_parity == 0:
                result_state[j] += state[j]
            else:
                if target_parity == 0:
                    result_state[j] += state[j]
                if target_parity == 1:
                    result_state[j] += phase_factor * state[j]
    print(f'Gate csk on control qubit {control_qubit} and target qubit {target_qubit} with k = {k}'),
    return result_state

def sign_flip(num_qubits, index, state):
    result_state = state
    result_state[index] = -state[index]
    print(f'Sign flip on index {index}'),
    return result_state

def reverse_state(num_qubits, state):
    result_state = np.zeros(2 ** num_qubits, dtype=np.complex128)
    for j in range(2 ** num_qubits):
        binary_str = ("{:0%db}" % num_qubits).format(j)[::]
        reversed_binary_str = ("{:0%db}" % num_qubits).format(j)[::-1]
        result_state[int(reversed_binary_str, 2)] = state[int(binary_str, 2)]
    return result_state

print(f'\nNumber of qubits: {num_qubits}')
state_vector = np.zeros(2 ** num_qubits, dtype=np.complex128)
measurement_array = np.zeros(num_qubits)

if initial_state == -1:
    state_vector[0] = 1
elif initial_state == -2:
    for k in range(2 ** num_qubits):
        if k >= qubit_start and k <= qubit_end:
            state_vector[k] = random.uniform(-1, 1) + 1j * random.uniform(-1, 1)
elif initial_state == -3:
    for k in range(2 ** num_qubits):
        state_vector[k] = (m ** k) % N

if initial_state != -1:
    norm = np.sqrt(np.sum(np.abs(state_vector) ** 2))
    state_vector /= norm

print('Initial state: |psi> = ')
if initial_state == -1:
    psi = ''
    for k in range(1):
        binary_str = ("{:0%db}" % num_qubits).format(k)[::-1]
        if state_vector[k] != 0:
            psi += f'({state_vector[k]:.3f})|{binary_str}> '
    print(psi)
else:
    psi = ''
    for k in range(2 ** num_qubits):
        binary_str = ("{:0%db}" % num_qubits).format(k)[::-1]
        if state_vector[k] != 0:
            psi += f'({state_vector[k]:.3f})|{binary_str}> '
    print(psi)

for i in range(len(command_list)):
    command = command_list[i]
    before, sep, after = command.rpartition(";")
    if before.split() != []:
        gate = before.split()[0]
    else:
        gate = ''

    if gate in ['id', 'h', 'hn', 'x', 'y', 'z', 's', 'sdg', 't', 'tdg', 'measure']:
        qubit_start, qubit_end, _, _ = extract_qubits(command)

        if gate == 'h':
            if qubit_start == 0 and qubit_end == num_qubits - 1:
                result_state = hadamard_n(num_qubits, state_vector)
                state_vector, _ = print_state(gate, num_qubits, verbose_mode, result_state)
            elif qubit_end >= qubit_start:
                for qubit in range(qubit_start, qubit_end + 1):
                    result_state = hadamard_gate(num_qubits, qubit, state_vector)
                    state_vector, _ = print_state(gate, num_qubits, verbose_mode, result_state)
            elif qubit_end < qubit_start:
                for qubit in range(qubit_start, qubit_end - 1, -1):
                    result_state = hadamard_gate(num_qubits, qubit, state_vector)
                    state_vector, _ = print_state(gate, num_qubits, verbose_mode, result_state)

        if gate == 'x':
            if qubit_end >= qubit_start:
                for qubit in range(qubit_start, qubit_end + 1):
                    result_state = pauli_x(num_qubits, qubit, state_vector)
                    state_vector, _ = print_state(gate, num_qubits, verbose_mode, result_state)
            elif qubit_end < qubit_start:
                for qubit in range(qubit_start, qubit_end - 1, -1):
                    result_state = pauli_x(num_qubits, qubit, state_vector)
                    state_vector, _ = print_state(gate, num_qubits, verbose_mode, result_state)

        if gate == 'y':
            if qubit_end >= qubit_start:
                for qubit in range(qubit_start, qubit_end + 1):
                    result_state = pauli_y(num_qubits, qubit, state_vector)
                    state_vector, _ = print_state(gate, num_qubits, verbose_mode, result_state)
            elif qubit_end < qubit_start:
                for qubit in range(qubit_start, qubit_end - 1, -1):
                    result_state = pauli_y(num_qubits, qubit, state_vector)
                    state_vector, _ = print_state(gate, num_qubits, verbose_mode, result_state)

        if gate == 'z':
            if qubit_end >= qubit_start:
                for qubit in range(qubit_start, qubit_end + 1):
                    result_state = pauli_z(num_qubits, qubit, state_vector)
                    state_vector, _ = print_state(gate, num_qubits, verbose_mode, result_state)
            elif qubit_end < qubit_start:
                for qubit in range(qubit_start, qubit_end - 1, -1):
                    result_state = pauli_z(num_qubits, qubit, state_vector)
                    state_vector, _ = print_state(gate, num_qubits, verbose_mode, result_state)

        if gate == 's':
            if qubit_end >= qubit_start:
                for qubit in range(qubit_start, qubit_end + 1):
                    result_state = phase_gate(num_qubits, qubit, state_vector)
                    state_vector, _ = print_state(gate, num_qubits, verbose_mode, result_state)
            elif qubit_end < qubit_start:
                for qubit in range(qubit_start, qubit_end - 1, -1):
                    result_state = phase_gate(num_qubits, qubit, state_vector)
                    state_vector, _ = print_state(gate, num_qubits, verbose_mode, result_state)

        if gate == 'sdg':
            if qubit_end >= qubit_start:
                for qubit in range(qubit_start, qubit_end + 1):
                    result_state = phase_dagger_gate(num_qubits, qubit, state_vector)
                    state_vector, _ = print_state(gate, num_qubits, verbose_mode, result_state)
            elif qubit_end < qubit_start:
                for qubit in range(qubit_start, qubit_end - 1, -1):
                    result_state = phase_dagger_gate(num_qubits, qubit, state_vector)
                    state_vector, _ = print_state(gate, num_qubits, verbose_mode, result_state)

        if gate == 't':
            if qubit_end >= qubit_start:
                for qubit in range(qubit_start, qubit_end + 1):
                    result_state = t_gate(num_qubits, qubit, state_vector)
                    state_vector, _ = print_state(gate, num_qubits, verbose_mode, result_state)
            elif qubit_end < qubit_start:
                for qubit in range(qubit_start, qubit_end - 1, -1):
                    result_state = t_gate(num_qubits, qubit, state_vector)
                    state_vector, _ = print_state(gate, num_qubits, verbose_mode, result_state)

        if gate == 'tdg':
            if qubit_end >= qubit_start:
                for qubit in range(qubit_start, qubit_end + 1):
                    result_state = t_dagger_gate(num_qubits, qubit, state_vector)
                    state_vector, _ = print_state(gate, num_qubits, verbose_mode, result_state)
            elif qubit_end < qubit_start:
                for qubit in range(qubit_start, qubit_end - 1, -1):
                    result_state = t_dagger_gate(num_qubits, qubit, state_vector)
                    state_vector, _ = print_state(gate, num_qubits, verbose_mode, result_state)

    if gate == 'sk':
        qubit_start, qubit_end, k, _ = extract_qubits(command)
        k_log = int(np.log2(abs(k)))
        if qubit_end >= qubit_start:
            for qubit in range(qubit_start, qubit_end + 1):
                result_state = sk_gate(num_qubits, qubit, k, state_vector)
                state_vector, _ = print_state(gate, num_qubits, verbose_mode, result_state)
        elif qubit_end < qubit_start:
            for qubit in range(qubit_start, qubit_end - 1, -1):
                result_state = sk_gate(num_qubits, qubit, k, state_vector)
                state_vector, _ = print_state(gate, num_qubits, verbose_mode, result_state)

    if gate == 'cx':
        control_start, control_end, target_start, target_end = extract_qubits(command)
        if control_end > control_start and target_start == target_end:
            target_qubit = target_start
            for control_qubit in range(control_start, control_end + 1):
                result_state = controlled_x(num_qubits, control_qubit, target_qubit, state_vector)
                state_vector, _ = print_state(gate, num_qubits, verbose_mode, result_state)
        elif control_end < control_start and target_start == target_end:
            target_qubit = target_start
            for control_qubit in range(control_start, control_end - 1, -1):
                result_state = controlled_x(num_qubits, control_qubit, target_qubit, state_vector)
                state_vector, _ = print_state(gate, num_qubits, verbose_mode, result_state)
        elif control_end == control_start and target_end >= target_start:
            control_qubit = control_start
            for target_qubit in range(target_start, target_end + 1):
                result_state = controlled_x(num_qubits, control_qubit, target_qubit, state_vector)
                state_vector, _ = print_state(gate, num_qubits, verbose_mode, result_state)
        elif control_end == control_start and target_start >= target_end:
            control_qubit = control_start
            for target_qubit in range(target_start, target_end - 1, -1):
                result_state = controlled_x(num_qubits, control_qubit, target_qubit, state_vector)
                state_vector, _ = print_state(gate, num_qubits, verbose_mode, result_state)

    if gate == 'csk':
        control_qubit, target_qubit, k, _ = extract_qubits(command)
        k_log = int(np.log2(abs(k)))
        result_state = controlled_sk(num_qubits, control_qubit, target_qubit, k, state_vector)
        state_vector, _ = print_state(gate, num_qubits, verbose_mode, result_state)

    if gate == 'Sign':
        index, _, _, _ = extract_qubits(command)
        result_state = sign_flip(num_qubits, index, state_vector)
        state_vector, _ = print_state(gate, num_qubits, verbose_mode, result_state)

    if gate == 'reverse':
        result_state = reverse_state(num_qubits, state_vector)
        state_vector, _ = print_state(gate, num_qubits, verbose_mode, result_state)

    if gate in ['QFT', 'IQFT']:
        qubit_start, qubit_end, _, _ = extract_qubits(command)
        if gate == 'QFT':
            print(f'Starting QFT from qubit {qubit_start} to qubit {qubit_end}')
            type = 1
        if gate == 'IQFT':
            print(f'Starting IQFT from qubit {qubit_start} to qubit {qubit_end}')
            type = -1
        result_state = dft(num_qubits, qubit_start, qubit_end, type, state_vector)
        if gate == 'QFT':
            print('Ending QFT ..')
        elif gate == 'IQFT':
            print('Ending IQFT ..')
        state_vector, _ = print_state(gate, num_qubits, verbose_mode, result_state)

    if gate == 'measure':
        if qubit_end >= qubit_start:
            for qubit in range(qubit_start, qubit_end + 1):
                measurement_array[qubit] = 1
                print(f'Measure qubit {qubit}')
        elif qubit_end < qubit_start:
            for qubit in range(qubit_start, qubit_end - 1, -1):
                measurement_array[qubit] = 1
                print(f'Measure qubit {qubit}')

def calculate_results(num_qubits, state, measurement_array):
    probabilities = np.zeros(int(2 ** np.sum(measurement_array)))
    amplitudes = ['' for _ in range(int(2 ** np.sum(measurement_array)))]

    for i in range(2 ** num_qubits):
        num = 0
        k = 0
        for j in range(num_qubits):
            if measurement_array[j] == 1:
                num += ((i >> j) & 1) * 2 ** k
                k += 1
        probabilities[num] += np.absolute(state[i]) ** 2

    for i in range(2 ** num_qubits):
        num = 0
        k = 0
        for j in range(num_qubits):
            if measurement_array[j] == 1:
                num += ((i >> j) & 1) * 2 ** k
                k += 1
        binary_str = ("{:0%db}" % num_qubits).format(i)[::-1]
        if np.absolute(state[i]) > 0.0001:
            if len(amplitudes[num]) == 0:
                amplitudes[num] = amplitudes[num] + f'({state[i] / np.sqrt(probabilities[num]):.3f})|{binary_str}>'
            else:
                amplitudes[num] = amplitudes[num] + f' + ({state[i] / np.sqrt(probabilities[num]):.3f})|{binary_str}>'
    if np.sum(measurement_array) > 0:
        if np.sum(measurement_array) > 1:
            print('\nProbabilities for measurements of qubits: '),
        else:
            print('\nProbability for measurement of qubit: '),
        for i in range(num_qubits):
            if measurement_array[i] == 1: print(f'{i} '),

    for i in range(int(2 ** np.sum(measurement_array))):
        binary_str = ("{:0%db}" % np.sum(measurement_array)).format(i)[::-1]
        if probabilities[i] > 0.00000000001:
            if np.sum(measurement_array) > 0:
                print(f'\nP({binary_str}) = '),
                print(probabilities[i])
            print('|psi> = '),
            print(amplitudes[i])

    return

def calculate_final_results(num_qubits, state, probability_threshold):
    print(f'\nFinal basis states with P > {probability_threshold}')
    probabilities = np.absolute(state) ** 2
    for k in range(2 ** num_qubits):
        if probabilities[k] > probability_threshold:
            binary_str = ("{:0%db}" % num_qubits).format(k)[::-1]
            psi = f'P(|{binary_str}>) = {probabilities[k]:.2e}\t Amplitude: {state[k]:.2e}'
            print(psi)
    return

import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def plot_results(num_qubits, state):
    probabilities = np.absolute(state) ** 2
    R = 1
    if num_qubits > 20:
        R = 2 ** (num_qubits - 20)
    y1 = np.reshape(probabilities, (-1, R)).max(axis=1)
    x = np.arange(len(y1)) * R

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_title("Probabilities for All Basis States", fontsize=14)
    ax.set_xlabel("Basis State", fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    ax.plot(x, y1, 'bo', x, y1, 'r--')

    plt.savefig('plot.png')
    plt.close(fig)
    print("Graph saved as plot.png!")


def plot_tkinter(num_qubits, state):

    probabilities = np.absolute(state) ** 2
    R = 1
    if num_qubits > 20:
        R = 2 ** (num_qubits - 20)
    y1 = np.reshape(probabilities, (-1, R)).max(axis=1)
    x = np.arange(len(y1)) * R

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_title("Probabilities for All Basis States", fontsize=14)
    ax.set_xlabel("Basis State", fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    ax.plot(x, y1, 'bo', x, y1, 'r--')

    root = tk.Tk()
    root.title("Quantum Probability Graph")

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    close_button = tk.Button(root, text="Close", command=lambda: quit_tkinter(root))
    close_button.pack()

    print("Graph successfully displayed in Tkinter!")

    root.mainloop()


def quit_tkinter(root):
    print("Closing Tkinter window...")
    root.quit()
    root.destroy()


if print_enabled == 1: calculate_final_results(num_qubits, state_vector, probability_threshold)
if np.sum(measurement_array) > 0: calculate_results(num_qubits, state_vector, measurement_array)

if plot_enabled == 1:
    import matplotlib.pyplot as plt
    print('Plotting the graph...')
    plot_results(num_qubits, state_vector)

    print('Opening Tkinter window...')
    plot_tkinter(num_qubits, state_vector)  #

print('Finished')


