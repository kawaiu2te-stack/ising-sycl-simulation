import os
import sys
import numpy as np
import json
import subprocess

class IsingSimulator:
    def __init__(self, L, multi=1, env={}):
        self.L = L
        self.multi = multi
        subprocess.check_output(["make"])
        self.process = subprocess.Popen(
            ["./main_sycl", str(L)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            env=os.environ.copy() | {"MULTI": str(multi)} | env,
            bufsize=0,
        )

    def __stdin_write(self, str):
        self.process.stdin.write(str.encode("utf-8"))

    def __stdout_read(self, size):
        res = self.process.stdout.read(size)
        while len(res) < size:
            res += self.process.stdout.read(size - len(res))
        return res

    def resize(self, L):
        self.L = L
        self.__stdin_write(f"resize: {L}\n")

    def reset_spins(self):
        """
        set all system spins to 1
        """
        self.__stdin_write("reset_spins\n")

    def randomize_spins(self):
        """
        randomize all system spins with ±1
        """
        self.__stdin_write("randomize_spins\n")

    def reset_bonds(self, same=False):
        """
        SG only
        randomize all system bonds (system pair shares same bond)
        """
        if same:
            self.__stdin_write("reset_bonds_same\n")
        else:
            self.__stdin_write("reset_bonds\n")

    def set_temperatures(self, temperatures):
        assert len(temperatures) == 32 * self.multi
        self.__stdin_write(f"T: {" ".join(str(i) for i in temperatures)}\n")

    def run(self, burn_in_steps, calc_steps):
        self.__stdin_write(f"burn_in: {burn_in_steps}\n")
        self.__stdin_write(f"calc_steps: {calc_steps}\n")
        self.__stdin_write("run_simulation\n")
        size = int(self.process.stdout.readline())
        res = self.__stdout_read(size).decode("utf-8")
        self.process.stdout.readline()
        results = json.loads(res)
        for k, v in results.items():
            results[k] = np.array(v, dtype=np.float64)
        return results

    def get_spin(self, sys_i):
        assert 0 <= sys_i < 32 * self.multi
        self.__stdin_write(f"get_spin: {sys_i}\n")
        res = self.__stdout_read(self.L * self.L * self.L * 4)
        res = np.frombuffer(res, dtype=np.int32)
        res.shape = (self.L, self.L, self.L)
        return res

    def get_bond(self, sys_i):
        assert 0 <= sys_i < 32 * self.multi
        self.__stdin_write(f"get_bond: {sys_i}\n")
        res = self.__stdout_read(self.L * self.L * self.L * 3 * 4)
        res = np.frombuffer(res, dtype=np.int32)
        res.shape = (self.L, self.L, self.L, 3)
        return res

    def swap(self, sys_i, sys_j):
        """
        swap two systems
        """
        assert 0 <= sys_i < 32 * self.multi
        assert 0 <= sys_j < 32 * self.multi
        self.__stdin_write(f"swap: {sys_i} {sys_j}\n")

    def make_kexp(self):
        """
        make kexp table of current temps for faster update
        """
        self.__stdin_write(f"make_kexp\n")
        self.process.stdout.readline()

    def disable_kexp(self):
        """
        permanently disable kexp for remc
        """
        self.__stdin_write(f"disable_kexp\n")

    def __del__(self):
        if self.process:
            self.__stdin_write("exit\n")
            self.process.wait()
