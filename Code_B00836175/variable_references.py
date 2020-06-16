
class VariableReference(object):
    registers = {}

    def __init__(self, number_of_registers):
        for idx in range(number_of_registers):
            self.registers["R"+str(idx)] = 0

    def set_registers(self, values):
        for idx in range(len(values)):
            self.registers["R"+str(idx)] = values[idx]

    def get_registers(self):
        return self.registers

    def reset_registers(self):
        for k, v in self.registers.items():
            self.registers[k] = 0
