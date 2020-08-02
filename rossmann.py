

def preprocess_data(df):
    return df


def validate_get_register_count(NUMBER_OF_REGISTERS):
    if NUMBER_OF_REGISTERS < 1:
        NUMBER_OF_REGISTERS = 4

    return NUMBER_OF_REGISTERS
