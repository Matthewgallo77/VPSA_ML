import os
import time
import threading
import math

import numpy as np
import snap7
from snap7.util import *
from snap7.exceptions import Snap7Exception


class PLCManager:
    def __init__(self, ip_address=None, rack=0, slot=1):
        self.client = snap7.client.Client()
        self.ip_address = '192.168.10.21'
        self.rack = 0
        self.slot = 1
        self.db_number = 10 # datablock number
        self.tag_datatype = {'Int': 2, 'Real': 4}
        
        self.is_connected = False

        self.plc_info_path = 'configs/plc_config.txt'
        self.data_frame = None

        self.lock = threading.Lock() # lock used to synchronize PLC TCP requests


        self.connect_plc() # connect to plc
        
    def connect_plc(self):
        '''attempt to connect to PLC'''
        timeout = time.time() + 5 # 5 second timeout
        while time.time() < timeout:
            try:
                self.client.connect(self.ip_address, self.rack, self.slot)
                self.is_connected = True
                print("PLC connected")
                self.write_plc_config() # if successfully connected write IP address information
                return True
            except (Snap7Exception, RuntimeError) as e:
                time.sleep(2) # delay to seconds

        return False

    def read_tag_data(self, address, data_type):
        '''read data from PLC at specified address and datatype'''
        with self.lock: # acquire lock before reading data
            offset = self.get_offset(address)
            value = None  # Initialize value as None

            if 'I' in address:
                raw_value = self.client.read_area(snap7.types.Areas.PE, 0, int(offset[0]), self.tag_datatype[data_type])
                value = self.read_data_util(raw_value, data_type)
            elif 'M' in address:
                raw_value = self.client.read_area(snap7.types.Areas.MK, 0, int(offset[0]), self.tag_datatype[data_type])
                value = self.read_data_util(raw_value, data_type)
            elif 'Q' in address:
                raw_value = self.client.read_area(snap7.types.Areas.PA, 0, int(offset[0]), self.tag_datatype[data_type])
                value = self.read_data_util(raw_value, data_type)

            if value is not None and not math.isinf(value):
                return round(value, 5)
            else:
                return None

    def read_datablock(self):
        with self.lock:
            number_variables = 7
            bytes_to_read = number_variables*4
            raw_data = self.client.db_read(self.db_number, 0, bytes_to_read)
                    
            real_values = [get_real(raw_data, i * 4) for i in range(number_variables)]

            return real_values

    def read_data_util(self, byte_array, data_type):
        '''convert byte array to specific datatype'''
        if data_type == 'Real':
            return get_real(byte_array, 0)
        elif data_type == 'Int':
            return get_int(byte_array, 0)
        else:
            return 0
        
    def get_offset(self, address):
        '''get offset from address string'''
        if '.' not in address:
            address += '.0'
        return list(map(int, re.sub("[^\d\.]", "", address).split('.')))
    
    def write_plc_config(self):
        '''write plc config to text file'''
        os.makedirs(os.path.dirname(self.plc_info_path), exist_ok=True)
        with open(self.plc_info_path, 'w') as file:
            file.write(f"{self.ip_address},{self.rack},{self.slot}")

    def disconnect_plc(self):
        self.client.disconnect()  

