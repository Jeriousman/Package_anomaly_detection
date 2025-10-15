import serial
import serial.tools.list_ports
from multiprocessing import shared_memory


# memory share
shm = shared_memory.SharedMemory(create=True, size=8)

with open("memory_address.txt","w") as f:
    f.write(shm.name)

# Get Serial Port 
ports = list(serial.tools.list_ports.comports())
arduino_port = None
for port in ports:
    print(port.usb_description())
    print(port.description)
    if("USB-SERIAL CH340" in port.description):
        arduino_port = port.usb_description()
        print(f"Selected : {arduino_port}")

if(arduino_port==None):
    raise Exception("Arduino Not Found")
else:
    print(arduino_port)

ser = serial.Serial(arduino_port)

while True:
    ser_value = ser.readline().strip()

    # with open("./ser_value.txt", "wb") as f:
    #     f.write(ser_value)

    shm.buf[:len(ser_value)] = ser_value

ser.close()
    