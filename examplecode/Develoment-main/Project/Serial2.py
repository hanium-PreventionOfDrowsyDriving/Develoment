import serial 


class counting():
    def __init__(self):
        return 'WARNING'
        
class Ultra(counting):
    def __init__(self):
        port = '/dev/ttyUSB0'
        brate = 9600 #boudrate
        cmd = 'temp'
        
        self.seri = serial.Serial(port, baudrate = brate, timeout = None) #Serial(port = rp`s port, transmission port, time limit inifity)
        self.seri.write(cmd.encode())
        print(self.seri.name) # connecting port name
        
    def examining(self):
        if self.seri.in_waiting != 0 :
            content = self.seri.readline()
        #print(content)
        #print(content[:-2].decode())
        
        if self.content == b'3\r\n':
            super.counting()
        