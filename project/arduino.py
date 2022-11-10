import serial 

class Warning:
    def printy(self, power):
        self.power = power
        return self.power
        
class Aserial:
    def __init__(self):
        port = '/dev/ttyUSB0'
        brate = 9600 #boudrate
        cmd = 'temp'
        
        self.seri = serial.Serial(port, baudrate = brate, timeout = None) #Serial(port = rp`s port, transmission port, time limit inifity)
        self.seri.write(cmd.encode())
        # print("testing")
        print(self.seri.name) # connecting port name
        
    def counting(self):
        
        path = self.seri.name
        print(path)
        # if path.in_waiting != 0 :
        # content = 
        # print(content[:-2].decode())
            
        if seri.readline() == b'3\r\n':
            print('aaaa')
            #super().printy('qqqqq')
        else:
            print('kkkk')
                #super.printy('wwwwww')
'''
       self.seri = serial.Serial(port, baudrate = brate, timeout = None) #Serial(port = rp`s port, transmission port, time limit inifity)
        self.seri.write(cmd.encode())
        # print("testing")
        print(self.seri.name) # connecting port name
        
        
        if self.seri.in_waiting:
            print("return")
            content = str(self.seri.readline())
            print(content)
            '''