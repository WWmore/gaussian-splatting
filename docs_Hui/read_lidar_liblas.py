"""
https://liblas.org/tutorial/python.html
pip install libLAS
"""

from liblas import file ##BUG: OSError: [WinError 126] The specified module could not be found
f = file.File(path + las1,mode='r')

print(f.color)

header = f.header
for p in f:
    #print(p.x, p.y, p.z)
    c = p.color
    print(c.red, c.green, c.blue)

p = f.read(0)
p.x, p.y, p.z
p.scan_angle
p.scan_direction
p.return_number
p.number_of_returns
p.flightline_edge
p.classification
p.time
p.intensity
c = p.color
c.red
c.blue
c.green