import socket                   # Import socket module

s = socket.socket()             # Create a socket object
host = socket.gethostname()     # Get local machine name
port = 60000                    # Reserve a port for your service.

s.connect((host, port))
s.send(b"Hello server!")

with open("my_file.txt", "r") as myfile:
  data = myfile.read().replace('\n', '')

s.send(data.encode('utf-8'))
print('Successfully sent the file')

outp = s.recv(1024)
outp = outp.decode("utf-8")
print('The letter you uploaded is predicted to be', outp)
s.close()
print('connection closed')
