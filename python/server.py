import socket

class Server:
  def __init__(self, host, port):
    self.host = host
    self.port = port
    self.server = None

  def connect(self):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as self.server:
      self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
      self.server.bind((self.host, self.port))
      self.server.listen()
      print(f"Server listening...")
      connection, _ = self.server.accept()
      print(f"Client connected.")
      return connection 
