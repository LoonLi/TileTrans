import socket
import time
import argparse

HOST = "0.0.0.0"

def log(logfile="training", port=19999):
    with socket.socket() as s:
        s.bind((HOST, port))
        s.listen()
        print("Start listning at {}:{}".format(HOST, port))
        while True:
            conn, addr = s.accept()
            with conn:
                print("Connected by", addr)
                while True:
                    data = conn.recv(4096)
                    if len(data) != 0:
                        msg = data.decode()
                        if "[STOP]" in msg:
                            print("Close connection.")
                            conn.close()
                            break
                        print(msg)
                        with open("tmp/{}.log".format(logfile), 'a') as f:
                            f.write(msg + "\n")
                    else:
                        time.sleep(1)
                    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', "--save-file", dest="logfile", default="training", help="The name of log file that recoreds sever output.")
    parser.add_argument('-p', dest="port", default="19999", help="Server port.")
    args = parser.parse_args()
    log(args.logfile, int(args.port))