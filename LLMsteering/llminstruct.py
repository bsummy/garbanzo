import google.generativeai as genai
import os
import socket
 
# Configure Generative AI
genai.configure(api_key=os.environ["API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")
chat = model.start_chat()
 
# Socket Setup
HOST = "127.0.0.1"  # Localhost
PORT = 65432        # Port for socket communication
 
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)
    print(f"AI Server listening on {HOST}:{PORT}...")
 
    conn, addr = server_socket.accept()
    with conn:
        print(f"Connected by {addr}")
        while True:
            # Receive instruction from the client
            instruction = conn.recv(1024).decode('utf-8')
            if not instruction or instruction.lower() == 'exit':
                print("Closing connection...")
                break
            print(f"received instruction {instruction}")
            # Generate response using the AI model
            response = chat.send_message(
                f"You are controlling a vehicle. You can use the control inputs left, right, forward, and backward to move. Each of the inputs moves the robot 2 centimeters towards the given direction. Keep track of any objects you might encounter. Produce an output consisting of control inputs separated by commas, and try to fulfill the following instructions: {instruction}"
            )
            print("got response")
            # Send response back to the client
            conn.sendall(response.text.encode('utf-8'))
            print(f"Sent response: {response.text}")