import google.generativeai as genai
import os

genai.configure(api_key=os.environ["API_KEY"])


model = genai.GenerativeModel("gemini-1.5-flash")
chat = model.start_chat()
while True:
    instruction = input("Enter instructions (or type 'exit' to quit): ")
    if instruction.lower() == 'exit':
        break
    response = chat.send_message(f"You are controlling a vehicle. You can use the control inputs left, right, forward, and backwards to move. Each of the inputs moves the robot 2 centimeters towards the given direction. Keep track of any objects you might encounter. Produce an output consisting of control inputs separated by commas, and try to fulfill the following instructions: {instruction}")
    print(response.text)

