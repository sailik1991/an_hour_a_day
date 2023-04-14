from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import os
import openai

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

openai.api_key = os.getenv("OPENAI_API_KEY")

@app.route('/')
def index():
    return render_template('index.html')

def pirate_style(text):
    prompt = f"Translate the following text to pirate speak: {text}"
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.5,
    )
    
    return response.choices[0].text.strip()

@socketio.on('message')
def handle_message(data):
    print('received message: ' + data)
    pirate_message = pirate_style(data)
    emit('message', pirate_message, broadcast=True)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
