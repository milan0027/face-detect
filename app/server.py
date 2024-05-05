"""Flask app for local server. This should be used for development purposes. Also integrates swagger ui"""
from flask import Flask, request, jsonify, render_template
from code5 import combined, realtime, mixed
from flask_socketio import SocketIO, join_room, leave_room
from flask_swagger_ui import get_swaggerui_blueprint

app = Flask(__name__)
socketio = SocketIO(app)

SWAGGER_URL="/swagger"
API_URL="/static/swagger.json"

swagger_ui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': 'Access API'
    }
)

app.register_blueprint(swagger_ui_blueprint, url_prefix=SWAGGER_URL)
@app.route('/')
def homepage():
    return render_template('homepage.html')

@app.route('/client')
def clientpage():
    return render_template('client.html')

@app.route('/realtime')
def realtimepage():
    return render_template('realtime.html')

@app.route('/mixed')
def mixedpage():
    return render_template('mixed.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force = True)
    base64_data = data['image_data_url']
    multiple_face = 0
    cover_ratio = -1
    try:
        multiple_face, cover_ratio = combined(base64_data)
    except Exception as e:
        print(e)
    
    data1 = {'multiple_face': str(multiple_face),'cover_ratio': str(cover_ratio)}
    return jsonify(data1)

@socketio.on('connect')
def test_connect():
    print('connected')

@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected')

@socketio.on('join')
def on_join(data):
    room = data['room']
    join_room(room)

@socketio.on('leave')
def on_leave(data):
    room = data['room']
    leave_room(room)


@socketio.on('frameinput1')
def on_frameinput1(data):
    emit_function1(data)

@socketio.on('realtimein')
def on_realtimein(data):
    emit_realtime(data)

@socketio.on('mixedin')
def on_mixedin(data):
    emit_mixed(data)


def emit_function1(data):
    base64_data = data['image_data_url']
    room = data['room']
    multiple_face = 0
    cover_ratio = -1
    try:
        multiple_face, cover_ratio = combined(base64_data)
    except Exception as e:
        print(e)
    
    data1 = {'multiple_face': str(multiple_face), 'cover_ratio': str(cover_ratio)} 
    try:
        socketio.emit('frameoutput1', data1, to=room)  
    except Exception as e:
        print(e)

def emit_realtime(data):
    base64_data = data['image_data_url']
    room = data['room']
    try:
       base64_data = realtime(base64_data)
    except Exception as e:
       print(e)

    data1 = {'image_data_url': base64_data}

    try:
        socketio.emit('realtimeout', data1, to = room)
    except Exception as e:
        print(e)

def emit_mixed(data):
    base64_data = data['image_data_url']
    room = data['room']
    multiple_face = 0
    cover_ratio = -1
    try:
        base64_data, multiple_face, cover_ratio = mixed(base64_data)
    except Exception as e:
        print(e)
    
    data1 = {'image_data_url':base64_data,'multiple_face': str(multiple_face), 'cover_ratio': str(cover_ratio)} 
    try:
        socketio.emit('mixedout', data1, to=room)  
    except Exception as e:
        print(e)


if __name__ == '__main__':
    socketio.run(app,debug=False)