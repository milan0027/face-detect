from flask import Flask, request, jsonify, render_template,  redirect
from flask_socketio import SocketIO, join_room, leave_room
from flask_celery import make_celery
from code2 import combined
import eventlet
eventlet.monkey_patch()  

# the app is an instance of the Flask class
app = Flask(__name__)
app.config['SECRET_KEY'] = 'there is no secret'

# app.config.update takes the following parameters:
# CELERY_BROKER_URL is the URL where the message broker is running
# CELERY_RESULT_BACKEND is required to keep track of task and store the status
app.config.update(
CELERY_BROKER_URL = 'redis://localhost:6379/0',
CELERY_RESULT_BACKEND='redis://localhost:6379/0'
)

# integrates Flask-SocketIO with the Flask application
socketio = SocketIO(app, message_queue='redis://localhost:6379/0')

# the app is passed to meke_celery function, this function sets up celery in order to integrate with the flask application
celery = make_celery(app)

@app.route('/')
def homepage():
    return render_template('homepage.html')

@app.route('/stream', methods=['GET','POST'])
def stream():
    if(request.method  == 'GET'):
        return render_template('streamform.html')
    if(request.method == 'POST'):
        room = request.form['room']
        return redirect('/stream/'+room)

@app.route('/watch', methods=['GET','POST'])
def watch():
    if(request.method  == 'GET'):
        return render_template('watchform.html')
    if(request.method == 'POST'):
        room = request.form['room']
        return redirect('/watch/'+room)


@app.route('/watch/<room>', methods=['GET'])
def watch_room(room):
    return render_template('watch.html', room=room)

@app.route('/stream/<room>', methods=['GET'])
def stream_room(room):
    return render_template('stream.html', room=room)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force = True)
    base64_data = data['image_data_url']
    result = base64_data
    multiple_face = 0
    live_confidence = -1
    cover_ratio = -1
    try:
        result, multiple_face, live_confidence, cover_ratio = combined(base64_data)
    except Exception as e:
        print(e)

    #print(multiple_face, live_confidence, cover_ratio)
    
    data1 = {'result': result, 'multiple_face': multiple_face, 'live_confidence': str(live_confidence), 'cover_ratio': str(cover_ratio)}
    return jsonify(data1)

@socketio.on('connect')
def test_connect():
    print('connected')

@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected')

@socketio.on('join')
def on_join(data):
    usertype = data['usertype']
    room = data['room']
    join_room(room)

@socketio.on('leave')
def on_leave(data):
    room = data['room']
    leave_room(room)

@socketio.on('frameinput')
def on_frameinput0(data):
    emit_function.delay(data)

@celery.task()
def emit_function(data):
    local_socketio = SocketIO(message_queue='redis://localhost:6379/0')
    base64_data = data['image_data_url']
    room = data['room']
    result = base64_data
    multiple_face = 0
    live_confidence = -1
    cover_ratio = -1
    try:
        result, multiple_face, live_confidence, cover_ratio = combined(base64_data)
    except Exception as e:
        print(e)
    #print(multiple_face, live_confidence, cover_ratio)
    data1 = {'result': result, 'multiple_face': multiple_face, 'live_confidence': str(live_confidence), 'cover_ratio': str(cover_ratio)}
    try:
        local_socketio.emit('frameoutput0', data1, to=room)
    except Exception as e:
        print(e)

if __name__ == '__main__':
    socketio.run(app,debug=False)