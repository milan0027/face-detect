from flask import Flask, request, jsonify, render_template,  redirect
from code2 import use_keras_after_zoom
from flask_socketio import SocketIO, emit, join_room, leave_room

app = Flask(__name__)
socketio = SocketIO(app)
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



@app.route('/predict1', methods=['POST'])
def predict1():
    data = request.get_json(force = True)
    # print(data['image'][0:10])
    #result = use_yolo_model(data['image'])
    #print(result[0:10])
    return jsonify(result=result)

@app.route('/predict2', methods=['POST'])
def predict2():
    data = request.get_json(force = True)
    # print(data['image'][0:10])
    #result = use_keras_after_zoom(data['image'])
    #print(result[0:10])
    return jsonify(result=result)

@app.route('/transmit', methods=['PUT'])
def transmit():
    #print('gotit')
    data = request.get_json(force = True)
    socketio.start_background_task(emit_function, data)
    return 1

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
    socketio.start_background_task(emit_function, data)



def emit_function(data):
    base64_data = data['image_data_url']
    room = data['room']
    result = base64_data
    multiple_face = 0
    live_confidence = -1
    cover_ratio = -1
    try:
        result, multiple_face, live_confidence, cover_ratio = use_keras_after_zoom(base64_data)
    except:
        print('An exception occured')

    #print(multiple_face, live_confidence, cover_ratio)
    
    data1 = {'result': result, 'multiple_face': multiple_face, 'live_confidence': str(live_confidence), 'cover_ratio': str(cover_ratio)} 
    try:
        socketio.emit('frameoutput0', data1, to=room)  
    except Exception as e:
        print(e)

if __name__ == '__main__':
    socketio.run(app,debug=True)