//client side code for emitting stream to server
//see how to install opencv4nodejs at https://www.npmjs.com/package/opencv4nodejs
//run - 'npm install' to install all dependencies
//run - 'node client.js time=7 fps=2' to start the code, here time is duration in seconds of stream
// and fps is frames per second
const { io } = require("socket.io-client");
const cv = require('opencv4nodejs');
const SERVER_URL = 'http://localhost:5000';
const socket = io(SERVER_URL);
let FRAME_RATE = 2
let DURATION = 7 //default duration in seconds
let room = '';
socket.on('connect', () => {
    console.log('connected');
    room = socket.id;
})

//get duration and fps from command line
process.argv.forEach( (val) => {
    if(val.startsWith('time=')){
        let duration = val.replace('time=','')
        if(isNumericString(duration))
        DURATION = duration*1
    }

    if(val.startsWith('fps=')){
        let fps = val.replace('fps=','')
        if(isNumericString(fps))
        FRAME_RATE = fps*1
    }
} )
//open camera using opencv
const wcap = new cv.VideoCapture(0);


let cover_array = []
let multiface_array = []

//handle incoming data from server for each frame
socket.on('frameoutput1', (data) => {
    
    
    let multiface = data['multiple_face']
    let cover = data['cover_ratio']
    cover_array.push(cover)
    
    if(multiface == '1')
        multiface_array.push(multiface)
    else
        multiface_array.push('0')
})

const accurateTimer = (fn, time = 1000) => {
    // nextAt is the value for the next time the timer should fire.
    // timeout holds the timeoutID so the timer can be stopped.
    let nextAt, timeout;
    // Initilzes nextAt as now + the time in milliseconds you pass
    // to accurateTimer.
    nextAt = new Date().getTime() + time;
   
    // This function schedules the next function call.
    const wrapper = () => {
      // The next function call is always calculated from when the
      // timer started.
      nextAt += time;
      // this is where the next setTimeout is adjusted to keep the
      //time accurate.
      timeout = setTimeout(wrapper, nextAt - new Date().getTime());
      // the function passed to accurateTimer is called.
      fn();
    };
   
    // this function stops the timer.
    const cancel = () => clearTimeout(timeout);
   
    // the first function call is scheduled.
    timeout = setTimeout(wrapper, nextAt - new Date().getTime());
   
    // the cancel function is returned so it can be called outside
    // accurateTimer.
    return { cancel };
};

//recurringli capture frames and emit to server
const intervalId = accurateTimer(() => {
    try{
        const frame = wcap.read();
        const image_data_url = cv.imencode('.jpg', frame).toString('base64');
        socket.emit('frameinput1', {image_data_url, room})
    }catch(e){
        console.log(e)
    }
}, 1000/FRAME_RATE);


//define when to stop the camera
setTimeout(() => {
    intervalId.cancel();
    //wcap.release();
   
    let multiface = average(multiface_array)
    let cover = average(cover_array)

    if(multiface == -1)
        console.log('Multiple Faces: Not Defined')
    else{
        multiface = Math.round(multiface*10000)/100
        console.log('Multiple Faces:', multiface, '%')
    }

    if(cover == -1){
        console.log('Cover: Not Defined')
        console.log('Uncover: Not Defined')
    }else{
        cover = Math.round(cover*10000)/100
        let uncover = 100 - cover
        console.log('Cover:', cover, '%')
        console.log('Uncover:', uncover, '%')
    }
    //terminate code successfully
    process.exit()
}, DURATION * 1000);


function average(arr){
    
    let poss = -1
    let avg = 0
    arr.forEach(e => {
        if(e != '-1'){
            poss = 1
            avg += e*1
        }
            
    })
    if (poss == -1)
        return poss
    
    return avg/arr.length
}

function isNumericString(input) {  
    return typeof input === 'string' && !Number.isNaN(input)
}