const socket = io({ transports: ['websocket'] });
const image = document.querySelector("img");
const FRAME_HEIGHT = 480;
const FRAME_WIDTH = 640;
const usertype = 'watcher';
const stream = [];
const FRAME_RATE = 25;
let count = 0;
let timeoutId = setTimeout(() => {
    image.src = `/logo/streamError.jpg`;
 },5000);
let streamId = -1;

socket.emit('join', {usertype, room}, ()=>{
    console.log('joined')
})

socket.on('frameoutput', frame => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => {
       image.src = `/logo/streamError.jpg`;
    },5000);
    console.log(9);
    //stream.push(`data:image/jpg;base64,${frame}`)
    image.src = `data:image/jpg;base64,${frame}`;
})

socket.on('frameoutput0', frame => {
    console.log(9);
    //stream.push(`data:image/jpg;base64,${frame}`)
    image.src = `data:image/jpg;base64,${frame}`;
})
socket.on('frameoutput1', frame => {
    console.log(9);
    //stream.push(`data:image/jpg;base64,${frame}`)
    image.src = `data:image/jpg;base64,${frame}`;
})
socket.on('frameoutput2', frame => {
    console.log(9);
    //stream.push(`data:image/jpg;base64,${frame}`)
    image.src = `data:image/jpg;base64,${frame}`;
})
socket.on('frameoutput3', frame => {
    console.log(9);
    //stream.push(`data:image/jpg;base64,${frame}`)
    image.src = `data:image/jpg;base64,${frame}`;
})
socket.on('frameoutput4', frame => {
    console.log(9);
    //stream.push(`data:image/jpg;base64,${frame}`)
    image.src = `data:image/jpg;base64,${frame}`;
})

// setInterval(() => {
//     clearInterval(streamId)
//     streamId = setInterval(() => {
//     //console.log(stream.length)
//     if(stream.length){
//         image.src = stream.shift();
//     }
//     //
//    },1000/FRAME_RATE)
   
// },10000)