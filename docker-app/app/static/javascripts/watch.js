const socket = io({ transports: ['websocket'] });
const image = document.querySelector("img");
const liveliness = document.querySelector('#live');
const multiface = document.querySelector('#multiface');
const cover = document.querySelector('#cover');
const uncover = document.querySelector('#uncover');
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

socket.on('frameoutput0', (data) => {
    
    
    //stream.push(`data:image/jpg;base64,${frame}`)
    image.src = `data:image/jpg;base64,${data.result}`;
    if(data.multiple_face == "-1"){
        multiface.textContent = 'Not Defined'
    }else{
        multiface.textContent = data.multiple_face
    }
    
    if(data.live_confidence == "-1")
    liveliness.textContent = 'Not Defined'
    else
    liveliness.textContent = data.live_confidence

    if(data.cover_ratio == "-1"){
        cover.textContent = 'Not Defined'
        uncover.textContent = 'Not Defined'
    }else{
        cover.textContent = data.cover_ratio
        uncover.textContent = 1 - data.cover_ratio
    }
})