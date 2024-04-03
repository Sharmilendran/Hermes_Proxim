class WebSocketService {
    constructor(url) {
      this.url = url;
      this.socket = null;
      this.messageHandlers = [];
    }
  
    connect() {
      this.socket = new WebSocket(this.url);
      this.socket.onmessage = this.handleMessage.bind(this);
      // Handle other WebSocket events (onopen, onerror, etc.) here
    }
  
    handleMessage(event) {
      const data = JSON.parse(event.data);
      this.messageHandlers.forEach(handler => handler(data));
    }
  
    sendMessage(message) {
      this.socket.send(JSON.stringify(message));
    }
  
    addMessageHandler(handler) {
      this.messageHandlers.push(handler);
    }
  
    disconnect() {
      this.socket.close();
    }
  }
  
  export default WebSocketService;
  