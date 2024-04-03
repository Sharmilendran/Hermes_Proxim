// WebSocketComponent.jsx (Presentation Layer)
import React, { useState, useEffect } from 'react';
import WebSocketService from './../../Services/WebSocketService';
import TeamMember from "../Classification/Classification";


const WebSocketComponent = () => {
  const [messages, setMessages] = useState([]);
  const wsService = new WebSocketService('ws://localhost:8765');

  useEffect(() => {
    wsService.connect();
    wsService.addMessageHandler(data => {
       setMessages(prevMessages => [...prevMessages, data]);
    });

    return () => {
      wsService.disconnect();
    };
  }, []);

  const sendMessage = () => {
    wsService.sendMessage({ type: 'chatMessage', text: 'Hello, WebSocket!' });
  };

  return (
    <TeamMember members={messages} />
  );
};

export default WebSocketComponent;
