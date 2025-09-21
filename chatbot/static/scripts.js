const ws = new WebSocket(`ws://${location.host}/ws`);
const messageInput = document.getElementById('messageText');
const chatBox = document.getElementById('chat-box');

ws.onmessage = function (event) {
    addMessage(event.data, 'bot-message');
};

function sendMessage() {
    const message = messageInput.value;
    if (message.trim() !== '') {
        ws.send(message);
        addMessage(message, 'user-message');
        messageInput.value = '';
    }
}

function addMessage(message, className) {
    const messageElement = document.createElement('div');
    messageElement.className = `message ${className}`;
    messageElement.textContent = message;
    chatBox.appendChild(messageElement);
    chatBox.scrollTop = chatBox.scrollHeight; // Auto-scroll to the latest message
}

messageInput.addEventListener('keypress', function (event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
});
