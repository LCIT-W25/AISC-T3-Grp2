document.addEventListener("DOMContentLoaded", function () {
  const messageForm = document.getElementById("message-form");
  const userInput = document.getElementById("user-input");
  const chatContainer = document.getElementById("chat-container");
  const typingIndicator = document.getElementById("typing-indicator");

  // Focus on input field when page loads
  userInput.focus();

  messageForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    const message = userInput.value.trim();
    if (!message) return;

    userInput.value = "";

    // Display user message
    addMessage(message, "user-message");

    // Show typing indicator
    typingIndicator.style.display = "block";

    // Scroll to bottom
    scrollToBottom();

    // Send message to backend
    try {
      const response = await fetch("/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ message }),
      });

      // Hide typing indicator
      typingIndicator.style.display = "none";

      const data = await response.json();

      // Display bot response
      addMessage(data.response, "bot-message");

      // Scroll to bottom again
      scrollToBottom();
    } catch (error) {
      console.error("Error:", error);

      // Hide typing indicator
      typingIndicator.style.display = "none";

      // Display error message
      addMessage(
        "Sorry, there was an error processing your request.",
        "bot-message"
      );

      // Scroll to bottom
      scrollToBottom();
    }
  });

  // Function to add a message to the chat
  function addMessage(text, className) {
    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${className}`;
    messageDiv.textContent = text;
    chatContainer.appendChild(messageDiv);
  }

  // Function to scroll chat to bottom
  function scrollToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
  }
});
