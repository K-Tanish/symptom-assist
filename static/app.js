// Example async function for handling chat submissions with improved error handling
async function submitSymptomChat(payload) {
    const chatInput = document.getElementById('chat-input');
    const submitBtn = document.getElementById('submit-btn');
    const errorDisplay = document.getElementById('error-message-banner'); // Ensure this element exists in UI

    try {
        // Toggle loading states safely
        submitBtn.disabled = true;
        if (errorDisplay) errorDisplay.classList.add('hidden');

        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `Server responded with status: ${response.status}`);
        }

        const data = await response.json();
        renderChatResponse(data);

    } catch (error) {
        console.error('Failed to process symptom analysis:', error);
        
        // Provide clear, actionable feedback to the user
        if (errorDisplay) {
            errorDisplay.textContent = `Error: ${error.message}. Please check your connection or try again later.`;
            errorDisplay.classList.remove('hidden');
        } else {
            alert(`System Error: ${error.message}`);
        }
    } finally {
        // Always restore UI interactive state
        submitBtn.disabled = false;
    }
}