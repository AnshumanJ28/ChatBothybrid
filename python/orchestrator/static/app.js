document.addEventListener("DOMContentLoaded", () => {
    const chatForm = document.getElementById("chat-form");
    const userInput = document.getElementById("user-input");
    const chatMessages = document.getElementById("chat-messages");
    const thinkingConsole = document.getElementById("thinking-console");
    const synthesisContent = document.getElementById("synthesis-content");
    const sourceList = document.getElementById("source-list");
    const resetBtn = document.getElementById("reset-btn");

    let isThinking = false;

    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const text = userInput.value.trim();
        if (!text || isThinking) return;

        isThinking = true;
        userInput.value = "";
        userInput.disabled = true;

        // 1. Add User Message
        appendMessage(text, "user");

        // 2. Clear panels & show loading status
        thinkingConsole.innerHTML = '<div class="log-entry info"><span class="log-bullet"></span><span>Initializing MiniBrain cognitive routing loop...</span></div>';
        synthesisContent.innerHTML = '<div class="placeholder-text">Awaiting search signals...</div>';
        sourceList.innerHTML = '<div class="placeholder-text text-sm">No sources cited.</div>';

        // 3. Add bot typing indicator
        const typingId = showTypingIndicator();

        try {
            const response = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: text })
            });

            if (!response.ok) {
                throw new Error("Server responded with error status");
            }

            const data = await response.json();
            
            // 4. Stagger-render thoughts
            await renderThoughts(data.thoughts);

            // 5. Remove typing indicator and render actual bot message
            removeTypingIndicator(typingId);
            appendMessage(data.answer, "bot");

            // 6. Handle synthesis panel updates
            if (data.subsystem === "web") {
                renderSynthesis(data.answer, data.sources);
            } else {
                synthesisContent.innerHTML = '<div class="placeholder-text">No active search triggered (query resolved locally).</div>';
                sourceList.innerHTML = '<div class="placeholder-text text-sm">No sources cited.</div>';
            }

        } catch (error) {
            removeTypingIndicator(typingId);
            appendMessage("Sorry, I encountered an internal error. Please check if the server is running.", "bot");
            thinkingConsole.innerHTML = `<div class="log-entry" style="color: var(--rose)"><span class="log-bullet" style="background: var(--rose)"></span><span>Error: ${error.message}</span></div>`;
        } finally {
            isThinking = false;
            userInput.disabled = false;
            userInput.focus();
        }
    });

    resetBtn.addEventListener("click", async () => {
        try {
            const response = await fetch("/api/reset", { method: "POST" });
            const data = await response.json();
            
            chatMessages.innerHTML = "";
            appendMessage("Dialogue state and C++ memory cell reset. Conversational history cleared.", "bot");
            
            thinkingConsole.innerHTML = `<div class="log-entry success"><span class="log-bullet"></span><span>${data.message}</span></div>`;
            synthesisContent.innerHTML = '<div class="placeholder-text">No active search triggered.</div>';
            sourceList.innerHTML = '<div class="placeholder-text text-sm">No sources cited.</div>';
        } catch (error) {
            console.error("Reset failed:", error);
        }
    });

    function appendMessage(text, sender) {
        const msgDiv = document.createElement("div");
        msgDiv.className = `message ${sender}-message`;
        
        const contentDiv = document.createElement("div");
        contentDiv.className = "message-content";
        contentDiv.innerText = text;
        
        msgDiv.appendChild(contentDiv);
        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function showTypingIndicator() {
        const id = "typing-" + Date.now();
        const msgDiv = document.createElement("div");
        msgDiv.className = "message bot-message";
        msgDiv.id = id;
        
        const indicator = document.createElement("div");
        indicator.className = "typing-indicator";
        indicator.innerHTML = '<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>';
        
        msgDiv.appendChild(indicator);
        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return id;
    }

    function removeTypingIndicator(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    async function renderThoughts(thoughts) {
        if (!thoughts || thoughts.length === 0) {
            thinkingConsole.innerHTML = '<div class="placeholder-text">No cognitive thoughts tracked.</div>';
            return;
        }

        thinkingConsole.innerHTML = "";
        for (let i = 0; i < thoughts.length; i++) {
            const step = thoughts[i];
            const entry = document.createElement("div");
            entry.className = "log-entry";
            
            if (step.includes("C++ LSTM") || step.includes("C++ similarity")) {
                entry.classList.add("info");
            } else if (step.includes("High-confidence") || step.includes("complete: True") || step.includes("complete. Submitting")) {
                entry.classList.add("success");
            } else if (step.includes("[Web Scraper]")) {
                entry.classList.add("info");
            }

            entry.innerHTML = `<span class="log-bullet"></span><span>${step}</span>`;
            thinkingConsole.appendChild(entry);
            thinkingConsole.scrollTop = thinkingConsole.scrollHeight;

            // Stagger spacing
            await new Promise(resolve => setTimeout(resolve, 150));
        }
    }

    function renderSynthesis(summary, sources) {
        // Render summary text card
        synthesisContent.innerHTML = `
            <div class="summary-card active">
                <p>${summary}</p>
            </div>
        `;

        // Render source tags
        if (sources && sources.length > 0) {
            sourceList.innerHTML = "";
            sources.forEach((url, index) => {
                if (!url) return;
                const a = document.createElement("a");
                a.href = url;
                a.target = "_blank";
                a.className = "source-tag";
                
                // Display domain name or snippet URL title
                let displayName = url;
                try {
                    const parsed = new URL(url);
                    displayName = parsed.hostname + (parsed.pathname.length > 10 ? parsed.pathname.substring(0, 10) + "..." : parsed.pathname);
                } catch(e) {}
                
                a.innerText = `[${index + 1}] ${displayName}`;
                a.title = url;
                sourceList.appendChild(a);
            });
        } else {
            sourceList.innerHTML = '<div class="placeholder-text text-sm">No sources cited (stub fallback triggered).</div>';
        }
    }
});
