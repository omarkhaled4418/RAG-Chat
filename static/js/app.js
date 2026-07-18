/* ================================================================
   RAG Chat — Client-side Application Logic
   ================================================================ */

(() => {
    "use strict";

    // ── DOM refs ────────────────────────────────────────────────────
    const $ = (sel) => document.querySelector(sel);
    const chatForm      = $("#chatForm");
    const chatInput      = $("#chatInput");
    const messagesEl     = $("#messages");
    const welcomeEl      = $("#welcome");
    const chatContainer  = $("#chatContainer");
    const btnSend        = $("#btnSend");
    const btnNewChat     = $("#btnNewChat");
    const btnClearIndex  = $("#btnClearIndex");
    const dropzone       = $("#dropzone");
    const fileInput      = $("#fileInput");
    const uploadProgress = $("#uploadProgress");
    const progressFill   = $("#progressFill");
    const progressText   = $("#progressText");
    const chunkCount     = $("#chunkCount");
    const sidebar        = $("#sidebar");
    const sidebarOpen    = $("#sidebarOpen");
    const sidebarClose   = $("#sidebarClose");

    // ── State ───────────────────────────────────────────────────────
    let sessionId = crypto.randomUUID();
    let isStreaming = false;

    // ── Init ────────────────────────────────────────────────────────
    fetchIndexStatus();

    // ── Toast notifications ─────────────────────────────────────────
    function createToastContainer() {
        let c = document.querySelector(".toast-container");
        if (!c) {
            c = document.createElement("div");
            c.className = "toast-container";
            document.body.appendChild(c);
        }
        return c;
    }

    function showToast(message, type = "success") {
        const container = createToastContainer();
        const toast = document.createElement("div");
        toast.className = `toast ${type}`;
        const icon = type === "success" ? "✅" : "❌";
        toast.innerHTML = `<span>${icon}</span><span>${message}</span>`;
        container.appendChild(toast);
        setTimeout(() => {
            toast.style.opacity = "0";
            toast.style.transform = "translateY(8px)";
            toast.style.transition = "all 0.3s ease";
            setTimeout(() => toast.remove(), 300);
        }, 3500);
    }

    // ── Sidebar toggle (mobile) ─────────────────────────────────────
    sidebarOpen.addEventListener("click", () => sidebar.classList.add("open"));
    sidebarClose.addEventListener("click", () => sidebar.classList.remove("open"));

    // ── Auto-resize textarea ────────────────────────────────────────
    chatInput.addEventListener("input", () => {
        chatInput.style.height = "auto";
        chatInput.style.height = Math.min(chatInput.scrollHeight, 150) + "px";
    });

    // ── Submit on Enter (Shift+Enter for newline) ───────────────────
    chatInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event("submit"));
        }
    });

    // ── Chat form submit ────────────────────────────────────────────
    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const message = chatInput.value.trim();
        if (!message || isStreaming) return;

        // Hide welcome
        welcomeEl.style.display = "none";

        // Add user message
        appendMessage("user", message);

        // Clear input
        chatInput.value = "";
        chatInput.style.height = "auto";
        btnSend.disabled = true;
        isStreaming = true;

        // Add bot placeholder with typing indicator
        const botEl = appendMessage("bot", null, true);

        try {
            const response = await fetch("/api/chat/stream", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message, session_id: sessionId }),
            });

            if (!response.ok) throw new Error("Server error");

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let botText = "";
            let sources = [];
            let buffer = "";

            // Remove typing indicator
            const bubble = botEl.querySelector(".message-bubble");
            bubble.innerHTML = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });

                const lines = buffer.split("\n");
                buffer = lines.pop(); // keep incomplete line

                for (const line of lines) {
                    if (!line.startsWith("data: ")) continue;
                    const payload = line.slice(6).trim();
                    if (payload === "[DONE]") continue;

                    try {
                        const data = JSON.parse(payload);
                        if (data.token) {
                            botText += data.token;
                            bubble.innerHTML = marked.parse(botText);
                            scrollToBottom();
                        }
                        if (data.sources) {
                            sources = data.sources;
                        }
                    } catch (_) {
                        // skip parse errors
                    }
                }
            }

            // Add sources
            if (sources.length > 0) {
                const srcContainer = document.createElement("div");
                srcContainer.className = "message-sources";
                for (const s of sources) {
                    const tag = document.createElement("span");
                    tag.className = "source-tag";
                    tag.textContent = `${s.source} — p.${s.page}`;
                    srcContainer.appendChild(tag);
                }
                botEl.querySelector(".message-body").appendChild(srcContainer);
            }

        } catch (err) {
            const bubble = botEl.querySelector(".message-bubble");
            bubble.textContent = "Sorry, something went wrong. Make sure Ollama is running.";
            showToast("Failed to get response", "error");
        }

        isStreaming = false;
        btnSend.disabled = false;
        chatInput.focus();
        scrollToBottom();
    });

    // ── File upload ─────────────────────────────────────────────────
    dropzone.addEventListener("click", () => fileInput.click());

    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("drag-over");
    });

    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("drag-over");
    });

    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("drag-over");
        const files = Array.from(e.dataTransfer.files).filter(f => f.name.endsWith(".pdf"));
        if (files.length > 0) uploadFiles(files);
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files.length > 0) {
            uploadFiles(Array.from(fileInput.files));
            fileInput.value = "";
        }
    });

    async function uploadFiles(files) {
        uploadProgress.hidden = false;
        progressFill.style.width = "10%";
        progressText.textContent = `Uploading ${files.length} file(s)...`;

        const formData = new FormData();
        for (const f of files) formData.append("files", f);

        try {
            progressFill.style.width = "40%";

            const res = await fetch("/api/upload", {
                method: "POST",
                body: formData,
            });

            progressFill.style.width = "80%";
            const data = await res.json();

            progressFill.style.width = "100%";

            const successes = data.results.filter(r => !r.error);
            const errors = data.results.filter(r => r.error);

            if (successes.length > 0) {
                const totalChunks = successes.reduce((s, r) => s + r.chunks, 0);
                showToast(`Indexed ${successes.length} file(s) — ${totalChunks} chunks`);
            }
            if (errors.length > 0) {
                for (const e of errors) showToast(`${e.filename}: ${e.error}`, "error");
            }

            progressText.textContent = "Done!";
            fetchIndexStatus();

            setTimeout(() => {
                uploadProgress.hidden = true;
                progressFill.style.width = "0%";
            }, 2000);

        } catch (err) {
            progressText.textContent = "Upload failed";
            showToast("Upload failed", "error");
            setTimeout(() => {
                uploadProgress.hidden = true;
                progressFill.style.width = "0%";
            }, 2000);
        }
    }

    // ── New Chat ────────────────────────────────────────────────────
    btnNewChat.addEventListener("click", async () => {
        await fetch("/api/history/clear", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id: sessionId }),
        });

        sessionId = crypto.randomUUID();
        messagesEl.innerHTML = "";
        welcomeEl.style.display = "";
        showToast("New chat started");
        sidebar.classList.remove("open");
    });

    // ── Clear Index ─────────────────────────────────────────────────
    btnClearIndex.addEventListener("click", async () => {
        if (!confirm("Clear all indexed documents? This cannot be undone.")) return;

        await fetch("/api/index/clear", { method: "POST" });
        fetchIndexStatus();
        showToast("Index cleared");
    });

    // ── Helpers ─────────────────────────────────────────────────────

    function appendMessage(role, text, typing = false) {
        const msg = document.createElement("div");
        msg.className = `message ${role}`;

        const avatar = document.createElement("div");
        avatar.className = "message-avatar";
        avatar.textContent = role === "user" ? "👤" : "🤖";

        const body = document.createElement("div");
        body.className = "message-body";

        const bubble = document.createElement("div");
        bubble.className = "message-bubble";

        if (typing) {
            bubble.innerHTML = `
                <div class="typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>`;
        } else {
            // Render regular messages (like user messages) as text, or bot messages as markdown if needed.
            // For simplicity, we just set textContent for user messages, but if a bot message is added statically, parse it.
            if (role === "bot") {
                bubble.innerHTML = marked.parse(text);
            } else {
                bubble.textContent = text;
            }
        }

        body.appendChild(bubble);
        msg.appendChild(avatar);
        msg.appendChild(body);
        messagesEl.appendChild(msg);
        scrollToBottom();
        return msg;
    }

    function scrollToBottom() {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    async function fetchIndexStatus() {
        try {
            const res = await fetch("/api/index/status");
            const data = await res.json();
            chunkCount.textContent = data.chunks_indexed.toLocaleString();
        } catch (_) {
            chunkCount.textContent = "—";
        }
    }
})();
