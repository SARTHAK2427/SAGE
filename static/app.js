// SAGE Web Client Logic

document.addEventListener("DOMContentLoaded", () => {

    const chatHistory = document.getElementById("chatHistory");

    const promptInput = document.getElementById("promptInput");

    const sendBtn = document.getElementById("sendBtn");

    const attachBtn = document.getElementById("attachBtn");

    const fileInput = document.getElementById("fileInput");

    const attachmentsTray = document.getElementById("attachmentsTray");

    const dropZoneOverlay = document.getElementById("dropZoneOverlay");

    const currentModelLabel = document.getElementById("currentModelLabel");



    // Telemetry Elements

    const telemetryCard = document.getElementById("telemetryCard");

    const metricAgentCalls = document.getElementById("metricAgentCalls");

    const metricDocCalls = document.getElementById("metricDocCalls");

    const metricCoderCalls = document.getElementById("metricCoderCalls");

    const metricSynthesizerCalls = document.getElementById("metricSynthesizerCalls");

    const metricSynthTime = document.getElementById("metricSynthTime");

    const metricSwitches = document.getElementById("metricSwitches");

    const metricWallTime = document.getElementById("metricWallTime");

    const traceTimeline = document.getElementById("traceTimeline");



    let attachedFiles = [];



    // Auto status poll

    async function updateStatus() {

        try {

            const res = await fetch("/api/status");

            if (res.ok) {

                const data = await res.json();

                if (data.current_model) {

                    const name = data.models[data.current_model] || data.current_model;

                    currentModelLabel.textContent = `Active: ${name}`;

                } else {

                    currentModelLabel.textContent = "System Ready (Sequential)";

                }

            }

        } catch (e) {

            currentModelLabel.textContent = "Offline";

        }

    }

    setInterval(updateStatus, 4000);

    updateStatus();



    // Attach File Trigger

    attachBtn.addEventListener("click", () => fileInput.click());



    fileInput.addEventListener("change", (e) => {

        handleFiles(Array.from(e.target.files));

        fileInput.value = "";

    });



    // Clipboard Paste for Images

    window.addEventListener("paste", (e) => {

        const items = (e.clipboardData || e.originalEvent.clipboardData).items;

        const pastedFiles = [];

        for (let item of items) {

            if (item.kind === "file") {

                const file = item.getAsFile();

                if (file) {

                    const ext = file.type.split("/")[1] || "png";

                    const renamed = new File([file], `clipboard_${Date.now()}.${ext}`, { type: file.type });

                    pastedFiles.push(renamed);

                }

            }

        }

        if (pastedFiles.length > 0) {

            handleFiles(pastedFiles);

        }

    });



    // Drag and Drop

    window.addEventListener("dragover", (e) => {

        e.preventDefault();

        dropZoneOverlay.classList.add("active");

    });



    window.addEventListener("dragleave", (e) => {

        if (e.clientX <= 0 || e.clientY <= 0) {

            dropZoneOverlay.classList.remove("active");

        }

    });



    window.addEventListener("drop", (e) => {

        e.preventDefault();

        dropZoneOverlay.classList.remove("active");

        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {

            handleFiles(Array.from(e.dataTransfer.files));

        }

    });



    function handleFiles(files) {

        files.forEach((file) => {

            // Avoid exact duplicate additions

            if (!attachedFiles.some(f => f.name === file.name && f.size === file.size)) {

                attachedFiles.push(file);

            }

        });

        renderAttachments();

    }



    function renderAttachments() {

        if (attachedFiles.length === 0) {

            attachmentsTray.style.display = "none";

            attachmentsTray.innerHTML = "";

            return;

        }



        attachmentsTray.style.display = "flex";

        attachmentsTray.innerHTML = "";



        attachedFiles.forEach((file, index) => {

            const pill = document.createElement("div");

            pill.className = "attachment-pill";



            const isImage = file.type.startsWith("image/");

            if (isImage) {

                const img = document.createElement("img");

                img.className = "attachment-thumb";

                img.src = URL.createObjectURL(file);

                pill.appendChild(img);

            }



            const info = document.createElement("span");

            const sizeKb = (file.size / 1024).toFixed(1);

            info.textContent = `${file.name} (${sizeKb} KB)`;

            pill.appendChild(info);



            const removeBtn = document.createElement("button");

            removeBtn.className = "remove-btn";

            removeBtn.innerHTML = "×";

            removeBtn.title = "Remove";

            removeBtn.addEventListener("click", () => {

                attachedFiles.splice(index, 1);

                renderAttachments();

            });

            pill.appendChild(removeBtn);



            attachmentsTray.appendChild(pill);

        });

    }



    // Auto-resize textarea

    promptInput.addEventListener("input", () => {

        promptInput.style.height = "auto";

        promptInput.style.height = Math.min(promptInput.scrollHeight, 150) + "px";

    });



    // Enter key to send (Shift+Enter for newline)

    promptInput.addEventListener("keydown", (e) => {

        if (e.key === "Enter" && !e.shiftKey) {

            e.preventDefault();

            sendMessage();

        }

    });



    sendBtn.addEventListener("click", sendMessage);



    async function sendMessage() {

        const text = promptInput.value.trim();

        if (!text && attachedFiles.length === 0) return;



        // Freeze input

        sendBtn.disabled = true;

        promptInput.disabled = true;

        sendBtn.innerHTML = `<span>Running...</span>`;



        // Render User Bubble

        appendUserMessage(text, attachedFiles);



        // Prepare FormData

        const formData = new FormData();

        formData.append("objective", text || "Analyze the attached files and perform the required operations.");

        attachedFiles.forEach((file) => {

            formData.append("files", file);

        });



        // Clear input and attachments

        const sendingFiles = [...attachedFiles];

        attachedFiles = [];

        renderAttachments();

        promptInput.value = "";

        promptInput.style.height = "auto";



        // Show thinking indicator

        const thinkingBubble = appendThinkingMessage();



        // Reset timeline for live execution

        traceTimeline.innerHTML = `<div class="trace-empty">🚀 Initializing multi-model orchestration stream...</div>`;

        telemetryCard.style.display = "none";



        try {

            // Attempt SSE Streaming first

            const streamRes = await fetch("/api/chat/stream", {

                method: "POST",

                body: formData

            });



            if (streamRes.ok && streamRes.body) {

                const reader = streamRes.body.getReader();

                const decoder = new TextDecoder();

                let buffer = "";

                let finalPayload = null;



                while (true) {

                    const { done, value } = await reader.read();

                    if (done) break;



                    buffer += decoder.decode(value, { stream: true });

                    const lines = buffer.split(/\r?\n/);

                    buffer = lines.pop() || "";



                    for (const line of lines) {

                        if (!line.startsWith("data: ")) continue;

                        try {

                            const packet = JSON.parse(line.slice(6));

                            if (packet.event === "done") {

                                finalPayload = packet.result;

                            } else if (packet.event === "error") {

                                throw new Error(packet.error || "Streaming error from backend");

                            } else {

                                handleLiveTraceEvent(packet);

                            }

                        } catch (parseErr) {

                            console.warn("SSE JSON parse notice:", parseErr, line);

                        }

                    }

                }



                thinkingBubble.remove();

                if (finalPayload) {

                    appendSageResponse(finalPayload.answer, finalPayload.telemetry, finalPayload.referenced_images);

                    updateTelemetryAndTrace(finalPayload.telemetry, finalPayload.trace);

                } else {

                    appendErrorMessage("Stream finished without generating a final response.");

                }

            } else {

                // Fallback to standard non-streaming endpoint

                const response = await fetch("/api/chat", {

                    method: "POST",

                    body: formData

                });



                const data = await response.json();

                thinkingBubble.remove();



                if (!response.ok || data.status === "error") {

                    appendErrorMessage(data.error || "An error occurred during multi-model orchestration.");

                    if (data.traceback) {

                        console.error("Backend Traceback:", data.traceback);

                    }

                } else {

                    appendSageResponse(data.answer, data.telemetry, data.referenced_images);

                    updateTelemetryAndTrace(data.telemetry, data.trace);

                }

            }

        } catch (err) {

            thinkingBubble.remove();

            appendErrorMessage(`Error: ${err.message}`);

        } finally {

            sendBtn.disabled = false;

            promptInput.disabled = false;

            sendBtn.innerHTML = `<span>Send</span><span class="send-icon">➤</span>`;

            promptInput.focus();

            updateStatus();

        }

    }



    function appendUserMessage(text, files) {

        const msgDiv = document.createElement("div");

        msgDiv.className = "chat-msg user";



        const header = document.createElement("div");

        header.className = "msg-header";

        header.textContent = "You";

        msgDiv.appendChild(header);



        const bubble = document.createElement("div");

        bubble.className = "msg-bubble";

        bubble.textContent = text;



        if (files && files.length > 0) {

            const attachContainer = document.createElement("div");

            attachContainer.className = "msg-attachments";

            files.forEach(f => {

                const chip = document.createElement("div");

                chip.className = "msg-file-chip";

                chip.textContent = `📎 ${f.name} (${(f.size/1024).toFixed(1)} KB)`;

                attachContainer.appendChild(chip);

            });

            bubble.appendChild(attachContainer);

        }



        msgDiv.appendChild(bubble);

        chatHistory.appendChild(msgDiv);

        chatHistory.scrollTop = chatHistory.scrollHeight;

    }



    function appendThinkingMessage() {

        const msgDiv = document.createElement("div");

        msgDiv.className = "chat-msg sage";



        const header = document.createElement("div");

        header.className = "msg-header";

        header.textContent = "SAGE Orchestrator";

        msgDiv.appendChild(header);



        const bubble = document.createElement("div");

        bubble.className = "msg-bubble";

        bubble.innerHTML = `<em>🧠 Gemma is reasoning, coordinating specialists, and switching models...</em>`;



        msgDiv.appendChild(bubble);

        chatHistory.appendChild(msgDiv);

        chatHistory.scrollTop = chatHistory.scrollHeight;

        return msgDiv;

    }



    function appendSageResponse(answer, telemetry, referencedImages) {

        const msgDiv = document.createElement("div");

        msgDiv.className = "chat-msg sage";



        const header = document.createElement("div");

        header.className = "msg-header";

        const timeStr = telemetry?.total_wall_time ? ` (${telemetry.total_wall_time.toFixed(1)}s)` : "";



        let pillsHtml = "";

        if (telemetry) {

            const pills = [];

            if (telemetry.agent_calls) {

                pills.push(`<span class="model-pill gemma">🧠 Gemma 4B (${telemetry.agent_calls})</span>`);

            }

            if (telemetry.document_calls) {

                pills.push(`<span class="model-pill doc">📄 Qwen3-VL (${telemetry.document_calls})</span>`);

            }

            if (telemetry.coder_calls) {

                pills.push(`<span class="model-pill coder">💻 Qwen2.5-Coder (${telemetry.coder_calls})</span>`);

            }

            if (telemetry.synthesizer_calls) {

                const sDur = telemetry.synthesizer_duration ? ` ${telemetry.synthesizer_duration.toFixed(1)}s` : "";

                pills.push(`<span class="model-pill qwen35">✨ Qwen3.5 2B${sDur}</span>`);

            }

            if (pills.length > 0) {

                pillsHtml = `<div class="msg-model-pills">${pills.join("")}</div>`;

            }

        }



        header.innerHTML = `<span>SAGE${escapeHtml(timeStr)}</span>${pillsHtml}`;

        msgDiv.appendChild(header);



        const bubble = document.createElement("div");

        bubble.className = "msg-bubble";

        bubble.innerHTML = formatMarkdown(answer);



        if (referencedImages && Array.isArray(referencedImages) && referencedImages.length > 0) {

            const gallery = document.createElement("div");

            gallery.className = "referenced-images-gallery";

            

            referencedImages.forEach(img => {

                const card = document.createElement("div");

                card.className = "artifact-image-card";

                card.innerHTML = `

                    <div class="artifact-image-header">

                        <span class="tag output">VISUAL ARTIFACT</span>

                        <span class="artifact-image-id">${escapeHtml(img.image_id || 'Image')}</span>

                    </div>

                    <a href="${escapeHtml(img.url)}" target="_blank" rel="noopener noreferrer" title="Click to view full image in a new tab">

                        <img src="${escapeHtml(img.url)}" alt="${escapeHtml(img.caption || img.image_id)}" loading="lazy" />

                    </a>

                    ${img.caption ? `<div class="artifact-image-caption">${escapeHtml(img.caption)}</div>` : ''}

                `;

                gallery.appendChild(card);

            });

            bubble.appendChild(gallery);

        }



        msgDiv.appendChild(bubble);

        chatHistory.appendChild(msgDiv);

        chatHistory.scrollTop = chatHistory.scrollHeight;

    }



    function appendErrorMessage(err) {

        const msgDiv = document.createElement("div");

        msgDiv.className = "chat-msg sage";



        const header = document.createElement("div");

        header.className = "msg-header";

        header.style.color = "var(--accent-red)";

        header.textContent = "SAGE Error";

        msgDiv.appendChild(header);



        const bubble = document.createElement("div");

        bubble.className = "msg-bubble";

        bubble.style.borderColor = "var(--accent-red)";

        bubble.innerHTML = `<strong>Error:</strong> <pre>${escapeHtml(err)}</pre>`;



        msgDiv.appendChild(bubble);

        chatHistory.appendChild(msgDiv);

        chatHistory.scrollTop = chatHistory.scrollHeight;

    }



    let activeLiveCard = null;



    function handleLiveTraceEvent(evt) {

        // Clear empty placeholder

        const placeholder = traceTimeline.querySelector(".trace-empty");

        if (placeholder) placeholder.remove();



        const type = evt.event;



        if (type === "model_invoking") {

            const isSynthesizer = evt.actor === "final_synthesizer";

            const actorLabel = isSynthesizer ? "✨ Qwen3.5 2B (Final Synthesizer)" : `🧠 Gemma 4B (Loop ${evt.loop})`;

            const actorClass = isSynthesizer ? "final_synthesizer" : "gemma";

            const statusColor = isSynthesizer ? "#d946ef" : "var(--accent-blue)";

            const statusText = isSynthesizer ? "Synthesizing response..." : "Invoking...";



            currentModelLabel.textContent = isSynthesizer

                ? "✨ Qwen3.5 2B Synthesizing Final Answer..."

                : `🧠 Gemma 4B Reasoning (Loop ${evt.loop})...`;



            const card = document.createElement("div");

            card.className = `trace-event ${actorClass} active`;



            const header = document.createElement("div");

            header.className = "trace-event-actor";

            header.innerHTML = `<span>${actorLabel}</span> <span style="color:${statusColor}">${statusText}</span>`;

            card.appendChild(header);



            const detail = document.createElement("div");

            detail.className = "trace-event-detail";

            detail.textContent = isSynthesizer

                ? "Faithfully rendering final response from controller brief and evidence..."

                : "Analyzing task and evaluating tool selection...";

            card.appendChild(detail);



            if (evt.input) {

                const details = document.createElement("details");

                details.className = "raw-turn";

                details.innerHTML = `

                    <summary>🔍 Inspect Turn (Input & Output)</summary>

                    <div style="margin-top:6px;"><span class="tag input">INPUT SENT</span><pre class="raw-box">${escapeHtml(evt.input)}</pre></div>

                    <div class="output-placeholder" style="margin-top:6px;"><span class="tag output">OUTPUT</span> <em style="font-size:11px;color:var(--text-muted)">Generating...</em></div>

                `;

                card.appendChild(details);

            }



            traceTimeline.appendChild(card);

            traceTimeline.scrollTop = traceTimeline.scrollHeight;

            activeLiveCard = card;



        } else if (type === "model_output") {

            if (activeLiveCard) {

                activeLiveCard.classList.remove("active");

                const timeSpan = activeLiveCard.querySelector(".trace-event-actor span:last-child");

                if (timeSpan) {

                    timeSpan.textContent = `(${evt.duration ? evt.duration.toFixed(2) : 0}s | ${evt.usage?.completion_tokens || 0} tok)`;

                    timeSpan.style.color = "var(--text-muted)";

                }



                const outWrap = activeLiveCard.querySelector(".output-placeholder");

                if (outWrap && evt.output) {

                    outWrap.innerHTML = `<span class="tag output">OUTPUT PRODUCED</span><pre class="raw-box">${escapeHtml(evt.output)}</pre>`;

                }

            }



        } else if (type === "tool_executing") {
            const isCoder = evt.tool === "code_specialist" || evt.tool === "coder";
            const isVision = evt.tool === "vision_ocr" || evt.tool === "vision";
            const isKnowledge = evt.tool === "general_knowledge" || evt.tool === "knowledge_specialist" || evt.tool === "general_chat";
            const actorName = isCoder ? "💻 Qwen2.5-Coder" : isVision ? "📄 Qwen3-VL" : isKnowledge ? "✨ Qwen3.5 2B (Knowledge Specialist)" : `⚙️ Tool: ${evt.tool}`;
            const actorClass = isCoder ? "coder" : isVision ? "document_analyzer" : isKnowledge ? "final_synthesizer" : "gemma";

            currentModelLabel.textContent = `⚡ Running ${actorName}...`;

            const card = document.createElement("div");
            card.className = `trace-event ${actorClass} active`;

            const header = document.createElement("div");
            header.className = "trace-event-actor";
            header.innerHTML = `<span>${actorName}</span> <span style="color:var(--accent-yellow)">Executing...</span>`;
            card.appendChild(header);

            const detail = document.createElement("div");
            detail.className = "trace-event-detail";
            detail.textContent = `Function: ${evt.function}`;
            card.appendChild(detail);

            if (evt.input) {
                const details = document.createElement("details");
                details.className = "raw-turn";
                const inStr = typeof evt.input === "object" ? JSON.stringify(evt.input, null, 2) : String(evt.input);
                details.innerHTML = `
                    <summary>🔍 Inspect Arguments & Output</summary>
                    <div style="margin-top:6px;"><span class="tag input">ARGUMENTS</span><pre class="raw-box">${escapeHtml(inStr)}</pre></div>
                    <div class="output-placeholder" style="margin-top:6px;"><span class="tag output">RESULT</span> <em style="font-size:11px;color:var(--text-muted)">${
                        evt.tool === "code_specialist" || evt.tool === "coder"
                            ? "Running in Docker sandbox..."
                            : evt.tool === "vision_ocr" || evt.tool === "vision"
                            ? "Analyzing with Qwen-VL..."
                            : isKnowledge
                            ? "Consulting Qwen3.5 2B knowledge specialist..."
                            : evt.tool === "document_database"
                            ? "Querying Document Database..."
                            : "Executing tool..."
                    }</em></div>
                `;
                card.appendChild(details);
            }
        } else if (type === "tool_result") {

            if (activeLiveCard) {

                activeLiveCard.classList.remove("active");

                const timeSpan = activeLiveCard.querySelector(".trace-event-actor span:last-child");

                if (timeSpan) {

                    const durStr = evt.duration_ms ? `${evt.duration_ms.toFixed(0)}ms` : "Done";

                    timeSpan.textContent = `[${(evt.status || "OK").toUpperCase()}] (${durStr})`;

                    timeSpan.style.color = evt.status === "error" ? "var(--accent-red)" : "var(--accent-green)";

                }



                const outWrap = activeLiveCard.querySelector(".output-placeholder");

                if (outWrap && evt.output) {

                    const outStr = typeof evt.output === "object" ? JSON.stringify(evt.output, null, 2) : String(evt.output);

                    outWrap.innerHTML = `<span class="tag output">RESULT</span><pre class="raw-box">${escapeHtml(outStr)}</pre>`;

                }

            }



        } else if (type === "final_synthesis") {

            currentModelLabel.textContent = "✨ SAGE Response Synthesized";

        } else if (type === "run_complete") {

            if (evt.telemetry) {

                updateTelemetryMetrics(evt.telemetry);

            }

        }

    }



    function updateTelemetryMetrics(telemetry) {

        if (telemetry) {

            telemetryCard.style.display = "block";

            metricAgentCalls.textContent = telemetry.agent_calls || 0;

            metricDocCalls.textContent = telemetry.document_calls || 0;

            metricCoderCalls.textContent = telemetry.coder_calls || 0;

            if (metricSynthesizerCalls) {

                metricSynthesizerCalls.textContent = telemetry.synthesizer_calls || 0;

            }

            if (metricSynthTime) {

                metricSynthTime.textContent = telemetry.synthesizer_duration ? `${telemetry.synthesizer_duration.toFixed(2)}s` : "0.00s";

            }

            metricSwitches.textContent = telemetry.model_switches || 0;

            metricWallTime.textContent = `${telemetry.total_wall_time.toFixed(2)}s`;

        }

    }



    function updateTelemetryAndTrace(telemetry, trace) {

        updateTelemetryMetrics(telemetry);



        if (trace && trace.length > 0) {

            traceTimeline.innerHTML = "";

            trace.forEach((evt) => {

                const evtDiv = document.createElement("div");

                evtDiv.className = `trace-event ${evt.actor}`;



                const actorHeader = document.createElement("div");

                actorHeader.className = "trace-event-actor";

                                const actorName = evt.actor === "gemma" ? "🧠 Gemma 4B" :
                                  evt.actor === "final_synthesizer" ? (evt.action === "general_knowledge" ? "✨ Qwen3.5 2B (Knowledge)" : "✨ Qwen3.5 2B") :
                                  evt.actor === "document_analyzer" ? "📄 Qwen3-VL" :
                                  evt.actor === "coder" ? "💻 Qwen2.5-Coder" :
                                  evt.actor === "sandbox" ? "📦 Docker Sandbox" : evt.actor;
                
                const timeInfo = evt.duration ? `(${evt.duration.toFixed(1)}s)` :
                                 evt.wall_time_ms ? `(${evt.wall_time_ms.toFixed(0)}ms)` :
                                 evt.duration_ms ? `(${evt.duration_ms.toFixed(0)}ms)` : "";
                actorHeader.innerHTML = `<span>${actorName}</span> <span style="font-weight:normal;color:var(--text-muted)">${timeInfo}</span>`;
                evtDiv.appendChild(actorHeader);

                const detail = document.createElement("div");
                detail.className = "trace-event-detail";
                if (evt.action === "requested_tools") {
                    detail.textContent = `Requested: ${evt.calls.map(c => c.tool).join(", ")}`;
                } else if (evt.action === "code_generated") {
                    detail.textContent = `Generated Code: ${evt.snippet}`;
                } else if (evt.action === "analyzed_content") {
                    detail.textContent = `Analyzed: ${evt.input}\nResult: ${evt.snippet}`;
                } else if (evt.action === "executed_code") {
                    const statusTag = evt.status ? evt.status.toUpperCase() : "DONE";
                    const stdoutSnippet = evt.stdout_preview ? `\nStdout: ${evt.stdout_preview}` : "";
                    detail.textContent = `[${statusTag}] Exit ${evt.exit_code} | Attempts: ${evt.attempts}${stdoutSnippet}`;
                } else if (evt.action === "general_knowledge") {
                    detail.textContent = `Knowledge Query: ${evt.query || evt.input || ""}\nAnswer: ${evt.answer_preview || ""}`;
                } else if (evt.action === "final_synthesis") {
                    detail.textContent = `Synthesized final answer for user`;
                } else {
                    detail.textContent = JSON.stringify(evt);
                }
                evtDiv.appendChild(detail);



                // Add collapsible inspector if input or output is available

                if (evt.input || evt.output) {

                    const details = document.createElement("details");

                    details.className = "raw-turn";

                    let contentHtml = "<summary>🔍 Inspect Turn (Input & Output)</summary>";

                    if (evt.input) {

                        const inStr = typeof evt.input === "object" ? JSON.stringify(evt.input, null, 2) : String(evt.input);

                        contentHtml += `<div style="margin-top:6px;"><span class="tag input">INPUT</span><pre class="raw-box">${escapeHtml(inStr)}</pre></div>`;

                    }

                    if (evt.output) {

                        const outStr = typeof evt.output === "object" ? JSON.stringify(evt.output, null, 2) : String(evt.output);

                        contentHtml += `<div style="margin-top:6px;"><span class="tag output">OUTPUT</span><pre class="raw-box">${escapeHtml(outStr)}</pre></div>`;

                    }

                    details.innerHTML = contentHtml;

                    evtDiv.appendChild(details);

                }



                traceTimeline.appendChild(evtDiv);

            });

        }

    }



    function escapeHtml(text) {

        return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

    }



    function formatMarkdown(text) {

        if (!text) return "";

        let formatted = escapeHtml(text);



        // Code blocks

        formatted = formatted.replace(/```([a-zA-Z0-9_]*)\n([\s\S]*?)```/g, (match, lang, code) => {

            return `<pre><code class="language-${lang}">${code}</code></pre>`;

        });



        // Inline code

        formatted = formatted.replace(/`([^`]+)`/g, '<code>$1</code>');



        // Bold

        formatted = formatted.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');



        // Headers

        formatted = formatted.replace(/^### (.*$)/gim, '<h3>$1</h3>');

        formatted = formatted.replace(/^## (.*$)/gim, '<h2>$1</h2>');

        formatted = formatted.replace(/^# (.*$)/gim, '<h1>$1</h1>');



        // Markdown Images: ![alt](url)

        formatted = formatted.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, (match, alt, url) => {

            return `<div class="inline-image-card">

                <a href="${url}" target="_blank" rel="noopener noreferrer" title="Click to view full image">

                    <img src="${url}" alt="${alt}" loading="lazy" />

                </a>

                <span class="inline-image-caption">${alt || 'Referenced Document Image'}</span>

            </div>`;

        });



        // Line breaks

        formatted = formatted.replace(/\n/g, '<br>');



        // Fix pre tags breaks

        formatted = formatted.replace(/<pre><code.*?>[\s\S]*?<\/code><\/pre>/g, (match) => {

            return match.replace(/<br>/g, '\n');

        });



        return formatted;

    }

});

