// SAGE — v2 UI Logic
document.addEventListener('DOMContentLoaded', () => {

    // ── DOM ────────────────────────────────────────────
    const chatMain      = document.getElementById('chatMain');
    const chatBody      = document.getElementById('chatBody');
    const chatMessages  = document.getElementById('chatMessages');
    const chatInputArea = document.getElementById('chatInputArea');
    const promptInput   = document.getElementById('promptInput');
    const sendBtn       = document.getElementById('sendBtn');
    const stopBtn       = document.getElementById('stopBtn');
    const attachBtn     = document.getElementById('attachBtn');
    const fileInput     = document.getElementById('fileInput');
    const attTray       = document.getElementById('attachmentsTray');
    const dropZone      = document.getElementById('dropZoneOverlay');
    const welcomeOverlay= document.getElementById('welcomeOverlay');
    const newChatBtn    = document.getElementById('newChatBtn');
    const searchInput   = document.getElementById('searchInput');

    // Sidebar left
    const sidebarLeft   = document.getElementById('sidebarLeft');
    const closeLeftBtn  = document.getElementById('closeLeftBtn');
    const openLeftBtn   = document.getElementById('openLeftBtn');
    const leftDragHandle= document.getElementById('leftDragHandle');

    // Panel right
    const panelRight    = document.getElementById('panelRight');
    const closeRightBtn = document.getElementById('closeRightBtn');
    const openRightBtn  = document.getElementById('openRightBtn');
    const rightDragHandle=document.getElementById('rightDragHandle');

    // Settings
    const settingsBtn      = document.getElementById('settingsBtn');
    const settingsPanel    = document.getElementById('settingsPanel');
    const closeSettingsBtn = document.getElementById('closeSettingsBtn');
    const openFlashRuntimeBtn = document.getElementById('openFlashRuntimeBtn');
    const flashRuntimeModal = document.getElementById('flashRuntimeModal');
    const flashRuntimeBackdrop = document.getElementById('flashRuntimeBackdrop');
    const closeFlashRuntimeBtn = document.getElementById('closeFlashRuntimeBtn');
    const flashModeBtn = document.getElementById('flashModeBtn');
    const reasoningModeBtn = document.getElementById('reasoningModeBtn');
    const resetChatInlineBtn = document.getElementById('resetChatInlineBtn');

    // Network status popup elements
    const netStatusBtn     = document.getElementById('netStatusBtn');
    const netStatusCard    = document.getElementById('netStatusCard');
    const closeNetCardBtn  = document.getElementById('closeNetCardBtn');

    // Right panel execution stream elements
    const execStatusDot   = document.getElementById('execStatusDot');
    const execStatusText  = document.getElementById('execStatusText');
    const execTilesTrack  = document.getElementById('execTilesTrack');
    const execEmptyState  = document.getElementById('execEmptyState');
    const edgeBlurTop     = document.getElementById('edgeBlurTop');
    const edgeBlurBottom  = document.getElementById('edgeBlurBottom');

    // Live Observer
    const observerLaunchBtn = document.getElementById('observerLaunchBtn');
    const observerEventCount = document.getElementById('observerEventCount');
    const observerPopover = document.getElementById('observerPopover');
    const observerPopoverStatus = document.getElementById('observerPopoverStatus');
    const observerMiniEvents = document.getElementById('observerMiniEvents');
    const observerExpandBtn = document.getElementById('observerExpandBtn');
    const observerModal = document.getElementById('observerModal');
    const observerModalBackdrop = document.getElementById('observerModalBackdrop');
    const observerCloseBtn = document.getElementById('observerCloseBtn');
    const observerExportBtn = document.getElementById('observerExportBtn');
    const observerRunLabel = document.getElementById('observerRunLabel');
    const observerTimeline = document.getElementById('observerTimeline');
    const observerDetailEmpty = document.getElementById('observerDetailEmpty');
    const observerDetailContent = document.getElementById('observerDetailContent');
    const observerDetailStep = document.getElementById('observerDetailStep');
    const observerDetailTitle = document.getElementById('observerDetailTitle');
    const observerDetailMeta = document.getElementById('observerDetailMeta');
    const observerDetailSummary = document.getElementById('observerDetailSummary');
    const observerDetailSections = document.getElementById('observerDetailSections');
    const observerDetailJson = document.getElementById('observerDetailJson');

    // In-site Document Preview Modal elements
    const docPreviewModal = document.getElementById('docPreviewModal');
    const dpmBackdrop     = document.getElementById('dpmBackdrop');
    const dpmCloseBtn     = document.getElementById('dpmCloseBtn');
    const dpmDownloadBtn  = document.getElementById('dpmDownloadBtn');
    const dpmFileIcon     = document.getElementById('dpmFileIcon');
    const dpmFileName     = document.getElementById('dpmFileName');
    const dpmFileMeta     = document.getElementById('dpmFileMeta');
    const dpmBody         = document.getElementById('dpmBody');
    const dpmLoading      = document.getElementById('dpmLoading');
    const dpmLoadingText  = document.getElementById('dpmLoadingText');

    // Chat continuity state (Phase 1)
    let currentChatId = null;
    const dpmContent      = document.getElementById('dpmContent');

    let currentPreviewFile = null;
    let currentPreviewBlobUrl = null;

    let attachedFiles = [];
    let isRunning     = false;
    let chatActive    = false;  // has the input moved to the bottom yet?
    let activeChatMode = 'flash';
    let flashSessionId = (crypto.randomUUID?.() || ('flash_' + Date.now()));
    let observerSource = null;
    let observerRunId = null;
    let observerEvents = [];
    let observerSelectedSequence = null;
    let observerSelectionPinned = false;
    const observerTiles = new Map();

    function observerProgressSpec(event) {
        const actor = String(event.actor || '').toLowerCase();
        const phase = String(event.phase || '').toLowerCase();
        if (actor === 'sage' && phase === 'request') {
            return { key: 'request', label: 'SAGE', actorClass: 'gemma', action: event.status === 'completed' ? 'Response ready' : 'Understanding your request', detail: event.status === 'completed' ? 'The answer has been prepared.' : 'Preparing the conversation and attachments.' };
        }
        if (actor === 'memory' && phase === 'recent_context') {
            return { key: 'recent-context', label: 'Conversation', actorClass: 'knowledge', action: 'Checking recent context', detail: 'Using the most relevant recent messages.' };
        }
        if (actor === 'gemma' && ['model_input', 'model_output'].includes(phase)) {
            return { key: 'routing', label: 'SAGE', actorClass: 'gemma', action: 'Choosing the best response path', detail: event.status === 'completed' ? 'The response approach is ready.' : 'Deciding whether vision or memory is needed.' };
        }
        if (actor === 'memory' && phase === 'semantic_search') {
            return { key: 'memory-search', label: 'Memory', actorClass: 'knowledge', action: 'Looking through relevant memory', detail: event.status === 'completed' ? 'Relevant past context has been checked.' : 'Searching only when earlier context may help.' };
        }
        if (actor === 'qwen' && ['model_input', 'model_output'].includes(phase)) {
            return { key: 'vision', label: 'Vision', actorClass: 'vision', action: 'Reading the attached image', detail: event.status === 'completed' ? 'Visual details have been extracted.' : 'Inspecting the image for grounded details.' };
        }
        if (actor === 'gemma' && ['synthesis_input', 'synthesis_output'].includes(phase)) {
            return { key: 'synthesis', label: 'SAGE', actorClass: 'gemma', action: 'Putting the answer together', detail: event.status === 'completed' ? 'The findings have been combined.' : 'Combining the request with the gathered evidence.' };
        }
        if (actor === 'postgres' && phase === 'conversation_commit') {
            return { key: 'save-chat', label: 'Conversation', actorClass: 'knowledge', action: 'Saving this conversation', detail: event.status === 'failed' ? 'The chat remains available, but memory could not be saved.' : 'This exchange is now available to recent context.' };
        }
        if (actor === 'memory-2b' && ['curation_queue', 'curation'].includes(phase)) {
            return { key: `memory-curation:${event.payload?.job_type || event.payload?.job_id || 'current'}`, label: 'Memory', actorClass: 'knowledge', action: 'Updating memory in the background', detail: event.status === 'failed' ? 'Memory processing needs attention. Open Observer for details.' : event.status === 'completed' ? 'Useful details were organized for later recall.' : 'Organizing useful details without delaying your answer.' };
        }
        return null;
    }

    function mirrorObserverEventToPanel(event) {
        const spec = observerProgressSpec(event);
        if (!spec) return;
        const location = event.payload?.provider === 'remote' ? 'Remote' : 'On this device';
        const duration = event.duration_ms != null ? `${(Number(event.duration_ms) / 1000).toFixed(1)}s` : undefined;
        if (['started', 'queued', 'running'].includes(event.status)) {
            if (!observerTiles.has(spec.key)) {
                observerTiles.set(spec.key, spawnExecTile({
                    id: `obs_${event.sequence}`, actor: spec.label, actorClass: spec.actorClass,
                    action: spec.action, detail: spec.detail, status: 'running', location,
                    observerSequence: event.sequence
                }));
            } else {
                observerTiles.get(spec.key).dataset.observerSequence = String(event.sequence);
            }
            if (spec.key.startsWith('memory-curation') && !isRunning) rightPanelDone();
            return;
        }
        const existing = observerTiles.get(spec.key);
        if (existing) {
            existing.dataset.observerSequence = String(event.sequence);
            completeExecTile(existing, spec.detail, duration, event.status);
            observerTiles.delete(spec.key);
        } else {
            spawnExecTile({
                id: `obs_${event.sequence}`, actor: spec.label, actorClass: spec.actorClass,
                action: spec.action, detail: spec.detail,
                status: event.status === 'failed' ? 'failed' : 'done', duration, location,
                observerSequence: event.sequence
            });
        }
        if (spec.key.startsWith('memory-curation') && !isRunning) rightPanelDone();
    }

    function observerEventTitle(event) {
        return String(event.phase || 'event').replaceAll('_', ' ').replace(/\b\w/g, letter => letter.toUpperCase());
    }

    function observerValue(value) {
        return typeof value === 'string' ? value : JSON.stringify(value, null, 2);
    }

    function appendObserverSection(title, value, tone = '') {
        if (!observerDetailSections || value === undefined || value === null || value === '') return;
        const section = document.createElement('section');
        section.className = `observer-io-card ${tone}`.trim();
        const heading = document.createElement('h4');
        heading.textContent = title;
        const pre = document.createElement('pre');
        pre.textContent = observerValue(value);
        section.append(heading, pre);
        observerDetailSections.appendChild(section);
    }

    function renderObserverDetails(event) {
        const payload = event.payload || {};
        if (observerDetailStep) observerDetailStep.textContent = `Event #${event.sequence}`;
        if (observerDetailTitle) observerDetailTitle.textContent = observerEventTitle(event);
        if (observerDetailSummary) observerDetailSummary.textContent = event.summary || '';
        if (observerDetailMeta) {
            const time = new Date(Number(event.timestamp || 0) * 1000).toLocaleTimeString();
            const duration = event.duration_ms != null ? ` · ${Number(event.duration_ms).toFixed(1)} ms` : '';
            observerDetailMeta.textContent = `${event.actor} · ${event.status} · ${time}${duration}`;
        }
        if (observerDetailSections) {
            observerDetailSections.innerHTML = '';
            const input = {};
            const output = {};
            const inputKeys = ['objective', 'messages', 'instruction', 'query', 'context', 'attachments', 'image_count', 'source_message_ids', 'input_chars', 'chunk_count', 'chunk_index'];
            const outputKeys = ['answer', 'route', 'flash_case', 'memories', 'candidate_ids', 'memory_ids', 'proposed', 'stored', 'memory_jobs'];
            inputKeys.forEach(key => { if (payload[key] !== undefined) input[key] = payload[key]; });
            outputKeys.forEach(key => { if (payload[key] !== undefined) output[key] = payload[key]; });
            if (payload.content !== undefined) {
                if (String(event.phase || '').includes('input')) input.content = payload.content;
                else output.content = payload.content;
            }
            const consumed = new Set([...inputKeys, ...outputKeys, 'content']);
            const metadata = Object.fromEntries(Object.entries(payload).filter(([key]) => !consumed.has(key)));
            appendObserverSection('Input', Object.keys(input).length ? input : null, 'input');
            appendObserverSection('Output', Object.keys(output).length ? output : null, 'output');
            appendObserverSection('Routing, timing & metadata', Object.keys(metadata).length ? metadata : null, 'metadata');
            if (!observerDetailSections.children.length) appendObserverSection('Event data', payload, 'metadata');
        }
        if (observerDetailJson) observerDetailJson.textContent = JSON.stringify(event, null, 2);
    }

    function inspectObserverEvent(event, button, { pin = true } = {}) {
        observerSelectedSequence = Number(event.sequence);
        if (pin) observerSelectionPinned = true;
        observerTimeline?.querySelectorAll('.observer-event').forEach(item => item.classList.remove('active'));
        const target = button || observerTimeline?.querySelector(`[data-sequence="${observerSelectedSequence}"]`);
        target?.classList.add('active');
        if (observerDetailEmpty) observerDetailEmpty.hidden = true;
        if (observerDetailContent) observerDetailContent.hidden = false;
        renderObserverDetails(event);
    }

    function renderObserver() {
        if (observerEventCount) observerEventCount.textContent = String(observerEvents.length);
        const last = observerEvents[observerEvents.length - 1];
        if (observerPopoverStatus) observerPopoverStatus.textContent = last ? `${last.actor}: ${last.summary}` : 'Waiting for a run…';
        if (observerRunLabel) observerRunLabel.textContent = observerRunId ? `Run ${observerRunId} · ${observerEvents.length} events` : 'No active run';
        if (observerMiniEvents) {
            observerMiniEvents.innerHTML = '';
            observerEvents.slice(-7).reverse().forEach(event => {
                const row = document.createElement('div');
                row.className = `observer-mini-event ${event.status}`;
                row.innerHTML = `<i></i><span><strong>${esc(event.actor)} · ${esc(event.phase)}</strong><br>${esc(event.summary)}</span><small>#${event.sequence}</small>`;
                observerMiniEvents.appendChild(row);
            });
        }
        if (observerTimeline) {
            observerTimeline.innerHTML = '';
            observerEvents.forEach(event => {
                const button = document.createElement('button');
                button.type = 'button';
                button.className = 'observer-event';
                button.dataset.sequence = String(event.sequence);
                if (Number(event.sequence) === observerSelectedSequence) button.classList.add('active');
                const time = new Date(Number(event.timestamp || 0) * 1000).toLocaleTimeString();
                button.innerHTML = `<span class="observer-event-seq">#${event.sequence}</span><span><strong>${esc(event.actor)} · ${esc(event.phase)}</strong><span>${esc(event.summary)}</span></span><time>${esc(time)}</time>`;
                observerTimeline.appendChild(button);
            });
            if (!observerSelectionPinned) observerTimeline.scrollTop = observerTimeline.scrollHeight;
        }
        const selected = observerEvents.find(event => Number(event.sequence) === observerSelectedSequence);
        if (selected) renderObserverDetails(selected);
        else if (!observerSelectionPinned && observerEvents.length && observerModal && !observerModal.hidden) {
            inspectObserverEvent(observerEvents[observerEvents.length - 1], null, { pin: false });
        }
    }

    observerTimeline?.addEventListener('click', event => {
        const button = event.target.closest('.observer-event');
        if (!button || !observerTimeline.contains(button)) return;
        const selected = observerEvents.find(item => Number(item.sequence) === Number(button.dataset.sequence));
        if (selected) inspectObserverEvent(selected, button);
    });

    async function hydrateObserverSnapshot() {
        if (!observerRunId) return;
        try {
            const response = await fetch(`/api/observer/runs/${encodeURIComponent(observerRunId)}`);
            if (!response.ok) return;
            const data = await response.json();
            (data.events || []).forEach(event => {
                if (!observerEvents.some(item => item.sequence === event.sequence)) {
                    observerEvents.push(event);
                    mirrorObserverEventToPanel(event);
                }
            });
            observerEvents.sort((a, b) => Number(a.sequence) - Number(b.sequence));
            renderObserver();
        } catch (error) {
            console.warn('[Observer] Snapshot unavailable', error);
        }
    }

    function startObserver(runId) {
        if (!runId) return;
        observerSource?.close();
        observerRunId = runId;
        observerEvents = [];
        observerSelectedSequence = null;
        observerSelectionPinned = false;
        observerTiles.clear();
        if (observerDetailEmpty) observerDetailEmpty.hidden = false;
        if (observerDetailContent) observerDetailContent.hidden = true;
        renderObserver();
        observerLaunchBtn?.classList.add('live');
        observerSource = new EventSource(`/api/observer/stream/${encodeURIComponent(runId)}`);
        observerSource.addEventListener('observation', event => {
            try {
                const payload = JSON.parse(event.data);
                if (!observerEvents.some(item => item.sequence === payload.sequence)) {
                    observerEvents.push(payload);
                    mirrorObserverEventToPanel(payload);
                }
                renderObserver();
            } catch (error) { console.warn('[Observer] Invalid event', error); }
        });
        observerSource.onerror = () => {
            if (observerPopoverStatus) observerPopoverStatus.textContent = 'Live stream reconnecting…';
        };
    }

    function setObserverModal(open) {
        if (!observerModal) return;
        observerModal.hidden = !open;
        observerModal.setAttribute('aria-hidden', open ? 'false' : 'true');
        if (open) {
            observerPopover.hidden = true;
            hydrateObserverSnapshot();
            renderObserver();
        }
    }

    observerLaunchBtn?.addEventListener('click', event => { event.stopPropagation(); observerPopover.hidden = !observerPopover.hidden; });
    observerExpandBtn?.addEventListener('click', () => setObserverModal(true));
    observerCloseBtn?.addEventListener('click', () => setObserverModal(false));
    observerModalBackdrop?.addEventListener('click', () => setObserverModal(false));
    observerExportBtn?.addEventListener('click', () => {
        const blob = new Blob([JSON.stringify({ run_id: observerRunId, events: observerEvents }, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a'); link.href = url; link.download = `sage-observer-${observerRunId || 'run'}.json`; link.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
    });
    if (window.location.hash === '#observer') setObserverModal(true);

    // ══════════════════════════════════════════════════
    // STATUS POLL (Sync backend model configs & health)
    // ══════════════════════════════════════════════════
    async function updateStatus() {
        try {
            const res = await fetch('/api/status');
            if (res.ok) {
                const data = await res.json();
                if (data.gemma_context) {
                    const ctxNum = Number(data.gemma_context);
                    const formatted = ctxNum.toLocaleString();
                    const gemma = INSTALLED_MODELS.find(m => m.id === 'gemma-4b');
                    if (gemma) {
                        gemma.context = `${formatted} tokens`;
                    }
                    const reasoningMeta = document.getElementById('tmSlotReasoningMeta');
                    if (reasoningMeta) {
                        reasoningMeta.innerHTML = `Format: GGUF Q4_K_M &middot; Context: ${formatted} &middot; VRAM: ~2.9 GB`;
                    }
                }
                if (data.model_configs) {
                    Object.entries(data.model_configs).forEach(([key, cfg]) => {
                        if (cfg.context) {
                            const formatted = Number(cfg.context).toLocaleString();
                            if (key === 'coder') {
                                const m = INSTALLED_MODELS.find(x => x.id === 'qwen2.5-coder-7b');
                                if (m) m.context = `${formatted} tokens`;
                            } else if (key === 'document_analyzer') {
                                const m = INSTALLED_MODELS.find(x => x.id === 'qwen3-vl-4b');
                                if (m) m.context = `${formatted} tokens`;
                            }
                        }
                    });
                }
            }
        } catch {}
    }
    updateStatus();
    setInterval(updateStatus, 8000);

    // ══════════════════════════════════════════════════
    // SIDEBAR COLLAPSE / EXPAND
    // ══════════════════════════════════════════════════
    function collapseSidebar() {
        sidebarLeft.style.width = '';
        sidebarLeft.classList.add('collapsed');
        openLeftBtn.classList.add('show');
        if (chatActive) positionInput(false);
    }
    function expandSidebar() {
        sidebarLeft.style.width = '';
        sidebarLeft.classList.remove('collapsed');
        openLeftBtn.classList.remove('show');
        if (chatActive) setTimeout(() => positionInput(false), 280);
    }
    closeLeftBtn?.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        collapseSidebar();
    });
    openLeftBtn?.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        expandSidebar();
    });

    function collapsePanel() {
        panelRight.style.width = '';
        panelRight.classList.add('collapsed');
        openRightBtn.classList.add('show');
        if (chatActive) positionInput(false);
    }
    function expandPanel() {
        panelRight.style.width = '';
        panelRight.classList.remove('collapsed');
        openRightBtn.classList.remove('show');
        if (chatActive) setTimeout(() => positionInput(false), 280);
    }
    closeRightBtn?.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        collapsePanel();
    });
    openRightBtn?.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        expandPanel();
    });

    // ══════════════════════════════════════════════════
    // DRAG RESIZE (manual dynamic resize, max stretch reduced)
    // ══════════════════════════════════════════════════
    function setupDrag(handle, panel, dir) {
        if (!handle || !panel) return;
        let startX, startW;
        handle.addEventListener('mousedown', e => {
            startX = e.clientX;
            startW = panel.offsetWidth;
            document.body.style.cursor = 'col-resize';
            document.body.style.userSelect = 'none';
            document.addEventListener('mousemove', onMove);
            document.addEventListener('mouseup', onUp);
            e.preventDefault();
        });
        function onMove(e) {
            const delta = dir === 'left' ? e.clientX - startX : startX - e.clientX;
            const maxW  = dir === 'left' ? 300 : 310;
            const newW  = Math.max(0, Math.min(startW + delta, maxW));
            // Use inline width (overrides CSS var during drag)
            panel.style.width = newW + 'px';
            // Show/hide reveal button
            if (dir === 'left') {
                const isCollapsed = newW < 55;
                panel.classList.toggle('collapsed', isCollapsed);
                openLeftBtn.classList.toggle('show', isCollapsed);
            } else {
                const isCollapsed = newW < 55;
                panel.classList.toggle('collapsed', isCollapsed);
                openRightBtn.classList.toggle('show', isCollapsed);
            }
            if (chatActive) positionInput(false);
        }
        function onUp() {
            document.removeEventListener('mousemove', onMove);
            document.removeEventListener('mouseup', onUp);
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
        }
    }
    setupDrag(leftDragHandle,  sidebarLeft, 'left');
    setupDrag(rightDragHandle, panelRight,  'right');

    // ══════════════════════════════════════════════════
    // SETTINGS PANEL
    // ══════════════════════════════════════════════════
    const SETTINGS_KEY = 'sage_ui_settings_v1';
    const settingTheme = document.getElementById('settingTheme');
    const settingFontSize = document.getElementById('settingFontSize');
    const settingDefaultModel = document.getElementById('settingDefaultModel');
    const settingTemperature = document.getElementById('settingTemperature');
    const settingSaveHistory = document.getElementById('settingSaveHistory');
    const settingTelemetry = document.getElementById('settingTelemetry');

    const uiSettings = {
        theme: 'light',
        fontSize: 'medium',
        defaultModel: 'auto',
        temperature: '0.7',
        saveHistory: true,
        telemetry: false,
    };

    function loadUiSettings() {
        try {
            const raw = localStorage.getItem(SETTINGS_KEY);
            if (!raw) return;
            const parsed = JSON.parse(raw);
            Object.assign(uiSettings, parsed || {});
        } catch { /* keep defaults */ }
    }

    function persistUiSettings() {
        localStorage.setItem(SETTINGS_KEY, JSON.stringify(uiSettings));
    }

    function resolveTheme(theme) {
        if (theme === 'system') {
            return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
        }
        return theme === 'dark' ? 'dark' : 'light';
    }

    function applyUiSettings() {
        document.documentElement.setAttribute('data-theme', resolveTheme(uiSettings.theme));
        document.documentElement.setAttribute('data-font-size', uiSettings.fontSize || 'medium');
        if (settingTheme) settingTheme.value = uiSettings.theme;
        if (settingFontSize) settingFontSize.value = uiSettings.fontSize;
        if (settingDefaultModel) settingDefaultModel.value = uiSettings.defaultModel;
        if (settingTemperature) settingTemperature.value = String(uiSettings.temperature);
        if (settingSaveHistory) settingSaveHistory.checked = !!uiSettings.saveHistory;
        if (settingTelemetry) settingTelemetry.checked = !!uiSettings.telemetry;
    }

    function syncSetting(key, value) {
        uiSettings[key] = value;
        persistUiSettings();
        applyUiSettings();
    }

    loadUiSettings();
    applyUiSettings();
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
        if (uiSettings.theme === 'system') applyUiSettings();
    });

    settingTheme?.addEventListener('change', () => syncSetting('theme', settingTheme.value));
    settingFontSize?.addEventListener('change', () => syncSetting('fontSize', settingFontSize.value));
    settingDefaultModel?.addEventListener('change', () => syncSetting('defaultModel', settingDefaultModel.value));
    settingTemperature?.addEventListener('change', () => syncSetting('temperature', settingTemperature.value));
    settingSaveHistory?.addEventListener('change', () => syncSetting('saveHistory', !!settingSaveHistory.checked));
    settingTelemetry?.addEventListener('change', () => syncSetting('telemetry', !!settingTelemetry.checked));

    // Settings opens the appearance/privacy drawer. Flash runtime has its own button.
    settingsBtn?.addEventListener('click', () => settingsPanel?.classList.toggle('open'));
    closeSettingsBtn?.addEventListener('click', () => settingsPanel.classList.remove('open'));
    if (window.location.hash === '#settings') settingsPanel?.classList.add('open');

    // FLASH RUNTIME CONTROL CENTER
    const flashPrimaryRemoteFields = document.getElementById('flashPrimaryRemoteFields');
    const flashRuntimeStatus = document.getElementById('flashRuntimeStatus');
    const flashApplyBtn = document.getElementById('flashApplyBtn');
    const flashTestBtn = document.getElementById('flashTestBtn');
    const flashDeployBtn = document.getElementById('flashDeployBtn');

    function flashStatus(kind, title, detail) {
        if (!flashRuntimeStatus) return;
        flashRuntimeStatus.dataset.kind = kind;
        const titleEl = flashRuntimeStatus.querySelector('strong');
        const detailEl = flashRuntimeStatus.querySelector('small');
        if (titleEl) titleEl.textContent = title;
        if (detailEl) detailEl.textContent = detail;
    }

    function renderFlashDiagnostics(roles, heading = '') {
        const panel = document.getElementById('flashDiagnostics');
        const list = document.getElementById('flashDiagnosticsList');
        if (!panel || !list) return;
        panel.hidden = false;
        list.innerHTML = '';
        if (heading) {
            const intro = document.createElement('div');
            intro.className = 'flash-diagnostic-row error';
            intro.innerHTML = `<span class="flash-diagnostic-role">Runtime</span><span class="flash-diagnostic-state">ERROR</span><span class="flash-diagnostic-copy">${esc(heading)}</span>`;
            list.appendChild(intro);
        }
        (roles || []).forEach(role => {
            const disabled = role.enabled === false;
            const healthy = !!role.healthy;
            const row = document.createElement('div');
            row.className = `flash-diagnostic-row ${healthy ? 'success' : 'error'}`;
            const detail = disabled
                ? 'Role is intentionally disabled.'
                : (role.error || `${role.provider || 'Endpoint'} is ready.`);
            const models = Array.isArray(role.models) && role.models.length
                ? `Models reported by server: ${role.models.join(', ')}`
                : 'No model IDs were reported by this role test.';
            const suggestion = role.suggested_model_id
                ? ` Auto-detected matching ID: ${role.suggested_model_id}.`
                : '';
            row.innerHTML = `
                <span class="flash-diagnostic-role">${esc(role.role || 'unknown')}</span>
                <span class="flash-diagnostic-state">${disabled ? 'DISABLED' : (healthy ? 'READY' : 'FAILED')}</span>
                <span class="flash-diagnostic-copy">${esc(detail + suggestion)}<small>${esc(models)}</small></span>
            `;
            list.appendChild(row);
        });
    }

    function useSuggestedFlashModelIds(roles) {
        const fields = {
            gemma: document.getElementById('flashGemmaModelId'),
            qwen: document.getElementById('flashQwenModelId'),
            memory: document.getElementById('flashMemoryModelId')
        };
        const changed = [];
        (roles || []).forEach(role => {
            if (role.suggested_model_id && fields[role.role]) {
                fields[role.role].value = role.suggested_model_id;
                changed.push(`${role.role} → ${role.suggested_model_id}`);
            }
        });
        return changed;
    }

    function primaryProvider() {
        return document.querySelector('input[name="flashPrimaryProvider"]:checked')?.value || 'local_gpu';
    }

    function memoryProvider() {
        return document.querySelector('input[name="flashMemoryProvider"]:checked')?.value || 'local_cpu';
    }

    function syncFlashRuntimeFields() {
        const remote = primaryProvider() === 'remote';
        const memoryRemote = memoryProvider() === 'remote';
        if (flashPrimaryRemoteFields) flashPrimaryRemoteFields.hidden = !remote;
        const memoryFields = document.getElementById('flashMemoryRemoteFields');
        if (memoryFields) memoryFields.hidden = !memoryRemote;
        document.querySelectorAll('#flashPrimaryChoices .flash-choice').forEach(choice => {
            choice.classList.toggle('active', !!choice.querySelector('input:checked'));
        });
        document.querySelectorAll('#flashMemoryChoices .flash-choice').forEach(choice => {
            choice.classList.toggle('active', !!choice.querySelector('input:checked'));
        });
        const deployTarget = document.getElementById('flashDeployTarget');
        const primaryOption = deployTarget?.querySelector('option[value="primary"]');
        const memoryOption = deployTarget?.querySelector('option[value="memory"]');
        if (primaryOption) primaryOption.disabled = !remote;
        if (memoryOption) memoryOption.disabled = !memoryRemote;
        if (deployTarget && deployTarget.selectedOptions[0]?.disabled) {
            deployTarget.value = remote ? 'primary' : (memoryRemote ? 'memory' : 'primary');
        }
        if (flashDeployBtn) {
            flashDeployBtn.disabled = !remote && !memoryRemote;
            flashDeployBtn.title = flashDeployBtn.disabled ? 'Choose a remote endpoint to deploy through its bridge.' : '';
        }
    }

    function setFlashModal(open) {
        if (!flashRuntimeModal) return;
        flashRuntimeModal.style.display = open ? 'flex' : 'none';
        flashRuntimeModal.setAttribute('aria-hidden', open ? 'false' : 'true');
        document.body.style.overflow = open ? 'hidden' : '';
        if (open) {
            settingsPanel?.classList.remove('open');
            loadFlashCatalog();
        }
    }

    async function loadFlashCatalog() {
        const note = document.getElementById('flashCatalogNote');
        try {
            const res = await fetch('/api/flash/catalog');
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || 'Catalog unavailable');
            const labels = Object.values(data.models || {}).map(model => `${model.role}: ${model.file}`).join('  •  ');
            if (note) note.textContent = labels || 'No catalog entries found.';
        } catch (error) {
            if (note) note.textContent = `Catalog error: ${error.message}`;
        }
    }

    function buildFlashRuntimePayload() {
        const primary = primaryProvider();
        const connections = [];
        const roles = {
            gemma: { provider: primary, connection_id: primary === 'remote' ? 'primary' : null, model_id: document.getElementById('flashGemmaModelId')?.value.trim() || 'gemma' },
            qwen: { provider: primary, connection_id: primary === 'remote' ? 'primary' : null, model_id: document.getElementById('flashQwenModelId')?.value.trim() || 'qwen' },
            memory: { provider: memoryProvider(), connection_id: memoryProvider() === 'remote' ? 'memory' : null, model_id: document.getElementById('flashMemoryModelId')?.value.trim() || 'memory' }
        };

        if (primary === 'remote') {
            const baseUrl = document.getElementById('flashPrimaryUrl')?.value.trim();
            if (!baseUrl) throw new Error('Enter the primary remote server URL.');
            connections.push({ id: 'primary', label: 'Primary inference server', base_url: baseUrl, api_key: document.getElementById('flashPrimaryKey')?.value || '' });
        }
        if (memoryProvider() === 'remote') {
            const baseUrl = document.getElementById('flashMemoryUrl')?.value.trim();
            if (!baseUrl) throw new Error('Enter the remote memory server URL.');
            connections.push({ id: 'memory', label: 'Memory curator server', base_url: baseUrl, api_key: document.getElementById('flashMemoryKey')?.value || '' });
        }
        return { connections, roles };
    }

    function updateNetworkDisclosure(payload) {
        const usesRemote = Object.values(payload.roles || {}).some(role => role.provider === 'remote');
        const badge = netStatusCard?.querySelector('.nsc-badge span:last-child');
        const desc = netStatusCard?.querySelector('.nsc-desc');
        const values = netStatusCard?.querySelectorAll('.nsc-stat-val');
        const footer = netStatusCard?.querySelector('.nsc-footer span');
        if (!badge || !desc || !values || values.length < 4 || !footer) return;
        if (usesRemote) {
            const hosts = (payload.connections || []).map(connection => {
                try { return new URL(connection.base_url).host; } catch { return 'configured server'; }
            }).join(', ');
            badge.textContent = 'PRIVATE REMOTE GPU ACTIVE';
            desc.textContent = 'Inference is routed only to the private endpoints configured for this running SAGE session.';
            values[0].textContent = 'Configured endpoint only';
            values[1].textContent = 'Controlled by your server';
            values[2].textContent = hosts || 'Private endpoint';
            values[3].textContent = 'User-configured';
            footer.textContent = 'URLs and keys clear when the SAGE server restarts';
        } else {
            badge.textContent = '100% LOCAL · ZERO LEAKAGE';
            desc.textContent = 'All agent steps, model reasoning, and attached documents run exclusively on your local machine.';
            values[0].textContent = 'Blocked / Air-Gapped';
            values[1].textContent = '0% (Completely Safe)';
            values[2].textContent = '127.0.0.1 (Localhost)';
            values[3].textContent = 'None / No API Keys';
            footer.textContent = 'Verified: Machine is fully air-gapped';
        }
    }

    async function applyFlashRuntime({ quiet = false } = {}) {
        const payload = buildFlashRuntimePayload();
        if (!quiet) flashStatus('working', 'Applying runtime', 'Updating ephemeral role bindings…');
        const res = await fetch('/api/flash/runtime', {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || data.error || 'Runtime configuration failed');
        updateNetworkDisclosure(payload);
        const summary = document.getElementById('flashSettingsSummary');
        if (summary) summary.textContent = primaryProvider() === 'remote' ? 'Remote GPU active for this session' : 'Local RTX active for this session';
        if (!quiet) flashStatus('success', 'Runtime applied', 'Configuration lives in memory only and clears when SAGE restarts.');
        refreshRuntimeReadiness();
        return data;
    }

    openFlashRuntimeBtn?.addEventListener('click', () => setFlashModal(true));
    closeFlashRuntimeBtn?.addEventListener('click', () => setFlashModal(false));
    flashRuntimeBackdrop?.addEventListener('click', () => setFlashModal(false));
    if (window.location.hash === '#flash-runtime') setFlashModal(true);
    document.querySelectorAll('input[name="flashPrimaryProvider"]').forEach(input => input.addEventListener('change', syncFlashRuntimeFields));
    document.querySelectorAll('input[name="flashMemoryProvider"]').forEach(input => input.addEventListener('change', syncFlashRuntimeFields));
    syncFlashRuntimeFields();

    async function refreshRuntimeReadiness() {
        let banner = document.getElementById('runtimeBanner');
        if (!banner) {
            banner = document.createElement('div');
            banner.id = 'runtimeBanner';
            banner.className = 'runtime-banner';
            chatMain?.appendChild(banner);
        }
        try {
            const res = await fetch('/api/flash/status');
            const data = await res.json();
            const readiness = data.readiness || {};
            const runtime = data.runtime || {};
            const roles = runtime.roles || {};
            const usesRemote = Object.values(roles).some(role => role && role.provider === 'remote');
            const summary = document.getElementById('flashSettingsSummary');
            if (readiness.mock_mode && !usesRemote) {
                if (summary) summary.textContent = 'Mock inference active (no local GGUF required)';
                banner.innerHTML = 'Running in mock mode so Flash works without local models. Configure a remote GPU or install llama-server + GGUF files for real inference. <button type="button" id="runtimeBannerOpen">Open runtime</button>';
                banner.classList.add('show');
            } else if (usesRemote) {
                banner.classList.remove('show');
                if (summary) summary.textContent = 'Remote GPU active for this session';
            } else if (!readiness.local_inference_ready && !runtime.configured) {
                if (summary) summary.textContent = 'Local models missing — configure runtime';
                const missing = (readiness.missing_model_files || []).slice(0, 2).join(', ');
                banner.innerHTML = `Local inference is not ready${missing ? ` (missing: ${esc(missing)})` : ''}. Open Settings → Flash runtime to use a remote GPU, or set LLAMA_SERVER_PATH and MODEL_DIR. <button type="button" id="runtimeBannerOpen">Open runtime</button>`;
                banner.classList.add('show');
            } else {
                banner.classList.remove('show');
                if (summary && runtime.configured) {
                    summary.textContent = 'Local RTX active for this session';
                }
            }
            document.getElementById('runtimeBannerOpen')?.addEventListener('click', () => setFlashModal(true));
        } catch {
            banner.classList.remove('show');
        }
    }
    refreshRuntimeReadiness();

    flashApplyBtn?.addEventListener('click', async () => {
        flashApplyBtn.disabled = true;
        try { await applyFlashRuntime(); }
        catch (error) { flashStatus('error', 'Could not apply runtime', error.message); }
        finally { flashApplyBtn.disabled = false; }
    });

    flashTestBtn?.addEventListener('click', async () => {
        flashTestBtn.disabled = true;
        try {
            await applyFlashRuntime({ quiet: true });
            flashStatus('working', 'Testing roles', 'Starting local models if needed and probing every enabled endpoint…');
            let res = await fetch('/api/flash/runtime/test', { method: 'POST' });
            let data = await res.json();
            if (!res.ok) throw new Error(data.detail || data.error || 'Connection test failed');
            const corrected = useSuggestedFlashModelIds(data.roles);
            if (corrected.length) {
                flashStatus('working', 'Matching bridge model IDs', corrected.join('  •  '));
                await applyFlashRuntime({ quiet: true });
                res = await fetch('/api/flash/runtime/test', { method: 'POST' });
                data = await res.json();
                if (!res.ok) throw new Error(data.detail || data.error || 'Connection re-test failed');
            }
            renderFlashDiagnostics(data.roles);
            const detail = (data.roles || []).map(role => `${role.role}: ${role.enabled === false ? 'disabled' : (role.healthy ? 'ready' : 'failed')}`).join('  •  ');
            flashStatus(data.healthy ? 'success' : 'error', data.healthy ? 'Flash is ready' : 'One or more roles failed', detail);
        } catch (error) {
            flashStatus('error', 'Connection test failed', error.message);
            renderFlashDiagnostics([], error.message);
        } finally { flashTestBtn.disabled = false; }
    });

    flashDeployBtn?.addEventListener('click', async () => {
        flashDeployBtn.disabled = true;
        try {
            await applyFlashRuntime({ quiet: true });
            const role = document.getElementById('flashDeployRole')?.value || 'gemma';
            const connectionId = document.getElementById('flashDeployTarget')?.value || 'primary';
            const gpu = Number(document.getElementById('flashDeployGpu')?.value || 0);
            flashStatus('working', `Deploying ${role}`, 'The remote bridge is downloading and starting the catalog model…');
            const res = await fetch('/api/flash/deploy', {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ role, connection_id: connectionId, gpu })
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || data.error || 'Deployment failed');
            flashStatus('success', `${role} deployed`, data.result?.message || `Model is running on GPU #${gpu}.`);
        } catch (error) {
            flashStatus('error', 'Deployment failed', error.message);
        } finally { flashDeployBtn.disabled = false; }
    });

    function showComposerNotice(message) {
        const old = document.getElementById('composerNotice');
        old?.remove();
        const notice = document.createElement('div');
        notice.id = 'composerNotice';
        notice.className = 'composer-notice';
        notice.textContent = message;
        document.getElementById('inputCard')?.appendChild(notice);
        setTimeout(() => notice.remove(), 2400);
    }

    flashModeBtn?.addEventListener('click', () => {
        activeChatMode = 'flash';
        flashModeBtn.classList.add('active');
        reasoningModeBtn?.classList.remove('active');
    });
    reasoningModeBtn?.addEventListener('click', () => {
        activeChatMode = 'flash';
        flashModeBtn?.classList.add('active');
        reasoningModeBtn.classList.remove('active');
        showComposerNotice('Reasoning Mode is coming soon. Flash remains active.');
    });

    // ══════════════════════════════════════════════════
    // NETWORK & LEAK STATUS POPOVER (Positioned to the left of the button)
    // ══════════════════════════════════════════════════
    function positionNetCard() {
        if (!netStatusCard || !netStatusBtn || netStatusCard.style.display === 'none') return;
        const rect = netStatusBtn.getBoundingClientRect();
        netStatusCard.style.position = 'fixed';
        netStatusCard.style.top = Math.max(12, rect.top - 4) + 'px';
        const rightOffset = window.innerWidth - rect.left + 12;
        if (window.innerWidth - rightOffset - 285 < 12) {
            netStatusCard.style.left = '12px';
            netStatusCard.style.right = 'auto';
        } else {
            netStatusCard.style.right = rightOffset + 'px';
            netStatusCard.style.left = 'auto';
        }
    }

    netStatusBtn?.addEventListener('click', (e) => {
        e.stopPropagation();
        const isOpen = netStatusCard && netStatusCard.style.display !== 'none';
        if (!netStatusCard) return;
        if (isOpen) {
            netStatusCard.style.display = 'none';
        } else {
            netStatusCard.style.display = 'block';
            positionNetCard();
        }
    });
    closeNetCardBtn?.addEventListener('click', (e) => {
        e.stopPropagation();
        if (netStatusCard) netStatusCard.style.display = 'none';
    });

    // Close popovers on click outside
    document.addEventListener('click', (e) => {
        if (netStatusCard && netStatusCard.style.display !== 'none') {
            if (!netStatusCard.contains(e.target) && !netStatusBtn?.contains(e.target)) {
                netStatusCard.style.display = 'none';
            }
        }
    });

    // Close on Escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            if (netStatusCard) netStatusCard.style.display = 'none';
            if (settingsPanel) settingsPanel.classList.remove('open');
            const tmDropdownMenu = document.getElementById('tmDropdownMenu');
            if (tmDropdownMenu) tmDropdownMenu.style.display = 'none';
            closeModelDetailsModal();
        }
    });

    // ══════════════════════════════════════════════════
    // TABS & 3D ARTIFACT GRAPH INTEGRATION
    // ══════════════════════════════════════════════════
    const artifactsView   = document.getElementById('artifactsView');
    const toolsModelsView = document.getElementById('toolsModelsView');
    const memoryVaultView = document.getElementById('memoryVaultView');
    let artifactGraph = null;
    const localArtifactFiles = new Map(); // id -> File object for instant offline preview

    function initArtifactGraph() {
        if (!artifactsView || typeof SageArtifactGraph === 'undefined') return;
        if (artifactGraph) return;

        artifactGraph = new SageArtifactGraph(artifactsView, {
            onNodeDblClick: (node) => {
                if (node && node.id) {
                    if (localArtifactFiles.has(node.id)) {
                        openFile(localArtifactFiles.get(node.id));
                    } else {
                        openArtifactById(node.id, node.name);
                    }
                }
            },
            onNodeDelete: (nodeId) => {
                localArtifactFiles.delete(nodeId);
            }
        });

        // Wire empty state upload button to the artifact file input
        const graphUploadPromptBtn = document.getElementById('graphUploadPromptBtn');
        const artifactFileInput = document.getElementById('artifactFileInput');
        if (graphUploadPromptBtn && artifactFileInput) {
            graphUploadPromptBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                artifactFileInput.value = '';
                artifactFileInput.click();
            });
        }

        // Wire Add Files button in graph HUD to preview files in 3D graph
        const graphAddFilesBtn = document.getElementById('graphAddFilesBtn');
        if (graphAddFilesBtn && artifactFileInput) {
            graphAddFilesBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                artifactFileInput.value = '';
                artifactFileInput.click();
            });
        }
    }

    async function handleUserAddedArtifacts(files) {
        if (!files || !files.length) return;

        if (!artifactGraph) {
            initArtifactGraph();
        }
        if (!artifactGraph) {
            console.warn('Artifact graph could not be initialized yet.');
            return;
        }

        const newNodes = [];
        const newEdges = [];

        files.forEach((file, idx) => {
            const ext = (file.name.split('.').pop() || '').toLowerCase();
            let type = 'text';
            if (['pdf'].includes(ext)) type = 'pdf';
            else if (['docx', 'doc'].includes(ext)) type = 'docx';
            else if (['png', 'jpg', 'jpeg', 'svg', 'webp', 'gif'].includes(ext)) type = 'image';
            else if (['csv', 'xlsx', 'xls', 'tsv'].includes(ext)) type = 'sheet';
            else if (['py', 'js', 'ts', 'html', 'css', 'json', 'sh', 'sql'].includes(ext)) type = 'code';

            const nodeId = 'user_art_' + Date.now() + '_' + idx;
            const sizeKB = (file.size / 1024).toFixed(1);

            localArtifactFiles.set(nodeId, file);

            const newNode = {
                id: nodeId,
                name: file.name,
                label: file.name,
                type: type,
                file_type: type,
                format: ext.toUpperCase(),
                size_bytes: file.size,
                size_formatted: `${sizeKB} KB`,
                status: 'ready',
                created_at: new Date().toISOString(),
                connected_count: 0
            };

            newNodes.push(newNode);
        });

        // Generate connections between existing and new nodes
        const existing = artifactGraph.nodesData || [];
        const allNodes = [...existing, ...newNodes];

        if (allNodes.length > 1) {
            newNodes.forEach((node, i) => {
                if (existing.length > 0) {
                    const target = existing[i % existing.length];
                    newEdges.push({
                        source: node.id,
                        target: target.id,
                        type: 'cross_referenced',
                        weight: 1.0
                    });
                } else if (newNodes.length > 1) {
                    const nextNode = newNodes[(i + 1) % newNodes.length];
                    if (node.id !== nextNode.id) {
                        newEdges.push({
                            source: node.id,
                            target: nextNode.id,
                            type: 'connected',
                            weight: 1.0
                        });
                    }
                }
            });
        }

        const combinedNodes = [...existing, ...newNodes];
        const combinedEdges = [...(artifactGraph.edgesData || []), ...newEdges];

        artifactGraph.setData(combinedNodes, combinedEdges, artifactGraph.summary);
        artifactGraph.start();
        artifactGraph.wakeSimulation();

        // Optional sync to backend if running
        uploadFilesToArtifacts(files).catch(() => {});
    }

    // Global file input listener for Artifacts tab
    const artifactFileInputEl = document.getElementById('artifactFileInput');
    if (artifactFileInputEl) {
        artifactFileInputEl.addEventListener('change', async (e) => {
            const files = Array.from(e.target.files || []);
            if (!files.length) return;
            await handleUserAddedArtifacts(files);
        });
    }

    const graphAddFilesBtnEl = document.getElementById('graphAddFilesBtn');
    if (graphAddFilesBtnEl && artifactFileInputEl) {
        graphAddFilesBtnEl.addEventListener('click', (e) => {
            e.stopPropagation();
            artifactFileInputEl.value = '';
            artifactFileInputEl.click();
        });
    }

    async function openArtifactById(docId, fileName) {
        if (localArtifactFiles.has(docId)) {
            openFile(localArtifactFiles.get(docId));
            return;
        }
        try {
            const res = await fetch(`/api/artifacts/${docId}/file`);
            if (!res.ok) throw new Error('Artifact file not found or failed to load');
            const blob = await res.blob();
            let name = fileName || 'document';
            const disposition = res.headers.get('content-disposition');
            if (disposition && disposition.includes('filename=')) {
                const match = disposition.match(/filename="?([^";]+)"?/);
                if (match && match[1]) name = match[1];
            }
            const file = new File([blob], name, { type: blob.type });
            openFile(file);
        } catch (err) {
            console.warn('Failed to open remote artifact (standalone/offline):', err.message);
            alert(`Document preview for "${fileName || docId}" requires the backend or an uploaded file.`);
        }
    }

    async function uploadFilesToArtifacts(files) {
        try {
            const formData = new FormData();
            for (const f of files) {
                formData.append('files', f);
            }
            const res = await fetch('/api/artifacts/upload', {
                method: 'POST',
                body: formData
            });
            if (!res.ok) return;
            const data = await res.json();
            if ((data.status === 'success' || data.status === 'ok') && artifactGraph) {
                artifactGraph.loadData();
            }
        } catch (err) {
            // Graceful offline mode: backend not running
            console.log('Artifacts operating client-side (backend offline)');
        }
    }

    // ══════════════════════════════════════════════════
    // THREADS WEBGL BACKGROUND (Chat Tab Exclusive)
    // ══════════════════════════════════════════════════
    class ChatThreadsBackground {
        constructor(canvas, container) {
            this.canvas = canvas;
            this.container = container;
            this.gl = null;
            this.program = null;
            this.animId = null;
            this.running = false;
            this.startTime = performance.now();
            this.mouse = [0.5, 0.5];
            this.targetMouse = [0.5, 0.5];

            this._init();
        }

        _init() {
            if (!this.canvas) return;
            const gl = this.canvas.getContext('webgl', { alpha: true, premultipliedAlpha: false }) ||
                       this.canvas.getContext('experimental-webgl');
            if (!gl) return;
            this.gl = gl;

            gl.clearColor(0, 0, 0, 0);
            gl.enable(gl.BLEND);
            gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

            const vShaderSource = `
                attribute vec2 position;
                attribute vec2 uv;
                varying vec2 vUv;
                void main() {
                    vUv = uv;
                    gl_Position = vec4(position, 0.0, 1.0);
                }
            `;

            const fShaderSource = `
                precision highp float;

                uniform float iTime;
                uniform vec3 iResolution;
                uniform vec3 uColor;
                uniform float uAmplitude;
                uniform float uDistance;
                uniform vec2 uMouse;

                #define PI 3.1415926538

                const int u_line_count = 40;
                const float u_line_width = 7.0;
                const float u_line_blur = 10.0;

                float Perlin2D(vec2 P) {
                    vec2 Pi = floor(P);
                    vec4 Pf_Pfmin1 = vec4(P, P) - vec4(Pi, Pi + 1.0);
                    vec4 Pt = vec4(Pi.xy, Pi.xy + 1.0);
                    Pt = Pt - floor(Pt * (1.0 / 71.0)) * 71.0;
                    Pt += vec4(26.0, 161.0, 26.0, 161.0);
                    Pt *= Pt;
                    Pt = Pt.xzxz * Pt.yyww;
                    vec4 hash_x = fract(Pt * (1.0 / 951.135664));
                    vec4 hash_y = fract(Pt * (1.0 / 642.949883));
                    vec4 grad_x = hash_x - 0.49999;
                    vec4 grad_y = hash_y - 0.49999;
                    vec4 grad_results = inversesqrt(grad_x * grad_x + grad_y * grad_y)
                        * (grad_x * Pf_Pfmin1.xzxz + grad_y * Pf_Pfmin1.yyww);
                    grad_results *= 1.4142135623730950;
                    vec2 blend = Pf_Pfmin1.xy * Pf_Pfmin1.xy * Pf_Pfmin1.xy
                               * (Pf_Pfmin1.xy * (Pf_Pfmin1.xy * 6.0 - 15.0) + 10.0);
                    vec4 blend2 = vec4(blend, vec2(1.0 - blend));
                    return dot(grad_results, blend2.zxzx * blend2.wwyy);
                }

                float pixel(float count, vec2 resolution) {
                    return (1.0 / max(resolution.x, resolution.y)) * count;
                }

                float lineFn(vec2 st, float width, float perc, float offset, vec2 mouse, float time, float amplitude, float distance) {
                    float split_offset = (perc * 0.4);
                    float split_point = 0.1 + split_offset;

                    float amplitude_normal = smoothstep(split_point, 0.7, st.x);
                    float amplitude_strength = 0.5;
                    float finalAmplitude = amplitude_normal * amplitude_strength
                                           * amplitude * (1.0 + (mouse.y - 0.5) * 0.2);

                    float time_scaled = time / 10.0 + (mouse.x - 0.5) * 1.0;
                    float blur = smoothstep(split_point, split_point + 0.05, st.x) * perc;

                    float xnoise = mix(
                        Perlin2D(vec2(time_scaled, st.x + perc) * 2.5),
                        Perlin2D(vec2(time_scaled, st.x + time_scaled) * 3.5) / 1.5,
                        st.x * 0.3
                    );

                    float y = 0.5 + (perc - 0.5) * distance + xnoise / 2.0 * finalAmplitude;

                    float line_start = smoothstep(
                        y + (width / 2.0) + (u_line_blur * pixel(1.0, iResolution.xy) * blur),
                        y,
                        st.y
                    );

                    float line_end = smoothstep(
                        y,
                        y - (width / 2.0) - (u_line_blur * pixel(1.0, iResolution.xy) * blur),
                        st.y
                    );

                    return clamp(
                        (line_start - line_end) * (1.0 - smoothstep(0.0, 1.0, pow(perc, 0.3))),
                        0.0,
                        1.0
                    );
                }

                void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                    vec2 uv = fragCoord / iResolution.xy;

                    float line_strength = 1.0;
                    for (int i = 0; i < u_line_count; i++) {
                        float p = float(i) / float(u_line_count);
                        line_strength *= (1.0 - lineFn(
                            uv,
                            u_line_width * pixel(1.0, iResolution.xy) * (1.0 - p),
                            p,
                            (PI * 1.0) * p,
                            uMouse,
                            iTime,
                            uAmplitude,
                            uDistance
                        ));
                    }

                    float colorVal = 1.0 - line_strength;
                    vec3 colBrand = vec3(0.145, 0.631, 0.761); // #25a1c2 lightened tone
                    fragColor = vec4(colBrand * colorVal, colorVal * 0.6);
                }

                void main() {
                    mainImage(gl_FragColor, gl_FragCoord.xy);
                }
            `;

            const vs = this._compileShader(gl.VERTEX_SHADER, vShaderSource);
            const fs = this._compileShader(gl.FRAGMENT_SHADER, fShaderSource);
            if (!vs || !fs) return;

            const prog = gl.createProgram();
            gl.attachShader(prog, vs);
            gl.attachShader(prog, fs);
            gl.linkProgram(prog);
            if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
                console.error('Shader link error:', gl.getProgramInfoLog(prog));
                return;
            }
            this.program = prog;

            const quadVerts = new Float32Array([
                -1, -1,  0, 0,
                 1, -1,  1, 0,
                -1,  1,  0, 1,
                -1,  1,  0, 1,
                 1, -1,  1, 0,
                 1,  1,  1, 1
            ]);
            const buffer = gl.createBuffer();
            gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
            gl.bufferData(gl.ARRAY_BUFFER, quadVerts, gl.STATIC_DRAW);

            const posLoc = gl.getAttribLocation(prog, 'position');
            const uvLoc  = gl.getAttribLocation(prog, 'uv');
            gl.enableVertexAttribArray(posLoc);
            gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 16, 0);
            if (uvLoc >= 0) {
                gl.enableVertexAttribArray(uvLoc);
                gl.vertexAttribPointer(uvLoc, 2, gl.FLOAT, false, 16, 8);
            }

            this.locs = {
                iTime: gl.getUniformLocation(prog, 'iTime'),
                iResolution: gl.getUniformLocation(prog, 'iResolution'),
                uColor: gl.getUniformLocation(prog, 'uColor'),
                uAmplitude: gl.getUniformLocation(prog, 'uAmplitude'),
                uDistance: gl.getUniformLocation(prog, 'uDistance'),
                uMouse: gl.getUniformLocation(prog, 'uMouse')
            };

            this._bindEvents();
            this.resize();
        }

        _compileShader(type, src) {
            const gl = this.gl;
            const s = gl.createShader(type);
            gl.shaderSource(s, src);
            gl.compileShader(s);
            if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
                console.error('Shader compile error:', gl.getShaderInfoLog(s));
                gl.deleteShader(s);
                return null;
            }
            return s;
        }

        _bindEvents() {
            // Mouse interaction disabled per user preference - waves flow smoothly on their own
            window.addEventListener('resize', () => this.resize());
        }

        resize() {
            if (!this.gl || !this.canvas) return;
            const rect = (this.container || this.canvas).getBoundingClientRect();
            const w = rect.width || window.innerWidth;
            const h = rect.height || window.innerHeight;
            const dpr = Math.min(window.devicePixelRatio || 1, 2);

            this.canvas.width = Math.floor(w * dpr);
            this.canvas.height = Math.floor(h * dpr);
            this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
        }

        start() {
            if (this.running) return;
            this.running = true;
            this.canvas.style.display = 'block';
            this.resize();

            const render = (now) => {
                if (!this.running) return;
                this.animId = requestAnimationFrame(render);

                const gl = this.gl;
                if (!gl || !this.program) return;

                gl.useProgram(this.program);
                gl.uniform1f(this.locs.iTime, (now - this.startTime) * 0.001);
                gl.uniform3f(this.locs.iResolution, this.canvas.width, this.canvas.height, this.canvas.width / this.canvas.height);
                gl.uniform3f(this.locs.uColor, 0.145, 0.388, 0.921); // Sapphire/Polar Blue #2563eb
                gl.uniform1f(this.locs.uAmplitude, 1.0);
                gl.uniform1f(this.locs.uDistance, 0.05);
                gl.uniform2f(this.locs.uMouse, 0.5, 0.5);

                gl.clear(gl.COLOR_BUFFER_BIT);
                gl.drawArrays(gl.TRIANGLES, 0, 6);
            };
            this.animId = requestAnimationFrame(render);
        }

        stop() {
            this.running = false;
            if (this.animId) {
                cancelAnimationFrame(this.animId);
                this.animId = null;
            }
            if (this.canvas) {
                this.canvas.style.display = 'none';
            }
        }
    }

    // Initialize Chat Threads Background (Chat Tab Only)
    let chatThreads = null;
    const threadsCanvas = document.getElementById('chatThreadsCanvas');
    const chatMainEl = document.getElementById('chatMain');
    if (threadsCanvas && chatMainEl) {
        chatThreads = new ChatThreadsBackground(threadsCanvas, chatMainEl);
        chatThreads.start();
    }

    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const tab = btn.dataset.tab;
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            if (tab === 'artifacts') {
                collapseSidebar();
                chatThreads?.stop();
                document.querySelector('.app-shell')?.classList.add('artifacts-mode');
                document.body.classList.add('artifacts-mode');
                if (chatBody) chatBody.style.display = 'none';
                if (toolsModelsView) toolsModelsView.style.display = 'none';
                if (memoryVaultView) memoryVaultView.style.display = 'none';
                if (artifactsView) {
                    artifactsView.style.display = 'flex';
                    initArtifactGraph();
                    if (artifactGraph) {
                        artifactGraph.start();
                        artifactGraph.loadData();
                        setTimeout(() => artifactGraph.onWindowResize(), 50);
                    }
                }
            } else if (tab === 'tools-models') {
                collapseSidebar();
                chatThreads?.stop();
                document.querySelector('.app-shell')?.classList.remove('artifacts-mode');
                document.body.classList.remove('artifacts-mode');
                if (artifactsView) artifactsView.style.display = 'none';
                if (memoryVaultView) memoryVaultView.style.display = 'none';
                if (chatBody) chatBody.style.display = 'none';
                if (artifactGraph) artifactGraph.stop();
                if (toolsModelsView) {
                    toolsModelsView.style.display = 'flex';
                    initToolsModelsView();
                }
            } else if (tab === 'chat') {
                expandSidebar();
                chatThreads?.start();
                document.querySelector('.app-shell')?.classList.remove('artifacts-mode');
                document.body.classList.remove('artifacts-mode');
                if (artifactsView) artifactsView.style.display = 'none';
                if (toolsModelsView) toolsModelsView.style.display = 'none';
                if (memoryVaultView) memoryVaultView.style.display = 'none';
                if (chatBody) chatBody.style.display = '';
                if (artifactGraph) {
                    artifactGraph.stop();
                }
                if (chatActive) positionInput(false);
            } else if (tab === 'memory') {
                collapseSidebar();
                chatThreads?.stop();
                document.querySelector('.app-shell')?.classList.remove('artifacts-mode');
                document.body.classList.remove('artifacts-mode');
                if (artifactsView) artifactsView.style.display = 'none';
                if (toolsModelsView) toolsModelsView.style.display = 'none';
                if (chatBody) chatBody.style.display = 'none';
                if (artifactGraph) {
                    artifactGraph.stop();
                }
                if (memoryVaultView) {
                    memoryVaultView.style.display = 'flex';
                    initMemoryVault();
                }
            }
        });
    });

    if (window.location.pathname === '/memory' || window.location.hash === '#memory') {
        document.querySelector('.tab-btn[data-tab="memory"]')?.click();
    }

    // ══════════════════════════════════════════════════
    // TOOLS & MODELS CONTROLLER (Frontend-First Prototype)
    // ══════════════════════════════════════════════════
    const INSTALLED_MODELS = [
        {
            id: 'gemma-4b',
            name: 'Gemma 4B Instruct',
            author: 'Google',
            type: 'reasoning',
            role: 'Central Controller',
            format: 'GGUF Q4_K_M',
            parameters: '4B',
            context: '16,384 tokens',
            reasoning: 'Enabled',
            vram: '~2.9 GB',
            status: 'active',
            capabilities: ['Text', 'Reasoning', 'Tool Dispatch']
        },
        {
            id: 'qwen2.5-coder-7b',
            name: 'Qwen2.5-Coder 7B Instruct',
            author: 'Qwen',
            type: 'coding',
            role: 'Code Specialist',
            format: 'GGUF Q4_K_M',
            parameters: '7.6B',
            context: '8,192 tokens',
            reasoning: 'Disabled',
            vram: '~4.8 GB',
            status: 'ready',
            capabilities: ['Code Synthesis', 'Bug Diagnosis', 'Execution Loop']
        },
        {
            id: 'qwen3-vl-4b',
            name: 'Qwen3-VL 4B Vision & OCR',
            author: 'Qwen',
            type: 'vision',
            role: 'Vision Specialist',
            format: 'GGUF Q4_K_M + mmproj',
            parameters: '4.1B',
            context: '4,096 tokens',
            reasoning: 'Disabled',
            vram: '~3.1 GB',
            status: 'ready',
            capabilities: ['Visual Grounding', 'Document OCR', 'Chart Parsing']
        },
        {
            id: 'deepseek-r1-7b',
            name: 'DeepSeek-R1 Distill Qwen 7B',
            author: 'DeepSeek',
            type: 'reasoning',
            role: 'Reasoning & Math Specialist',
            format: 'GGUF Q4_K_M',
            parameters: '7B',
            context: '32,768 tokens',
            reasoning: 'Native R1 Chain',
            vram: '~4.9 GB',
            status: 'ready',
            capabilities: ['Multi-Step Logic', 'Complex Math', 'Planning']
        },
        {
            id: 'llama-3.2-3b',
            name: 'Llama 3.2 3B Instruct',
            author: 'Meta',
            type: 'small_local',
            role: 'Edge & Fast Inference',
            format: 'GGUF Q4_K_M',
            parameters: '3.2B',
            context: '8,192 tokens',
            reasoning: 'Disabled',
            vram: '~2.2 GB',
            status: 'ready',
            capabilities: ['Low-Latency Chat', 'Summary', 'Text Parsing']
        }
    ];

    const RECOMMENDED_MODELS = [
        {
            id: 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
            name: 'DeepSeek-R1-Distill-Qwen-7B',
            provider: 'deepseek-ai',
            badge: 'New release',
            badgeClass: 'new',
            description: 'Trained via large-scale reinforcement learning on Qwen2.5. Demonstrates exceptional mathematical, coding, and multi-step reasoning capabilities comparable to larger frontier models.',
            parameters: '7B',
            context: '128K',
            modality: 'Text',
            categories: ['reasoning', 'coding'],
            bestFor: ['Reasoning', 'Mathematics', 'Code Logic'],
            hardwareFit: {
                gpu: '8 GB VRAM (~4.9 GB allocation)',
                ram: '16 GB system memory',
                level: 'good',
                label: 'Compatible'
            },
            quantization: [
                { tag: 'Q4', state: 'ok' },
                { tag: 'Q5', state: 'ok' },
                { tag: 'Q8', state: 'warn' },
                { tag: 'FP16', state: 'warn' }
            ],
            whyRecommended: 'Unrivaled open-weights reasoning density for a 7B model. Perfect for local complex problem decomposition without external API leakage.',
            hfUrl: 'https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
            license: 'MIT',
            architecture: 'Qwen2ForCausalLM',
            supportedFormats: ['GGUF', 'Safetensors', 'Ollama'],
            hardwareNotes: 'Runs at full speed with all 28 layers offloaded to CUDA VRAM (approx 4.9 GB allocation). Leaves 3.1 GB free for KV cache and context.',
            useCases: 'Complex multi-step task planning, scientific and statistical reasoning, Python logic verification.'
        },
        {
            id: 'Qwen/Qwen2.5-Coder-7B-Instruct',
            name: 'Qwen2.5-Coder-7B-Instruct',
            provider: 'Qwen',
            badge: 'Recommended',
            badgeClass: 'top',
            description: 'State-of-the-art open code model benchmarked against frontier models in Python code synthesis, bug fixing, and repository understanding.',
            parameters: '7.6B',
            context: '128K',
            modality: 'Text',
            categories: ['coding'],
            bestFor: ['Code Generation', 'Bug Diagnosis', 'Python Scripting'],
            hardwareFit: {
                gpu: '8 GB VRAM (~4.8 GB allocation)',
                ram: '16 GB system memory',
                level: 'good',
                label: 'Compatible'
            },
            quantization: [
                { tag: 'Q4', state: 'ok' },
                { tag: 'Q5', state: 'ok' },
                { tag: 'Q8', state: 'warn' },
                { tag: 'FP16', state: 'warn' }
            ],
            whyRecommended: "SAGE's default Coder Specialist. Extremely high pass@1 reliability generating deterministic code for our Docker sandbox.",
            hfUrl: 'https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct',
            license: 'Apache 2.0',
            architecture: 'Qwen2ForCausalLM',
            supportedFormats: ['GGUF', 'Safetensors'],
            hardwareNotes: 'Fits completely inside 8 GB VRAM with 8K context. Supports llama.cpp GPU acceleration.',
            useCases: 'Air-gapped Python scripts, SQL queries, data transformations, self-debugging loops.'
        },
        {
            id: 'google/gemma-2-9b-it',
            name: 'Gemma-2-9B-It',
            provider: 'google',
            badge: 'New release',
            badgeClass: 'top',
            description: 'Google’s high-efficiency lightweight open model built with knowledge distillation and interleaving sliding window attention.',
            parameters: '9.2B',
            context: '8K',
            modality: 'Text',
            categories: ['reasoning'],
            bestFor: ['General Reasoning', 'Instruction Following', 'Tool Planning'],
            hardwareFit: {
                gpu: '8 GB VRAM (~6.1 GB in Q4_K_M)',
                ram: '16 GB system memory',
                level: 'warn',
                label: 'Compatible (Q4_K_M)'
            },
            quantization: [
                { tag: 'Q4', state: 'ok' },
                { tag: 'Q5', state: 'warn' },
                { tag: 'Q8', state: 'warn' },
                { tag: 'FP16', state: 'warn' }
            ],
            whyRecommended: 'Top-tier conversational instruction adherence and safety alignment for the central controller role.',
            hfUrl: 'https://huggingface.co/google/gemma-2-9b-it',
            license: 'Gemma Open',
            architecture: 'Gemma2ForCausalLM',
            supportedFormats: ['GGUF', 'Safetensors'],
            hardwareNotes: 'Fits in 8 GB VRAM when using 4-bit quantization (Q4_K_M). High context (>8K) may require partial system RAM offloading.',
            useCases: 'High-quality user conversational orchestration, policy synthesis, multi-turn reasoning.'
        },
        {
            id: 'Qwen/Qwen2-VL-7B-Instruct',
            name: 'Qwen2-VL-7B-Instruct',
            provider: 'Qwen',
            badge: 'Multimodal',
            badgeClass: 'top',
            description: 'Advanced visual-language model supporting arbitrary image resolution with NaViT dynamics and video understanding.',
            parameters: '7.6B',
            context: '32K',
            modality: 'Text + Image + Video',
            categories: ['vision'],
            bestFor: ['Document OCR', 'Chart Analysis', 'Visual Grounding'],
            hardwareFit: {
                gpu: '8 GB VRAM (~5.2 GB with mmproj)',
                ram: '16 GB system memory',
                level: 'good',
                label: 'Compatible'
            },
            quantization: [
                { tag: 'Q4', state: 'ok' },
                { tag: 'Q5', state: 'ok' },
                { tag: 'Q8', state: 'warn' },
                { tag: 'FP16', state: 'warn' }
            ],
            whyRecommended: 'Handles high-density tables, multi-column research papers, and technical diagrams with pinpoint coordinate accuracy.',
            hfUrl: 'https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct',
            license: 'Apache 2.0',
            architecture: 'Qwen2VLForConditionalGeneration',
            supportedFormats: ['GGUF + mmproj', 'Safetensors'],
            hardwareNotes: 'Requires mmproj GGUF file for vision encoder. Offloads vision tower to VRAM seamlessly.',
            useCases: 'Extracting balance sheets, scanned engineering diagrams, handwriting transcription.'
        },
        {
            id: 'meta-llama/Llama-3.2-3B-Instruct',
            name: 'Llama-3.2-3B-Instruct',
            provider: 'meta-llama',
            badge: 'Edge Optimized',
            badgeClass: 'new',
            description: 'Meta’s highly optimized lightweight model specifically pruned and distilled for on-device, low-latency applications.',
            parameters: '3.2B',
            context: '128K',
            modality: 'Text',
            categories: ['small_local', 'reasoning'],
            bestFor: ['Low Latency', 'Small Footprint', 'Edge Inference'],
            hardwareFit: {
                gpu: '8 GB VRAM (~2.2 GB allocation)',
                ram: '8 GB+ system memory',
                level: 'good',
                label: 'Ultra Low Footprint'
            },
            quantization: [
                { tag: 'Q4', state: 'ok' },
                { tag: 'Q5', state: 'ok' },
                { tag: 'Q8', state: 'ok' },
                { tag: 'FP16', state: 'ok' }
            ],
            whyRecommended: 'Blazing fast inference with negligible memory footprint. Can co-exist in VRAM alongside vision or embedding models.',
            hfUrl: 'https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct',
            license: 'Llama 3.2 Community',
            architecture: 'LlamaForCausalLM',
            supportedFormats: ['GGUF', 'Safetensors', 'Ollama'],
            hardwareNotes: 'Consumes less than 2.5 GB VRAM even at 8-bit precision. Extremely fast on laptop GPUs.',
            useCases: 'Rapid text filtering, summarization, low-power laptop battery mode.'
        },
        {
            id: 'HuggingFaceTB/SmolLM2-1.7B-Instruct',
            name: 'SmolLM2-1.7B-Instruct',
            provider: 'HuggingFaceTB',
            badge: 'Ultra Compact',
            badgeClass: 'top',
            description: 'Hugging Face’s compact instruction-tuned model trained on 11 trillion tokens. Outperforms previous-generation models twice its size.',
            parameters: '1.7B',
            context: '8K',
            modality: 'Text',
            categories: ['small_local'],
            bestFor: ['Ultra-low VRAM', 'Background Utilities', 'Fast Parsing'],
            hardwareFit: {
                gpu: '8 GB VRAM (~1.2 GB allocation)',
                ram: '8 GB system memory',
                level: 'good',
                label: 'CPU & GPU Compatible'
            },
            quantization: [
                { tag: 'Q4', state: 'ok' },
                { tag: 'Q5', state: 'ok' },
                { tag: 'Q8', state: 'ok' },
                { tag: 'FP16', state: 'ok' }
            ],
            whyRecommended: 'Can run on virtually any laptop hardware (even CPU-only). Great for lightweight deterministic utility tasks.',
            hfUrl: 'https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct',
            license: 'Apache 2.0',
            architecture: 'LlamaForCausalLM',
            supportedFormats: ['GGUF', 'Safetensors'],
            hardwareNotes: 'Runs at 60+ tokens/second on RTX 4060 with < 1.5 GB memory footprint.',
            useCases: 'Data cleaning, format translation, keyword tagging, deterministic entity parsing.'
        }
    ];

    const activeSlotModels = {
        reasoning: 'gemma-4b',
        vision: 'qwen3-vl-4b',
        coder: 'qwen2.5-coder-7b'
    };

    let currentRecFilter = 'all';
    let currentRecSearch = '';
    let isTmInitialized = false;

    function initToolsModelsView() {
        if (isTmInitialized) return;
        isTmInitialized = true;

        setupSlotModelPickers();
        setupRecommendations();
        setupRefreshActions();
    }

    function setupSlotModelPickers() {
        const slots = ['reasoning', 'vision', 'coder'];

        slots.forEach(slot => {
            const cap = slot.charAt(0).toUpperCase() + slot.slice(1);
            const btn = document.getElementById(`tmSlotBtn${cap}`);
            const menu = document.getElementById(`tmSlotMenu${cap}`);
            const slotItem = document.querySelector(`.tm-slot-${slot}`);
            const addMoreBtn = menu?.querySelector('.tm-slot-add-more-btn');

            if (!btn || !menu) return;

            renderSlotDropdown(slot);

            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                // Close other slot menus
                slots.forEach(other => {
                    if (other !== slot) {
                        const otherCap = other.charAt(0).toUpperCase() + other.slice(1);
                        const otherMenu = document.getElementById(`tmSlotMenu${otherCap}`);
                        if (otherMenu) otherMenu.style.display = 'none';
                        document.querySelector(`.tm-slot-${other}`)?.classList.remove('menu-open');
                    }
                });

                const isClosed = menu.style.display === 'none';
                menu.style.display = isClosed ? 'block' : 'none';
                if (isClosed) {
                    slotItem?.classList.add('menu-open');
                } else {
                    slotItem?.classList.remove('menu-open');
                }
            });

            addMoreBtn?.addEventListener('click', (e) => {
                e.stopPropagation();
                menu.style.display = 'none';
                slotItem?.classList.remove('menu-open');
                const recSection = document.getElementById('tmRecommendationsSection');
                if (recSection) {
                    recSection.scrollIntoView({ behavior: 'smooth' });
                    recSection.classList.add('tm-section-highlight');
                    setTimeout(() => recSection.classList.remove('tm-section-highlight'), 1300);
                }
            });
        });

        // Close when clicking outside
        document.addEventListener('click', (e) => {
            slots.forEach(slot => {
                const cap = slot.charAt(0).toUpperCase() + slot.slice(1);
                const btn = document.getElementById(`tmSlotBtn${cap}`);
                const menu = document.getElementById(`tmSlotMenu${cap}`);
                if (menu && btn && !btn.contains(e.target) && !menu.contains(e.target)) {
                    menu.style.display = 'none';
                    document.querySelector(`.tm-slot-${slot}`)?.classList.remove('menu-open');
                }
            });
        });
    }

    function renderSlotDropdown(slot) {
        const cap = slot.charAt(0).toUpperCase() + slot.slice(1);
        const list = document.getElementById(`tmSlotList${cap}`);
        if (!list) return;
        list.innerHTML = '';

        INSTALLED_MODELS.forEach(m => {
            const isSelected = activeSlotModels[slot] === m.id;
            const opt = document.createElement('div');
            opt.className = `tm-slot-model-opt ${isSelected ? 'active' : ''}`;
            opt.innerHTML = `
                <div class="tm-smo-info">
                    <span class="tm-smo-name">${esc(m.name)}</span>
                    <span class="tm-smo-meta">${esc(m.format)} &middot; Context: ${esc(m.context)} &middot; VRAM: ${esc(m.vram)}</span>
                </div>
                <span class="tm-smo-tag">${isSelected ? 'ACTIVE' : 'SELECT'}</span>
            `;

            opt.addEventListener('click', (e) => {
                e.stopPropagation();
                selectSlotModel(slot, m);
            });

            list.appendChild(opt);
        });
    }

    function selectSlotModel(slot, model) {
        activeSlotModels[slot] = model.id;

        const cap = slot.charAt(0).toUpperCase() + slot.slice(1);
        const nameEl = document.getElementById(`tmSlot${cap}Name`);
        if (nameEl) nameEl.textContent = model.name;

        const metaEl = document.getElementById(`tmSlot${cap}Meta`);
        if (metaEl) {
            metaEl.textContent = `Format: ${model.format} · Context: ${model.context} · VRAM: ${model.vram}`;
        }

        const menu = document.getElementById(`tmSlotMenu${cap}`);
        if (menu) menu.style.display = 'none';

        renderSlotDropdown(slot);
    }

    function setupRecommendations() {
        const searchInput = document.getElementById('tmRecSearchInput');
        const pillsWrap   = document.getElementById('tmRecPills');

        searchInput?.addEventListener('input', (e) => {
            currentRecSearch = (e.target.value || '').trim().toLowerCase();
            renderRecommendations();
        });

        pillsWrap?.querySelectorAll('.tm-tab, .tm-pill').forEach(tab => {
            tab.addEventListener('click', () => {
                pillsWrap.querySelectorAll('.tm-tab, .tm-pill').forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                currentRecFilter = tab.dataset.filter || 'all';
                renderRecommendations();
            });
        });

        renderRecommendations();
    }

    function renderRecommendations() {
        const stream = document.getElementById('tmRecCardsStream');
        if (!stream) return;

        let filtered = RECOMMENDED_MODELS.filter(m => {
            if (currentRecFilter !== 'all' && !m.categories.includes(currentRecFilter)) {
                return false;
            }
            if (currentRecSearch) {
                const text = `${m.name} ${m.provider} ${m.description} ${m.bestFor.join(' ')}`.toLowerCase();
                if (!text.includes(currentRecSearch)) return false;
            }
            return true;
        });

        if (!filtered.length) {
            stream.innerHTML = `
                <div style="padding:32px 20px;text-align:center;color:#64748b;background:#f8fafc;border-radius:8px;border:1px dashed #cbd5e1;font-size:12px;">
                    No models match your current filter. Try a different search term or category.
                </div>
            `;
            return;
        }

        stream.innerHTML = '';

        filtered.forEach(m => {
            const card = document.createElement('div');
            card.className = 'tm-rec-card';

            const quantList = m.quantization.map(q => q.tag).join(', ');
            const fitLevel = m.hardwareFit.level === 'good' ? 'good' : 'warn';

            card.innerHTML = `
                <div class="tm-rc-header">
                    <div class="tm-rc-title-row">
                        <span class="tm-rc-name">${esc(m.name)}</span>
                        <span class="tm-rc-release-tag">(${esc(m.badge)})</span>
                    </div>
                </div>

                <p class="tm-rc-desc">${esc(m.description)}</p>

                <div class="tm-rc-hw-row">
                    <span class="tm-rc-hw-lbl">Hardware required:</span>
                    <span class="tm-rc-hw-val">${esc(m.hardwareFit.gpu)} &middot; RAM: ${esc(m.hardwareFit.ram)} &middot; Quantization: ${esc(quantList)}</span>
                    <span class="tm-rc-hw-fit-badge ${fitLevel}">${esc(m.hardwareFit.label)}</span>
                </div>

                <div class="tm-rc-footer">
                    <div class="tm-rc-why">
                        <span class="tm-rc-link-prefix">Link &rarr;</span> <a class="tm-rc-hf-link" href="${esc(m.hfUrl)}" target="_blank" rel="noopener noreferrer">${esc(m.hfUrl)}</a>
                    </div>
                    <button class="tm-rc-details-btn" data-id="${esc(m.id)}">Details &rarr;</button>
                </div>
            `;

            card.querySelector('.tm-rc-details-btn')?.addEventListener('click', () => {
                openModelDetailsModal(m);
            });

            stream.appendChild(card);
        });
    }



    function openModelDetailsModal(model) {
        const modal = document.getElementById('modelDetailsModal');
        if (!modal) return;

        const titleEl = document.getElementById('tmModalTitle');
        if (titleEl) titleEl.textContent = model.name;

        const authorEl = document.getElementById('tmModalAuthor');
        if (authorEl) authorEl.textContent = `Maintained by ${model.provider} · ${model.license} License`;

        const hfBtn = document.getElementById('tmModalHfBtn');
        if (hfBtn) hfBtn.href = model.hfUrl;

        const bodyEl = document.getElementById('tmModalBody');
        if (bodyEl) {
            bodyEl.innerHTML = `
                <div class="tm-md-sec">
                    <span class="tm-md-lbl">Architecture &amp; Specifications</span>
                    <div class="tm-md-grid">
                        <div class="tm-md-grid-item">
                            <span class="sc-lbl">Architecture</span>
                            <span class="sc-val">${esc(model.architecture)}</span>
                        </div>
                        <div class="tm-md-grid-item">
                            <span class="sc-lbl">Parameters</span>
                            <span class="sc-val">${esc(model.parameters)}</span>
                        </div>
                        <div class="tm-md-grid-item">
                            <span class="sc-lbl">Context Window</span>
                            <span class="sc-val">${esc(model.context)} tokens</span>
                        </div>
                        <div class="tm-md-grid-item">
                            <span class="sc-lbl">Supported Formats</span>
                            <span class="sc-val">${esc(model.supportedFormats.join(', '))}</span>
                        </div>
                    </div>
                </div>

                <div class="tm-md-sec">
                    <span class="tm-md-lbl">Overview &amp; Capabilities</span>
                    <p style="margin:0;font-size:12.5px;color:var(--tx2);line-height:1.5;">${esc(model.description)}</p>
                </div>

                <div class="tm-md-sec">
                    <span class="tm-md-lbl">Hardware Considerations (Your RTX 4060 System)</span>
                    <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;padding:10px 12px;font-size:12px;color:#166534;line-height:1.45;">
                        <strong>Deployment Profile:</strong> ${esc(model.hardwareNotes)}
                    </div>
                </div>

                <div class="tm-md-sec">
                    <span class="tm-md-lbl">Optimal Use Cases</span>
                    <p style="margin:0;font-size:12px;color:var(--tx3);">${esc(model.useCases)}</p>
                </div>
            `;
        }

        modal.style.display = 'flex';
        modal.setAttribute('aria-hidden', 'false');
    }

    function closeModelDetailsModal() {
        const modal = document.getElementById('modelDetailsModal');
        if (!modal) return;
        modal.style.display = 'none';
        modal.setAttribute('aria-hidden', 'true');
    }

    document.getElementById('tmModalCloseBtn')?.addEventListener('click', closeModelDetailsModal);
    document.getElementById('tmModalCloseFooterBtn')?.addEventListener('click', closeModelDetailsModal);
    document.getElementById('tmModalBackdrop')?.addEventListener('click', closeModelDetailsModal);

    function setupRefreshActions() {
        const topRefreshBtn = document.getElementById('tmRefreshBtn');
        const ddDetectBtn   = document.getElementById('tmDdDetectBtn');

        function triggerScan() {
            if (topRefreshBtn) {
                topRefreshBtn.classList.add('loading');
                const span = topRefreshBtn.querySelector('span');
                if (span) span.textContent = 'Detecting...';
            }

            setTimeout(() => {
                if (topRefreshBtn) {
                    topRefreshBtn.classList.remove('loading');
                    const span = topRefreshBtn.querySelector('span');
                    if (span) span.textContent = 'Refresh';
                }
                const badge = document.getElementById('tmInstalledCountBadge');
                if (badge) {
                    badge.textContent = `${INSTALLED_MODELS.length} Installed Local`;
                }
                renderDropdownList();
            }, 550);
        }

        topRefreshBtn?.addEventListener('click', triggerScan);
        ddDetectBtn?.addEventListener('click', triggerScan);
    }

    // ══════════════════════════════════════════════════
    // HISTORY & DELETE
    // ══════════════════════════════════════════════════
    const historyList = document.getElementById('historyList');

    function formatHistoryTime(iso) {
        if (!iso) return '';
        const date = new Date(iso);
        if (Number.isNaN(date.getTime())) return String(iso).slice(0, 16);
        const now = new Date();
        const sameDay = date.toDateString() === now.toDateString();
        const yesterday = new Date(now);
        yesterday.setDate(now.getDate() - 1);
        const time = String(date.getHours()).padStart(2, '0') + ':' + String(date.getMinutes()).padStart(2, '0');
        if (sameDay) return 'Today · ' + time;
        if (date.toDateString() === yesterday.toDateString()) return 'Yesterday · ' + time;
        return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) + ' · ' + time;
    }

    function createHistoryItem(chat, { active = false } = {}) {
        const item = document.createElement('div');
        item.className = 'history-item' + (active ? ' active' : '');
        item.dataset.chatId = chat.chat_id;
        item.innerHTML = `
            <div class="h-text">
                <span class="h-title">${esc(chat.title || 'New conversation')}</span>
                <span class="h-time">${esc(formatHistoryTime(chat.updated_at || chat.created_at))}</span>
            </div>
            <button class="h-del-btn" title="Delete chat" aria-label="Delete chat">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
            </button>
        `;
        initHistoryItem(item);
        return item;
    }

    function renderHistoryEmpty() {
        if (!historyList) return;
        historyList.innerHTML = '<div class="history-empty">No saved chats yet. Start a conversation to keep it here.</div>';
    }

    async function loadChatHistory() {
        if (!historyList || !uiSettings.saveHistory) {
            if (historyList) renderHistoryEmpty();
            return;
        }
        try {
            const res = await fetch('/api/chats');
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || 'Failed to load chats');
            const chats = data.chats || [];
            historyList.innerHTML = '';
            if (!chats.length) {
                renderHistoryEmpty();
                return;
            }
            chats.forEach(chat => {
                historyList.appendChild(createHistoryItem(chat, {
                    active: chat.chat_id === flashSessionId || chat.chat_id === currentChatId
                }));
            });
        } catch (error) {
            console.warn('Chat history unavailable:', error);
            renderHistoryEmpty();
        }
    }

    function upsertHistoryItem(chatId, title) {
        if (!historyList || !uiSettings.saveHistory || !chatId) return;
        historyList.querySelector('.history-empty')?.remove();
        let item = historyList.querySelector(`.history-item[data-chat-id="${CSS.escape(chatId)}"]`);
        if (!item) {
            item = createHistoryItem({
                chat_id: chatId,
                title: title || 'New conversation',
                updated_at: new Date().toISOString(),
            }, { active: true });
            historyList.insertBefore(item, historyList.firstChild);
        } else {
            const titleEl = item.querySelector('.h-title');
            const timeEl = item.querySelector('.h-time');
            if (titleEl && title) titleEl.textContent = title;
            if (timeEl) timeEl.textContent = formatHistoryTime(new Date().toISOString());
            historyList.insertBefore(item, historyList.firstChild);
        }
        document.querySelectorAll('.history-item').forEach(i => i.classList.toggle('active', i === item));
    }

    async function openChatFromHistory(chatId) {
        if (!chatId || isRunning) return;
        try {
            const res = await fetch(`/api/chats/${encodeURIComponent(chatId)}`);
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || 'Could not load chat');
            resetChat({ keepHistorySelection: true, newSession: false });
            flashSessionId = chatId;
            currentChatId = chatId;
            // If the Memory Vault is currently open, refresh its recent-chat context
            // to reflect the newly selected chat session.
            if (memoryVaultView && memoryVaultView.style.display !== 'none') {
                refreshMemoryVaultData();
            }
            document.querySelectorAll('.history-item').forEach(i => {
                i.classList.toggle('active', i.dataset.chatId === chatId);
            });
            const messages = data.messages || [];
            if (!messages.length) {
                showComposerNotice('This chat has no saved messages.');
                return;
            }
            activateChat();
            welcomeOverlay.style.display = 'none';
            messages.forEach(msg => {
                if (msg.role === 'user') appendUserMsg(msg.content || '', []);
                else if (msg.role === 'assistant') appendSageReply(msg.content || '', null);
            });
            positionInput(false);
        } catch (error) {
            showComposerNotice(error.message || 'Failed to open chat');
        }
    }

    function initHistoryItem(item) {
        if (!item || item.dataset.bound) return;
        item.dataset.bound = 'true';

        item.addEventListener('click', (e) => {
            if (e.target.closest('.h-del-btn')) return;
            const chatId = item.dataset.chatId;
            if (chatId) openChatFromHistory(chatId);
        });

        const delBtn = item.querySelector('.h-del-btn');
        delBtn?.addEventListener('click', async (e) => {
            e.preventDefault();
            e.stopPropagation();
            const chatId = item.dataset.chatId;
            const wasActive = item.classList.contains('active');
            item.style.opacity = '0';
            item.style.transform = 'translateX(-12px)';
            item.style.pointerEvents = 'none';
            try {
                if (chatId) {
                    await fetch(`/api/chats/${encodeURIComponent(chatId)}`, { method: 'DELETE' });
                }
            } catch { /* UI still removes the row */ }
            setTimeout(() => {
                item.remove();
                if (!historyList.querySelector('.history-item')) renderHistoryEmpty();
                if (wasActive) resetChat();
            }, 180);
        });
    }

    searchInput?.addEventListener('input', () => {
        const q = searchInput.value.toLowerCase();
        document.querySelectorAll('.history-item').forEach(item => {
            const t = item.querySelector('.h-title')?.textContent.toLowerCase() || '';
            item.style.display = t.includes(q) ? '' : 'none';
        });
    });

    loadChatHistory();

    // ══════════════════════════════════════════════════
    // NEW CHAT — reset to initial centered state
    // ══════════════════════════════════════════════════
    newChatBtn?.addEventListener('click', () => resetChat());
    resetChatInlineBtn?.addEventListener('click', async () => {
        if (isRunning) return;
        const sessionToClear = flashSessionId;
        try {
            await fetch(`/api/flash/session/${encodeURIComponent(sessionToClear)}`, { method: 'DELETE' });
        } catch {
            // The visible session is still reset even if persistence is unavailable.
        }
        document.querySelector(`.history-item[data-chat-id="${CSS.escape(sessionToClear)}"]`)?.remove();
        if (!historyList?.querySelector('.history-item')) renderHistoryEmpty();
        resetChat();
        showComposerNotice('Chat session cleared.');
    });
    function resetChat({ keepHistorySelection = false, newSession = true } = {}) {
        currentChatId = null;
        chatMessages.innerHTML = '';
        chatActive = false;
        if (newSession) {
            flashSessionId = (crypto.randomUUID?.() || ('flash_' + Date.now()));
        }

        // Restore welcome overlay
        welcomeOverlay.style.display = '';
        welcomeOverlay.style.opacity = '';
        welcomeOverlay.style.transition = '';

        // Remove active state → back to initial
        chatMain.setAttribute('data-state', 'initial');
        chatMessages.style.bottom = '';

        // Clear input inline styles so CSS [data-state=initial] takes over
        chatInputArea.style.cssText = '';
        chatInputArea.style.transition = 'none';

        // Reset attachments
        attachedFiles = [];
        renderAttachments();
        promptInput.value = '';
        promptInput.style.height = 'auto';
        hideCardStatus();

        if (!keepHistorySelection) {
            document.querySelectorAll('.history-item').forEach(i => i.classList.remove('active'));
        }

        // Reset execution tiles to idle and empty state
        clearExecTiles();
        rightPanelIdle();
    }

    // ══════════════════════════════════════════════════
    // ACTIVATE CHAT: input center → bottom animation
    // ══════════════════════════════════════════════════
    function activateChat() {
        if (chatActive) return;
        chatActive = true;

        // Capture current rendered position of input
        const bodyRect  = chatBody.getBoundingClientRect();
        const inputRect = chatInputArea.getBoundingClientRect();
        const curTop    = inputRect.top - bodyRect.top;
        const curWidth  = inputRect.width;

        // Freeze input at current pixel position, disable transition momentarily
        chatInputArea.style.transition = 'none';
        chatInputArea.style.top       = curTop   + 'px';
        chatInputArea.style.width     = curWidth + 'px';
        chatInputArea.style.left      = '50%';
        chatInputArea.style.transform = 'translateX(-50%)';

        // Switch to active (removes CSS class-based initial positioning)
        chatMain.setAttribute('data-state', 'active');

        // Fade out welcome
        welcomeOverlay.style.transition = 'opacity .30s ease';
        welcomeOverlay.style.opacity    = '0';

        // Force reflow so the browser registers the "current" position
        chatInputArea.getBoundingClientRect();

        // Enable transition and animate to bottom
        requestAnimationFrame(() => {
            chatInputArea.style.transition =
                'top .45s cubic-bezier(0.34,1.2,0.64,1), width .45s cubic-bezier(0.34,1.2,0.64,1)';
            positionInput(true);

            setTimeout(() => {
                welcomeOverlay.style.display = 'none';
            }, 320);
        });
    }

    // ══════════════════════════════════════════════════
    // POSITION INPUT AT BOTTOM (active state)
    // ══════════════════════════════════════════════════
    function positionInput(animate) {
        if (!chatActive) return;
        const bodyH = chatBody.offsetHeight;
        const bodyW = chatBody.offsetWidth;
        const inputH = chatInputArea.offsetHeight;

        const newTop   = bodyH - inputH - 18;
        const newWidth = Math.min(bodyW - 40, 800);

        if (!animate) {
            // Instant reposition (resize / sidebar toggle)
            const prev = chatInputArea.style.transition;
            chatInputArea.style.transition = 'none';
            chatInputArea.style.top    = newTop   + 'px';
            chatInputArea.style.width  = newWidth + 'px';
            chatInputArea.style.left   = '50%';
            chatInputArea.style.transform = 'translateX(-50%)';
            // Restore transition after reflow
            requestAnimationFrame(() => {
                chatInputArea.style.transition = prev ||
                    'top .45s cubic-bezier(0.34,1.2,0.64,1), width .45s';
            });
        } else {
            chatInputArea.style.top    = newTop   + 'px';
            chatInputArea.style.width  = newWidth + 'px';
        }

        // Keep messages above the input
        chatMessages.style.bottom = (inputH + 24) + 'px';
    }

    // Reposition on window resize
    window.addEventListener('resize', () => {
        if (chatActive) positionInput(false);
        updateTileBlurs();
        positionNetCard();
    });

    // ══════════════════════════════════════════════════
    // FILES & ATTACHMENT PREVIEW (ChatGPT Style)
    // ══════════════════════════════════════════════════
    attachBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', e => {
        handleFiles(Array.from(e.target.files));
        fileInput.value = '';
    });

    window.addEventListener('paste', e => {
        const items = (e.clipboardData || e.originalEvent?.clipboardData)?.items || [];
        const pasted = [];
        for (const item of items) {
            if (item.kind === 'file') {
                const f = item.getAsFile();
                if (f) {
                    const ext = f.type.split('/')[1] || 'png';
                    pasted.push(new File([f], `clipboard_${Date.now()}.${ext}`, { type: f.type }));
                }
            }
        }
        if (pasted.length) handleFiles(pasted);
    });

    window.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('active'); });
    window.addEventListener('dragleave', e => { if (e.clientX <= 0 || e.clientY <= 0) dropZone.classList.remove('active'); });
    window.addEventListener('drop', e => {
        e.preventDefault(); dropZone.classList.remove('active');
        if (e.dataTransfer?.files?.length) handleFiles(Array.from(e.dataTransfer.files));
    });

    function handleFiles(files) {
        files.forEach(f => {
            if (!attachedFiles.some(a => a.name === f.name && a.size === f.size)) attachedFiles.push(f);
        });
        renderAttachments();
        const isArtifactsTab = artifactsView && artifactsView.style.display !== 'none';
        if (isArtifactsTab && files.length > 0) {
            uploadFilesToArtifacts(files);
        }
    }

    // ══════════════════════════════════════════════════
    // IN-SITE DOCUMENT PREVIEW & OPTIONAL DOWNLOAD
    // ══════════════════════════════════════════════════
    function closeDocPreview() {
        if (!docPreviewModal) return;
        docPreviewModal.style.display = 'none';
        docPreviewModal.setAttribute('aria-hidden', 'true');
        if (dpmContent) dpmContent.innerHTML = '';
        if (currentPreviewBlobUrl) {
            URL.revokeObjectURL(currentPreviewBlobUrl);
            currentPreviewBlobUrl = null;
        }
        currentPreviewFile = null;
    }

    dpmCloseBtn?.addEventListener('click', closeDocPreview);
    dpmBackdrop?.addEventListener('click', closeDocPreview);
    window.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && docPreviewModal && docPreviewModal.style.display !== 'none') {
            closeDocPreview();
        }
    });

    dpmDownloadBtn?.addEventListener('click', () => {
        if (!currentPreviewFile) return;
        try {
            const url = currentPreviewBlobUrl || URL.createObjectURL(currentPreviewFile);
            const a = document.createElement('a');
            a.href = url;
            a.download = currentPreviewFile.name || 'download';
            document.body.appendChild(a);
            a.click();
            a.remove();
            if (!currentPreviewBlobUrl) {
                setTimeout(() => URL.revokeObjectURL(url), 2000);
            }
        } catch (err) {
            console.error('Download error:', err);
        }
    });

    function bindFallbackDownload() {
        const btn = document.getElementById('dpmFallbackDownloadBtn');
        btn?.addEventListener('click', () => {
            dpmDownloadBtn?.click();
        });
    }

    // Pure JS ZIP / DOCX Parser without external dependencies (Air-gapped)
    async function extractDocxXml(file) {
        const buffer = await file.arrayBuffer();
        const view = new DataView(buffer);
        let eocdOffset = -1;
        for (let i = buffer.byteLength - 22; i >= Math.max(0, buffer.byteLength - 65557); i--) {
            if (view.getUint32(i, true) === 0x06054b50) {
                eocdOffset = i;
                break;
            }
        }
        if (eocdOffset === -1) {
            throw new Error('Not a valid DOCX or ZIP archive (EOCD record not found).');
        }

        const cdOffset = view.getUint32(eocdOffset + 16, true);
        const cdCount = view.getUint16(eocdOffset + 10, true);
        let p = cdOffset;
        let docEntry = null;
        const decoder = new TextDecoder('utf-8');

        for (let i = 0; i < cdCount; i++) {
            if (p + 46 > buffer.byteLength) break;
            if (view.getUint32(p, true) !== 0x02014b50) break;
            const method = view.getUint16(p + 10, true);
            const compSize = view.getUint32(p + 20, true);
            const fnLen = view.getUint16(p + 28, true);
            const efLen = view.getUint16(p + 30, true);
            const cmLen = view.getUint16(p + 32, true);
            const localHeaderOffset = view.getUint32(p + 42, true);
            const fnBytes = new Uint8Array(buffer, p + 46, fnLen);
            const filename = decoder.decode(fnBytes);

            if (filename === 'word/document.xml') {
                docEntry = { method, compSize, localHeaderOffset };
                break;
            }
            p += 46 + fnLen + efLen + cmLen;
        }

        if (!docEntry) {
            throw new Error('Could not locate word/document.xml in DOCX file.');
        }

        const lh = docEntry.localHeaderOffset;
        if (view.getUint32(lh, true) !== 0x04034b50) {
            throw new Error('Invalid local header signature in ZIP archive.');
        }
        const lFnLen = view.getUint16(lh + 26, true);
        const lEfLen = view.getUint16(lh + 28, true);
        const dataStart = lh + 30 + lFnLen + lEfLen;
        const compSlice = new Uint8Array(buffer, dataStart, docEntry.compSize);

        if (docEntry.method === 0) {
            return decoder.decode(compSlice);
        } else if (docEntry.method === 8) {
            if (typeof DecompressionStream !== 'undefined') {
                const ds = new DecompressionStream('deflate-raw');
                const stream = new Response(compSlice).body.pipeThrough(ds);
                const decompressed = await new Response(stream).arrayBuffer();
                return decoder.decode(decompressed);
            } else {
                throw new Error('Browser runtime does not support DecompressionStream.');
            }
        } else {
            throw new Error(`Unsupported compression method (${docEntry.method}) in DOCX archive.`);
        }
    }

    function renderDocxRuns(pNode) {
        let html = '';
        const nodes = Array.from(pNode.children);

        nodes.forEach(child => {
            const localName = child.localName || child.nodeName.split(':').pop();
            if (localName === 'r') {
                const rPr = child.querySelector('rPr') || child.getElementsByTagNameNS('*', 'rPr')[0];
                let isBold = false;
                let isItalic = false;
                let isUnderline = false;
                let isStrike = false;

                if (rPr) {
                    const b = rPr.querySelector('b') || rPr.getElementsByTagNameNS('*', 'b')[0];
                    if (b && b.getAttribute('w:val') !== '0' && b.getAttribute('w:val') !== 'false') isBold = true;
                    const i = rPr.querySelector('i') || rPr.getElementsByTagNameNS('*', 'i')[0];
                    if (i && i.getAttribute('w:val') !== '0' && i.getAttribute('w:val') !== 'false') isItalic = true;
                    const u = rPr.querySelector('u') || rPr.getElementsByTagNameNS('*', 'u')[0];
                    if (u && u.getAttribute('w:val') !== 'none') isUnderline = true;
                    const strike = rPr.querySelector('strike') || rPr.getElementsByTagNameNS('*', 'strike')[0];
                    if (strike) isStrike = true;
                }

                let runText = '';
                Array.from(child.children).forEach(rc => {
                    const rcName = rc.localName || rc.nodeName.split(':').pop();
                    if (rcName === 't') {
                        runText += esc(rc.textContent || '');
                    } else if (rcName === 'br') {
                        runText += '<br>';
                    } else if (rcName === 'tab') {
                        runText += '&emsp;';
                    }
                });

                if (!runText) return;

                if (isBold) runText = `<strong>${runText}</strong>`;
                if (isItalic) runText = `<em>${runText}</em>`;
                if (isUnderline) runText = `<u>${runText}</u>`;
                if (isStrike) runText = `<s>${runText}</s>`;

                html += runText;
            } else if (localName === 'hyperlink') {
                const linkRuns = renderDocxRuns(child);
                html += `<span style="color:#2563eb;text-decoration:underline;">${linkRuns}</span>`;
            }
        });

        return html;
    }

    function renderDocxTable(tblNode) {
        let html = '<table><tbody>';
        const rows = Array.from(tblNode.getElementsByTagNameNS('*', 'tr'));

        rows.forEach((tr, rIdx) => {
            html += '<tr>';
            const cells = Array.from(tr.getElementsByTagNameNS('*', 'tc'));
            cells.forEach(tc => {
                const tag = rIdx === 0 ? 'th' : 'td';
                const cellPs = Array.from(tc.getElementsByTagNameNS('*', 'p'));
                let cellContent = '';
                if (cellPs.length) {
                    cellContent = cellPs.map(p => renderDocxRuns(p)).join('<br>');
                } else {
                    cellContent = esc(tc.textContent || '');
                }
                html += `<${tag}>${cellContent || '&nbsp;'}</${tag}>`;
            });
            html += '</tr>';
        });

        html += '</tbody></table>';
        return html;
    }

    function convertDocxXmlToHtml(xmlText) {
        const parser = new DOMParser();
        const xmlDoc = parser.parseFromString(xmlText, 'application/xml');
        const parseError = xmlDoc.querySelector('parsererror');
        if (parseError) {
            throw new Error('XML parsing error: ' + parseError.textContent);
        }

        const body = xmlDoc.querySelector('body') || xmlDoc.getElementsByTagNameNS('*', 'body')[0];
        if (!body) throw new Error('No body element found in DOCX XML.');

        let outHtml = '';
        const children = Array.from(body.children);
        let inList = false;

        function closeList() {
            if (inList) {
                outHtml += '</ul>';
                inList = false;
            }
        }

        children.forEach(node => {
            const localName = node.localName || node.nodeName.split(':').pop();

            if (localName === 'tbl') {
                closeList();
                outHtml += renderDocxTable(node);
            } else if (localName === 'p') {
                const pStyle = node.querySelector('pStyle') || node.getElementsByTagNameNS('*', 'pStyle')[0];
                const styleVal = pStyle ? (pStyle.getAttribute('w:val') || pStyle.getAttribute('val') || '') : '';
                const numPr = node.querySelector('numPr') || node.getElementsByTagNameNS('*', 'numPr')[0];
                const isList = !!numPr;
                const pContent = renderDocxRuns(node);

                if (isList) {
                    if (!inList) {
                        inList = true;
                        outHtml += '<ul>';
                    }
                    outHtml += `<li>${pContent || '&nbsp;'}</li>`;
                } else {
                    closeList();
                    if (!pContent.trim()) {
                        outHtml += '<div class="docx-empty-p"></div>';
                        return;
                    }
                    const lowerStyle = styleVal.toLowerCase();
                    if (lowerStyle.includes('heading1') || lowerStyle === 'title') {
                        outHtml += `<h1>${pContent}</h1>`;
                    } else if (lowerStyle.includes('heading2') || lowerStyle === 'subtitle') {
                        outHtml += `<h2>${pContent}</h2>`;
                    } else if (lowerStyle.includes('heading3')) {
                        outHtml += `<h3>${pContent}</h3>`;
                    } else if (lowerStyle.includes('heading4') || lowerStyle.includes('heading5')) {
                        outHtml += `<h4>${pContent}</h4>`;
                    } else {
                        outHtml += `<p>${pContent}</p>`;
                    }
                }
            }
        });

        closeList();
        return outHtml || '<p><em>(Empty document)</em></p>';
    }

    function renderCsvPreview(text) {
        const lines = text.split(/\r\n|\n|\r/).filter(l => l.trim().length > 0);
        if (!lines.length) {
            return '<div style="padding:32px;color:var(--tx3);text-align:center;">Empty spreadsheet.</div>';
        }

        function parseCsvLine(line) {
            const res = [];
            let curr = '';
            let inQuotes = false;
            for (let i = 0; i < line.length; i++) {
                const ch = line[i];
                if (ch === '"') {
                    if (inQuotes && line[i + 1] === '"') {
                        curr += '"';
                        i++;
                    } else {
                        inQuotes = !inQuotes;
                    }
                } else if ((ch === ',' || ch === '\t') && !inQuotes) {
                    res.push(curr);
                    curr = '';
                } else {
                    curr += ch;
                }
            }
            res.push(curr);
            return res;
        }

        let html = '<div class="dpm-table-container"><table class="dpm-data-table"><thead><tr>';
        const headerCols = parseCsvLine(lines[0]);
        headerCols.forEach(col => {
            html += `<th>${esc(col)}</th>`;
        });
        html += '</tr></thead><tbody>';

        const maxRows = Math.min(lines.length, 500);
        for (let r = 1; r < maxRows; r++) {
            html += '<tr>';
            const cols = parseCsvLine(lines[r]);
            for (let c = 0; c < headerCols.length; c++) {
                html += `<td>${esc(cols[c] !== undefined ? cols[c] : '')}</td>`;
            }
            html += '</tr>';
        }
        html += '</tbody></table></div>';
        if (lines.length > 500) {
            html += `<div style="padding:10px 18px;font-size:12px;color:var(--tx3);background:#f8fafc;border-top:1px solid var(--border);text-align:center;">Showing first 500 of ${lines.length} rows</div>`;
        }
        return html;
    }

    function renderFallbackPreview(info, errorMsg) {
        return `
            <div class="dpm-fallback-container">
                <div class="dpm-fallback-card">
                    <div class="dpm-fallback-icon">
                        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>
                    </div>
                    <div class="dpm-fallback-title">${esc(info.name)}</div>
                    <div class="dpm-fallback-desc">
                        ${errorMsg ? esc(errorMsg) + '<br>' : ''}
                        Direct inline rendering is not supported for this file format, but you can download it to view in your application.
                    </div>
                    <button class="dpm-fallback-btn" id="dpmFallbackDownloadBtn">
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
                        <span>Download ${esc(info.name)}</span>
                    </button>
                </div>
            </div>
        `;
    }

    // Open attached document or image in the on-site preview modal (with optional download button)
    async function openFile(file) {
        if (!file) return;

        // Clean up previous preview URL
        if (currentPreviewBlobUrl) {
            URL.revokeObjectURL(currentPreviewBlobUrl);
            currentPreviewBlobUrl = null;
        }
        currentPreviewFile = file;

        const info = getFileInfo(file);

        // Update modal header
        if (dpmFileName) dpmFileName.textContent = info.name;
        if (dpmFileMeta) dpmFileMeta.textContent = `${info.typeLabel} · ${info.sizeStr}`;
        if (dpmFileIcon) {
            dpmFileIcon.className = `dpm-file-icon ${info.ext || 'other'}`;
            dpmFileIcon.innerHTML = info.iconHtml;
        }

        // Show modal & loading spinner
        if (docPreviewModal) {
            docPreviewModal.style.display = 'flex';
            docPreviewModal.setAttribute('aria-hidden', 'false');
        }
        if (dpmLoading) {
            dpmLoading.style.display = 'flex';
            if (dpmLoadingText) dpmLoadingText.textContent = `Preparing preview for ${info.name}...`;
        }
        if (dpmContent) dpmContent.innerHTML = '';

        try {
            const ext = (info.ext || '').toLowerCase();
            const mime = (file.type || '').toLowerCase();

            // 1. DOCX Preview (Pure JS client-side extraction)
            if (ext === 'docx' || ext === 'dotx') {
                try {
                    const xmlText = await extractDocxXml(file);
                    const html = convertDocxXmlToHtml(xmlText);
                    dpmContent.innerHTML = `<div class="dpm-docx-container"><div class="dpm-docx-sheet">${html}</div></div>`;
                } catch (err) {
                    console.warn('DOCX inline parse fallback:', err);
                    dpmContent.innerHTML = renderFallbackPreview(info, 'Document could not be parsed inline.');
                    bindFallbackDownload();
                }
            }
            // 2. PDF Preview
            else if (ext === 'pdf' || mime === 'application/pdf') {
                currentPreviewBlobUrl = URL.createObjectURL(file);
                dpmContent.innerHTML = `<iframe class="dpm-pdf-frame" src="${currentPreviewBlobUrl}#toolbar=1" title="${esc(info.name)}"></iframe>`;
            }
            // 3. Image Preview
            else if (mime.startsWith('image/') || ['png', 'jpg', 'jpeg', 'gif', 'svg', 'webp', 'bmp', 'ico'].includes(ext)) {
                currentPreviewBlobUrl = URL.createObjectURL(file);
                dpmContent.innerHTML = `<div class="dpm-image-container"><img class="dpm-preview-img" src="${currentPreviewBlobUrl}" alt="${esc(info.name)}"></div>`;
            }
            // 4. CSV / TSV Preview
            else if (ext === 'csv' || ext === 'tsv') {
                const text = await file.text();
                dpmContent.innerHTML = renderCsvPreview(text);
            }
            // 5. Code & Plain Text Preview
            else if (['txt', 'md', 'js', 'ts', 'jsx', 'tsx', 'py', 'html', 'htm', 'css', 'scss', 'json', 'xml', 'sql', 'log', 'sh', 'bash', 'bat', 'cmd', 'ps1', 'yml', 'yaml', 'c', 'cpp', 'h', 'hpp', 'java', 'go', 'rs', 'php', 'ini', 'env', 'diff'].includes(ext) || mime.startsWith('text/')) {
                const text = await file.text();
                dpmContent.innerHTML = `<div class="dpm-code-container"><pre class="dpm-code-block"><code>${esc(text)}</code></pre></div>`;
            }
            // 6. Video Preview
            else if (mime.startsWith('video/') || ['mp4', 'webm', 'ogg', 'mov'].includes(ext)) {
                currentPreviewBlobUrl = URL.createObjectURL(file);
                dpmContent.innerHTML = `<div class="dpm-image-container"><video controls autoplay style="max-width:90%;max-height:80vh;border-radius:8px;" src="${currentPreviewBlobUrl}"></video></div>`;
            }
            // 7. Audio Preview
            else if (mime.startsWith('audio/') || ['mp3', 'wav', 'aac', 'flac'].includes(ext)) {
                currentPreviewBlobUrl = URL.createObjectURL(file);
                dpmContent.innerHTML = `<div class="dpm-fallback-container"><audio controls src="${currentPreviewBlobUrl}" style="width:360px;"></audio></div>`;
            }
            // 8. Other / Unknown Binary Fallback
            else {
                dpmContent.innerHTML = renderFallbackPreview(info);
                bindFallbackDownload();
            }
        } catch (err) {
            console.error('Error generating preview:', err);
            dpmContent.innerHTML = renderFallbackPreview(info, 'Failed to display preview.');
            bindFallbackDownload();
        } finally {
            if (dpmLoading) dpmLoading.style.display = 'none';
        }
    }

    function getFileInfo(file) {
        const name = file.name || 'Document';
        const ext  = (name.split('.').pop() || '').toLowerCase();
        let iconHtml = '';
        let typeLabel = ext.toUpperCase() || 'FILE';

        if (file.type?.startsWith('image/')) {
            const url = URL.createObjectURL(file);
            iconHtml = `<div class="att-card-icon img"><img src="${url}" alt="${esc(name)}"></div>`;
            typeLabel = 'IMAGE';
        } else if (ext === 'pdf') {
            iconHtml = `<div class="att-card-icon pdf"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="9" y1="15" x2="15" y2="15"/></svg></div>`;
            typeLabel = 'PDF';
        } else if (['doc', 'docx', 'odt', 'rtf'].includes(ext)) {
            iconHtml = `<div class="att-card-icon doc"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg></div>`;
            typeLabel = 'DOCX';
        } else if (['js', 'ts', 'py', 'html', 'css', 'json', 'xml', 'sql'].includes(ext)) {
            iconHtml = `<div class="att-card-icon code"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg></div>`;
            typeLabel = ext.toUpperCase();
        } else if (['csv', 'xlsx', 'xls'].includes(ext)) {
            iconHtml = `<div class="att-card-icon other" style="background:#dcfce7;color:#15803d;border-color:#bbf7d0"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="3" y1="15" x2="21" y2="15"/><line x1="9" y1="3" x2="9" y2="21"/><line x1="15" y1="3" x2="15" y2="21"/></svg></div>`;
            typeLabel = 'SHEET';
        } else {
            iconHtml = `<div class="att-card-icon other"><svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg></div>`;
        }

        const sizeKB = (file.size / 1024).toFixed(file.size > 1024 * 1024 ? 0 : 1);
        const sizeStr = file.size > 1024 * 1024 ? (file.size / (1024 * 1024)).toFixed(1) + ' MB' : sizeKB + ' KB';

        return { name, ext, iconHtml, typeLabel, sizeStr };
    }

    function renderAttachments() {
        attTray.innerHTML = '';
        if (!attachedFiles.length) {
            attTray.classList.remove('has-files');
            if (chatActive) positionInput(false);
            return;
        }
        attTray.classList.add('has-files');
        attachedFiles.forEach((file, idx) => {
            const info = getFileInfo(file);
            const card = document.createElement('div');
            card.className = 'att-card';
            card.title = `Click to open ${info.name}`;

            card.innerHTML = `
                ${info.iconHtml}
                <div class="att-card-info">
                    <span class="att-card-name">${esc(info.name)}</span>
                    <span class="att-card-meta">${info.typeLabel} · ${info.sizeStr}</span>
                </div>
                <button class="att-card-rm" title="Remove attachment" aria-label="Remove">&times;</button>
            `;

            // Open document on click
            card.addEventListener('click', (e) => {
                if (e.target.closest('.att-card-rm')) return;
                openFile(file);
            });

            // Remove button
            const rmBtn = card.querySelector('.att-card-rm');
            rmBtn?.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                attachedFiles.splice(idx, 1);
                renderAttachments();
            });

            attTray.appendChild(card);
        });

        if (chatActive) positionInput(false);
    }

    // ══════════════════════════════════════════════════
    // TEXTAREA AUTO-RESIZE
    // ══════════════════════════════════════════════════
    promptInput.addEventListener('input', () => {
        promptInput.style.height = 'auto';
        promptInput.style.height = Math.min(promptInput.scrollHeight, 150) + 'px';
        if (chatActive) positionInput(false);
    });

    // ══════════════════════════════════════════════════
    // EXECUTION TILES STREAM & DEPTH-OF-FIELD EDGE BLUR
    // ══════════════════════════════════════════════════
    let activeTimers = [];
    let currentAbortController = null;
    let execStepCounter = 0;

    function clearExecTimers() {
        activeTimers.forEach(id => clearTimeout(id));
        activeTimers = [];
    }

    // Progressive depth-of-field scroll blur:
    // 1. If tiles do NOT overflow the container, ALL tiles remain 100% crisp.
    // 2. When tiles overflow and user is at/near top, the bottom 1-2 slides blur slightly to indicate more content below.
    // 3. When scrolling down, bottom slides become completely clear.
    function updateTileBlurs() {
        if (!execTilesTrack) return;
        const trackRect = execTilesTrack.getBoundingClientRect();
        if (trackRect.height === 0) return;

        const tiles = execTilesTrack.querySelectorAll('.exec-tile');
        if (!tiles.length) {
            edgeBlurTop?.classList.remove('active');
            edgeBlurBottom?.classList.remove('active');
            return;
        }

        const scrollH = execTilesTrack.scrollHeight;
        const clientH = execTilesTrack.clientHeight;
        const scrollT = execTilesTrack.scrollTop;
        const hasOverflow = scrollH > clientH + 12;

        // If tiles do not fill up the workspace, nothing should be blurred!
        if (!hasOverflow) {
            tiles.forEach(tile => {
                tile.style.filter = 'none';
                tile.style.opacity = '1';
            });
            edgeBlurTop?.classList.remove('active');
            edgeBlurBottom?.classList.remove('active');
            return;
        }

        const canScrollDown = (scrollH - scrollT - clientH) > 12;
        const canScrollUp   = scrollT > 12;

        // Edge gradient overlays
        edgeBlurTop?.classList.toggle('active', canScrollUp);
        edgeBlurBottom?.classList.toggle('active', canScrollDown);

        const bottomThreshold = 72; // px from container bottom where blur begins
        const topThreshold    = 60; // px from container top where blur begins when scrolled down

        tiles.forEach(tile => {
            const rect = tile.getBoundingClientRect();
            const distFromBottom = trackRect.bottom - rect.bottom;
            const distFromTop    = rect.top - trackRect.top;

            // Only blur bottom tiles if there is more content below (user is at or near top / middle)
            if (canScrollDown && distFromBottom < bottomThreshold) {
                const ratio = Math.max(0, distFromBottom + 20) / (bottomThreshold + 20);
                const blurPx = Math.min(3.5, ((1 - ratio) * 4)).toFixed(1);
                const opacity = (0.55 + ratio * 0.45).toFixed(2);
                tile.style.filter = blurPx > 0.4 ? `blur(${blurPx}px)` : 'none';
                tile.style.opacity = opacity;
            }
            // Only blur top tiles if the user has scrolled DOWN away from the top
            else if (canScrollUp && distFromTop < topThreshold) {
                const ratio = Math.max(0, distFromTop + 20) / (topThreshold + 20);
                const blurPx = Math.min(3.5, ((1 - ratio) * 4)).toFixed(1);
                const opacity = (0.55 + ratio * 0.45).toFixed(2);
                tile.style.filter = blurPx > 0.4 ? `blur(${blurPx}px)` : 'none';
                tile.style.opacity = opacity;
            }
            // In the active focal reading zone: 100% crisp and clear
            else {
                tile.style.filter = 'none';
                tile.style.opacity = '1';
            }
        });
    }

    execTilesTrack?.addEventListener('scroll', updateTileBlurs);

    // Progress is chronological: the first step stays at the top and new work is appended below it.
    function spawnExecTile({ id, actor, actorClass, action, detail, status = 'running', duration, location = 'On this device', observerSequence }) {
        if (execEmptyState) execEmptyState.style.display = 'none';

        const tile = document.createElement('div');
        tile.className = 'exec-tile';
        tile.dataset.id = id || ('tile_' + Date.now() + '_' + Math.random().toString(36).substr(2, 4));
        tile.dataset.status = status;
        tile.dataset.step = String(++execStepCounter);
        if (observerSequence != null) tile.dataset.observerSequence = String(observerSequence);
        tile.tabIndex = 0;
        tile.setAttribute('role', 'button');
        tile.setAttribute('aria-label', `Step ${execStepCounter}: ${action || 'Processing'}. Open technical details.`);

        const isDone = status === 'done';
        const isFailed = status === 'failed';
        const isTerminal = isDone || isFailed;
        tile.innerHTML = `
            <div class="et-head">
                <span class="et-step-number">${execStepCounter}</span>
                <span class="et-actor-badge ${actorClass || 'gemma'}">${esc(actor || 'SAGE')}</span>
                <span class="et-status ${isFailed ? 'failed' : isDone ? 'done' : 'running'}">
                    ${isFailed ? 'Needs attention' : isDone ? 'Done' : '<span class="et-spinner"></span> Working'}
                </span>
            </div>
            <div class="et-body">
                <div class="et-action">${esc(action || 'Processing')}</div>
                <div class="et-detail">${esc(detail || 'Working...')}</div>
            </div>
            <div class="et-foot">
                <span class="et-time">${duration || (isTerminal ? '0.2s' : '0.0s')}</span>
                <span class="et-pill">${esc(location)}</span>
            </div>
            <span class="et-details-hint">Open technical details</span>
        `;

        execTilesTrack.appendChild(tile);
        const openDetails = () => {
            const sequence = Number(tile.dataset.observerSequence);
            const event = observerEvents.find(item => Number(item.sequence) === sequence);
            setObserverModal(true);
            if (event) inspectObserverEvent(event, null);
        };
        tile.addEventListener('click', openDetails);
        tile.addEventListener('keydown', event => {
            if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); openDetails(); }
        });

        if (!isTerminal) {
            const startTime = performance.now();
            const timeEl = tile.querySelector('.et-time');
            const timer = setInterval(() => {
                if (tile.dataset.status !== 'running') {
                    clearInterval(timer);
                    return;
                }
                const sec = ((performance.now() - startTime) / 1000).toFixed(1);
                if (timeEl) timeEl.textContent = sec + 's';
            }, 100);
            tile._timer = timer;
            tile._startTime = startTime;
        }

        execTilesTrack.scrollTop = execTilesTrack.scrollHeight;
        updateTileBlurs();
        setTimeout(updateTileBlurs, 340);

        return tile;
    }

    function completeExecTile(tile, completionDetail, durationText, finalStatus = 'completed') {
        if (!tile) return;
        if (tile._timer) clearInterval(tile._timer);
        const failed = finalStatus === 'failed';
        tile.dataset.status = failed ? 'failed' : 'done';

        const st = tile.querySelector('.et-status');
        if (st) {
            st.className = `et-status ${failed ? 'failed' : 'done'}`;
            st.textContent = failed ? 'Needs attention' : 'Done';
        }
        if (completionDetail) {
            const dt = tile.querySelector('.et-detail');
            if (dt) dt.textContent = completionDetail;
        }
        const timeEl = tile.querySelector('.et-time');
        if (durationText && timeEl) {
            timeEl.textContent = durationText;
        } else if (tile._startTime && timeEl) {
            const s = ((performance.now() - tile._startTime) / 1000).toFixed(1);
            timeEl.textContent = s + 's';
        }

        updateTileBlurs();
    }

    function clearExecTiles() {
        clearExecTimers();
        execStepCounter = 0;
        observerTiles.clear();
        if (execTilesTrack) {
            const tiles = execTilesTrack.querySelectorAll('.exec-tile');
            tiles.forEach(t => {
                if (t._timer) clearInterval(t._timer);
                t.remove();
            });
        }
        if (execEmptyState) execEmptyState.style.display = 'flex';
        edgeBlurTop?.classList.remove('active');
        edgeBlurBottom?.classList.remove('active');
    }

    function rightPanelRunning() {
        if (execStatusDot) execStatusDot.className = 'exec-dot running';
        if (execStatusText) execStatusText.textContent = 'Running';
    }

    function rightPanelDone() {
        const memoryWorking = [...observerTiles.keys()].some(key => key.startsWith('memory-curation'));
        if (execStatusDot) execStatusDot.className = memoryWorking ? 'exec-dot running' : 'exec-dot done';
        if (execStatusText) execStatusText.textContent = memoryWorking ? 'Answer ready · memory working' : 'Done';
    }

    function rightPanelIdle() {
        if (execStatusDot) execStatusDot.className = 'exec-dot';
        if (execStatusText) execStatusText.textContent = 'Idle';
    }

    // ══════════════════════════════════════════════════
    // STOP BUTTON
    // ══════════════════════════════════════════════════
    stopBtn?.addEventListener('click', async () => {
        try {
            await fetch('/api/stop', { method: 'POST' });
        } catch {}
        if (currentAbortController) {
            currentAbortController.abort();
        }
    });

    function showCardStatus(text) {
        let st = document.getElementById('inputCardStatus');
        if (!st) {
            st = document.createElement('div');
            st.className = 'input-card-status';
            st.id = 'inputCardStatus';
            const card = document.getElementById('inputCard');
            if (card) card.appendChild(st);
        }
        st.innerHTML = `<span class="et-spinner"></span> <span>${esc(text)}</span>`;
        st.style.display = 'flex';
    }

    function hideCardStatus() {
        const st = document.getElementById('inputCardStatus');
        if (st) st.remove();
    }

    // ══════════════════════════════════════════════════
    // SEND MESSAGE
    // ══════════════════════════════════════════════════
    promptInput.addEventListener('keydown', e => {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
    });
    sendBtn.addEventListener('click', sendMessage);

    async function sendMessage() {
        if (isRunning) return;
        const text = promptInput.value.trim();
        if (!text && !attachedFiles.length) return;

        const isFirstMessage = !chatActive;
        const pendingText = text;
        const pendingFiles = [...attachedFiles];
        const historyTitle = text || (attachedFiles[0]?.name ? 'File: ' + attachedFiles[0].name : 'New conversation');

        isRunning = true;
        sendBtn.disabled = true;
        promptInput.disabled = true;
        stopBtn.style.display = 'flex';

        const formData = new FormData();
        formData.append('objective', text || 'Analyze the attached files.');
        const requestObserverRunId = `run_${crypto.randomUUID?.() || Date.now()}`;
        if (activeChatMode === 'flash') {
            formData.append('session_id', flashSessionId);
            formData.append('observer_run_id', requestObserverRunId);
            formData.append('temperature', String(uiSettings.temperature || '0.7'));
            formData.append('save_history', uiSettings.saveHistory ? '1' : '0');
        } else if (currentChatId) {
            formData.append('chat_id', currentChatId);
        }
        const fileNames = attachedFiles.map(f => f.name);
        const hasFiles = attachedFiles.length > 0;
        attachedFiles.forEach(f => formData.append('files', f));

        // If already active, show user bubble and thinking immediately
        let thinking = null;
        if (!isFirstMessage) {
            appendUserMsg(text, attachedFiles);
            attachedFiles = [];
            renderAttachments();
            promptInput.value = '';
            promptInput.style.height = 'auto';
            positionInput(false);
            thinking = appendThinking();
        } else {
            // First message: chatbox stays in the center while user types and models run
            showCardStatus('SAGE is reasoning & processing...');
        }

        // Reset execution tiles, expand right panel, and set status to running
        clearExecTiles();
        expandPanel();
        rightPanelRunning();

        // Execution tiles now come only from real backend Observer events.
        let currentTile = null;

        currentAbortController = new AbortController();
        startObserver(activeChatMode === 'flash' ? requestObserverRunId : currentChatId);

        try {
            const res  = await fetch(activeChatMode === 'flash' ? '/api/flash' : '/api/chat', {
                method: 'POST',
                body: formData,
                signal: currentAbortController.signal
            });
            const data = await res.json();
            if (data && data.session_id) {
                flashSessionId = data.session_id;
            }
            if (data && data.chat_id) {
                currentChatId = data.chat_id;
            }
            if (uiSettings.saveHistory && data?.status !== 'error') {
                upsertHistoryItem(data.session_id || data.chat_id || flashSessionId, historyTitle);
            }
            if (thinking) thinking.remove();
            clearExecTimers();
            hideCardStatus();

            if (!res.ok || data.status === 'error') {
                completeExecTile(currentTile, 'Orchestration stopped: ' + (data.error || 'Server error'));
                if (isFirstMessage) {
                    attachedFiles = [];
                    renderAttachments();
                    promptInput.value = '';
                    promptInput.style.height = 'auto';
                    activateChat();
                    appendUserMsg(pendingText, pendingFiles);
                }
                appendError(data.error || 'Orchestration error.');
                rightPanelIdle();
            } else {
                // PROMPT RESULT IS READY -> MAKE THE CHATBOX GO DOWN NOW!
                if (isFirstMessage) {
                    attachedFiles = [];
                    renderAttachments();
                    promptInput.value = '';
                    promptInput.style.height = 'auto';
                    activateChat();
                    appendUserMsg(pendingText, pendingFiles);
                }

                appendSageReply(data.answer, data.telemetry);
                rightPanelDone();
                if (artifactGraph) {
                    artifactGraph.loadData();
                }
            }
        } catch (err) {
            if (thinking) thinking.remove();
            clearExecTimers();
            hideCardStatus();
            if (err.name === 'AbortError') {
                completeExecTile(currentTile, 'Execution halted by user.');
            } else {
                completeExecTile(currentTile, 'Network error during request.');
                if (isFirstMessage) {
                    attachedFiles = [];
                    renderAttachments();
                    promptInput.value = '';
                    promptInput.style.height = 'auto';
                    activateChat();
                    appendUserMsg(pendingText, pendingFiles);
                }
                appendError('Network error: ' + err.message);
            }
            rightPanelIdle();
        } finally {
            isRunning = false;
            currentAbortController = null;
            sendBtn.disabled = false;
            promptInput.disabled = false;
            stopBtn.style.display = 'none';
            promptInput.focus();
        }
    }

    // ══════════════════════════════════════════════════
    // MESSAGE BUILDERS
    // ══════════════════════════════════════════════════
    function appendUserMsg(text, files) {
        const msg = document.createElement('div');
        msg.className = 'chat-msg user fade-in';
        const hdr = document.createElement('div'); hdr.className = 'msg-hdr'; hdr.textContent = 'You';
        const bub = document.createElement('div'); bub.className = 'msg-bub'; bub.textContent = text;
        if (files?.length) {
            const fc = document.createElement('div'); fc.className = 'msg-files';
            files.forEach(f => {
                const c = document.createElement('div');
                c.className = 'msg-chip';
                c.title = 'Click to open ' + f.name;
                c.innerHTML = `
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
                    <span>${esc(f.name)}</span>
                `;
                c.addEventListener('click', () => openFile(f));
                fc.appendChild(c);
            });
            bub.appendChild(fc);
        }
        msg.appendChild(hdr); msg.appendChild(bub);
        chatMessages.appendChild(msg);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function appendThinking() {
        const msg = document.createElement('div'); msg.className = 'chat-msg sage fade-in';
        const hdr = document.createElement('div'); hdr.className = 'msg-hdr'; hdr.textContent = 'SAGE';
        const bub = document.createElement('div'); bub.className = 'msg-bub';
        bub.innerHTML = '<em style="color:var(--tx3)">Reasoning\u2026</em>';
        msg.appendChild(hdr); msg.appendChild(bub);
        chatMessages.appendChild(msg);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return msg;
    }

    function appendSageReply(answer, telemetry) {
        const msg = document.createElement('div'); msg.className = 'chat-msg sage fade-in';
        const hdr = document.createElement('div'); hdr.className = 'msg-hdr';
        const showTelemetry = uiSettings.telemetry && telemetry?.total_wall_time != null;
        const t = showTelemetry ? ' \u00B7 ' + Number(telemetry.total_wall_time).toFixed(1) + 's' : '';
        hdr.textContent = 'SAGE' + t;
        const bub = document.createElement('div'); bub.className = 'msg-bub';
        bub.innerHTML = fmtMd(answer);
        msg.appendChild(hdr); msg.appendChild(bub);
        chatMessages.appendChild(msg);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function appendError(err) {
        const msg = document.createElement('div'); msg.className = 'chat-msg sage fade-in';
        const hdr = document.createElement('div'); hdr.className = 'msg-hdr'; hdr.style.color = '#c0392b'; hdr.textContent = 'SAGE Error';
        const bub = document.createElement('div'); bub.className = 'msg-bub'; bub.style.borderColor = '#f5b7b1';
        bub.innerHTML = '<strong>Error:</strong> <pre>' + esc(err) + '</pre>';
        msg.appendChild(hdr); msg.appendChild(bub);
        chatMessages.appendChild(msg);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // ══════════════════════════════════════════════════
    // HELPERS
    // ══════════════════════════════════════════════════
    function esc(t) {
        return String(t).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    }
    function fmtMd(text) {
        if (!text) return '';
        let f = esc(text);
        f = f.replace(/```([a-zA-Z0-9_]*)\n([\s\S]*?)```/g, (_, lang, code) => '<pre><code>' + code + '</code></pre>');
        f = f.replace(/`([^`]+)`/g, '<code>$1</code>');
        f = f.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
        f = f.replace(/^### (.*$)/gim, '<h3>$1</h3>');
        f = f.replace(/^## (.*$)/gim,  '<h2>$1</h2>');
        f = f.replace(/^# (.*$)/gim,   '<h1>$1</h1>');
        f = f.replace(/\n/g, '<br>');
        f = f.replace(/<pre><code[\s\S]*?<\/code><\/pre>/g, m => m.replace(/<br>/g, '\n'));
        return f;
    }

    // ══════════════════════════════════════════════════
    // SAGE MEMORY VAULT CONTROLLER — REAL BACKEND INTEGRATION
    // ══════════════════════════════════════════════════
    let mvMemories = [];
    let mvDerivedMemories = [];
    let mvRecentMessages = [];
    let mvActivities = [];
    let mvCurrentTab = 'vault'; // 'vault' | 'recent' | 'derived' | 'activity'
    let mvCurrentCategory = 'all';
    let mvSearchQuery = '';
    let mvActiveEditingId = null;
    let mvInitialized = false;
    let mvIsLoading = false;
    const mvSelectedGlobalIds = new Set();
    const mvSelectedActivityIds = new Set();

    function getMvTagClass(cat) {
        const c = (cat || '').toLowerCase();
        switch (c) {
            case 'task': return 'mv-tag-task';
            case 'decision': return 'mv-tag-decision';
            case 'preference': return 'mv-tag-preference';
            case 'personal': return 'mv-tag-personal';
            case 'project': return 'mv-tag-project';
            case 'technical': return 'mv-tag-technical';
            case 'fact': return 'mv-tag-fact';
            default: return 'mv-tag-summary';
        }
    }

    function formatTimeAgo(dateStr) {
        if (!dateStr) return 'Just now';
        try {
            const d = new Date(dateStr);
            if (isNaN(d.getTime())) return String(dateStr);
            const diffMs = Date.now() - d.getTime();
            const diffMins = Math.floor(diffMs / 60000);
            if (diffMins < 1) return 'Just now';
            if (diffMins < 60) return `${diffMins}m ago`;
            const diffHours = Math.floor(diffMins / 60);
            if (diffHours < 24) return `${diffHours}h ago`;
            const diffDays = Math.floor(diffHours / 24);
            return `${diffDays}d ago`;
        } catch {
            return 'Recently';
        }
    }

    function getActBadgeClass(action) {
        const a = (action || '').toUpperCase();
        switch (a) {
            case 'STORE': return 'mv-act-store';
            case 'SEARCH': return 'mv-act-search';
            case 'UPDATE': return 'mv-act-update';
            case 'DELETE': return 'mv-act-delete';
            case 'PROMOTE': return 'mv-act-promote';
            case 'SUMMARIZE': return 'mv-act-summarize';
            default: return 'mv-act-store';
        }
    }

    // ── Live Backend API Fetchers ────────────────────
    async function fetchMemoryStatus() {
        try {
            const res = await fetch('/api/memory/status');
            if (!res.ok) return;
            const data = await res.json();
            
            // Database status pill
            const dbPillText = document.getElementById('mvDbStatusText');
            if (dbPillText && data.database) {
                const engine = (data.database.engine || 'SQLite').toUpperCase();
                const state = data.database.status === 'connected' ? 'Active' : 'Offline';
                dbPillText.textContent = `${engine} · ${state}`;
            }

            // Vector Index status pill
            const chromaText = document.getElementById('mvChromaStatusText');
            if (chromaText && data.vector_index) {
                const eng = data.vector_index.engine || 'Chroma';
                const status = data.vector_index.status || 'Ready';
                const hotCount = data.vector_index.hot_vectors || 0;
                const coldCount = data.vector_index.cold_vectors || 0;
                chromaText.textContent = `${eng} · ${status} (${hotCount + coldCount} vec)`;
            }

            // Embedding pill
            const embText = document.getElementById('mvEmbStatusText');
            if (embText && data.embedding) {
                embText.textContent = data.embedding.model || 'all-MiniLM-L6-v2';
            }

            // Model runtime pill
            const modelPill = document.getElementById('mvModelStatusPill');
            if (modelPill && data.model_runtime) {
                const jobs = data.model_runtime.jobs || {};
                const jobSummary = ` Jobs: ${jobs.queued || 0} queued, ${jobs.scheduled || 0} scheduled, ${jobs.running || 0} running, ${jobs.completed || 0} completed, ${jobs.failed || 0} failed.`;
                const lastError = data.model_runtime.last_error ? ` Last error: ${data.model_runtime.last_error}` : '';
                modelPill.title = (data.model_runtime.detail || 'Memory curator runtime status.') + jobSummary + lastError;
                const label = modelPill.querySelector('span:last-child');
                if (label) label.textContent = data.model_runtime.label || '2B CURATOR';
                modelPill.classList.toggle('pill-model-unavail', !data.model_runtime.available);
            }

            // Update badge counts if available
            if (data.counts) {
                const hotCountEl = document.getElementById('mvHotCount');
                const coldCountEl = document.getElementById('mvColdCount');
                const derivedCountEl = document.getElementById('mvDerivedCount');
                if (hotCountEl && data.counts.hot_active !== undefined) hotCountEl.textContent = data.counts.hot_active;
                if (coldCountEl && data.counts.cold_active !== undefined) coldCountEl.textContent = data.counts.cold_active;
                if (derivedCountEl && data.counts.derived_compacted !== undefined) derivedCountEl.textContent = data.counts.derived_compacted;
            }
        } catch (e) {
            console.warn('[MemoryVault] Status fetch failed:', e);
        }
    }

    async function fetchMemories() {
        try {
            let url = '/api/memory/memories?status=active';
            const activeMemoryChatId = currentChatId || flashSessionId;
            if (activeMemoryChatId) {
                url += `&chat_id=${encodeURIComponent(activeMemoryChatId)}`;
            }
            if (mvSearchQuery && mvSearchQuery.trim()) {
                url += `&search=${encodeURIComponent(mvSearchQuery.trim())}`;
            }
            if (mvCurrentCategory && mvCurrentCategory !== 'all') {
                url += `&category=${encodeURIComponent(mvCurrentCategory)}`;
            }
            const res = await fetch(url);
            if (!res.ok) return;
            const data = await res.json();
            mvMemories = Array.isArray(data.memories) ? data.memories : [];
            mvDerivedMemories = Array.isArray(data.derived) ? data.derived : [];
        } catch (e) {
            console.warn('[MemoryVault] Memories fetch failed:', e);
        }
    }

    async function fetchRecentChat() {
        try {
            const chatIdParam = currentChatId ? `?chat_id=${encodeURIComponent(currentChatId)}&limit=5` : '?limit=5';
            const res = await fetch(`/api/memory/recent-chat${chatIdParam}`);
            if (!res.ok) return;
            const data = await res.json();
            mvRecentMessages = Array.isArray(data.messages) ? data.messages : [];
            const countEl = document.getElementById('mvRecentMsgCount');
            if (countEl) countEl.textContent = mvRecentMessages.length;
        } catch (e) {
            console.warn('[MemoryVault] Recent chat fetch failed:', e);
        }
    }

    async function fetchActivityLogs() {
        try {
            const res = await fetch('/api/memory/activity?limit=50');
            if (!res.ok) return;
            const data = await res.json();
            mvActivities = Array.isArray(data.activities) ? data.activities : [];
            const countEl = document.getElementById('mvActivityCount');
            if (countEl) countEl.textContent = mvActivities.length;
        } catch (e) {
            console.warn('[MemoryVault] Activity logs fetch failed:', e);
        }
    }

    async function refreshMemoryVaultData(showIndicator = false) {
        if (mvIsLoading) return;
        mvIsLoading = true;
        const syncBtn = document.getElementById('mvSyncBtn');
        if (showIndicator && syncBtn) {
            syncBtn.style.opacity = '0.6';
            syncBtn.style.pointerEvents = 'none';
        }

        try {
            await Promise.all([
                fetchMemoryStatus(),
                fetchMemories(),
                fetchRecentChat(),
                fetchActivityLogs()
            ]);
            renderCurrentView();
        } finally {
            mvIsLoading = false;
            if (syncBtn) {
                syncBtn.style.opacity = '';
                syncBtn.style.pointerEvents = '';
            }
        }
    }

    function updateMvCounters() {
        const hotCount = mvMemories.filter(m => (m.scope === 'cold') || ((m.memory_tier || m.type) === 'cold' && m.source_chat_id)).length;
        const coldCount = mvMemories.filter(m => (m.scope === 'global') || ((m.memory_tier || m.type) === 'cold' && !m.source_chat_id)).length;
        const hotCountEl = document.getElementById('mvHotCount');
        const coldCountEl = document.getElementById('mvColdCount');
        if (hotCountEl) hotCountEl.textContent = hotCount;
        if (coldCountEl) coldCountEl.textContent = coldCount;
        const derivedCountEl = document.getElementById('mvDerivedCount');
        if (derivedCountEl) derivedCountEl.textContent = mvMemories.filter(m => m.scope === 'cold' || ((m.memory_tier || m.type) === 'cold' && m.source_chat_id)).length;
        const recentCountEl = document.getElementById('mvRecentMsgCount');
        if (recentCountEl) recentCountEl.textContent = mvRecentMessages.length;
        const activityCountEl = document.getElementById('mvActivityCount');
        if (activityCountEl) activityCountEl.textContent = mvActivities.length;
    }

    function createMvCardElement(item) {
        const id = item.memory_id || item.id;
        const card = document.createElement('div');
        card.className = 'mv-card';
        card.id = `mv_card_${id}`;

        const isEditing = (mvActiveEditingId === id);
        const timeAgo = formatTimeAgo(item.updated_at || item.created_at);
        const tier = item.memory_tier || item.type || 'cold';
        const category = (item.category || 'fact').toLowerCase();
        const isGlobal = item.scope === 'global' || ((item.memory_tier || item.type) === 'cold' && !item.source_chat_id);
        if (isGlobal) card.classList.add('has-bulk-select');

        if (isEditing) {
            card.innerHTML = `
                <div class="mv-card-header">
                    <div class="mv-card-header-left">
                        <span class="mv-tag ${getMvTagClass(category)}">${category.toUpperCase()}</span>
                        <span class="mv-card-time">${esc(timeAgo)}</span>
                    </div>
                    <span class="mv-card-id">id: ${esc(id)}</span>
                </div>
                <div class="mv-edit-box">
                    <textarea class="mv-edit-textarea" id="mv_edit_ta_${esc(id)}">${esc(item.content)}</textarea>
                    <div class="mv-edit-actions">
                        <button class="mv-btn-cancel-edit" onclick="window.mvCancelEdit('${esc(id)}')" title="Discard changes (Esc)">Cancel</button>
                        <button class="mv-btn-save-edit" onclick="window.mvSaveEdit('${esc(id)}')" title="Save changes">Save Changes</button>
                    </div>
                </div>
            `;
        } else {
            const promoteHtml = (tier === 'hot')
                ? `<button class="mv-btn mv-btn-promote" onclick="window.mvPromote('${esc(id)}')" title="Promote to persistent cold memory">Promote to Cold &nearr;</button>`
                : ``;

            card.innerHTML = `
                <div class="mv-card-header">
                    <div class="mv-card-header-left">
                        <span class="mv-tag ${getMvTagClass(category)}">${category.toUpperCase()}</span>
                        <span class="mv-card-time">${esc(timeAgo)}</span>
                    </div>
                    <span class="mv-card-id">id: ${esc(id)}</span>
                </div>
                <div class="mv-card-body">${esc(item.content)}</div>
                <div class="mv-card-toolbar">
                    <button class="mv-btn" onclick="window.mvStartEdit('${esc(id)}')" title="Edit content">Edit</button>
                    <button class="mv-btn mv-btn-delete" onclick="${isGlobal ? `window.mvHardDeleteGlobal(['${esc(id)}'])` : `window.mvDelete('${esc(id)}')`}" title="${isGlobal ? 'Permanently delete global memory' : 'Delete record'}">${isGlobal ? 'Delete permanently' : 'Delete'}</button>
                    ${promoteHtml}
                </div>
                ${isGlobal ? `<label class="mv-card-select" title="Select for permanent deletion"><input data-memory-id="${esc(id)}" type="checkbox" ${mvSelectedGlobalIds.has(id) ? 'checked' : ''} onchange="window.mvToggleGlobalSelection('${esc(id)}', this.checked)"></label>` : ''}
            `;
        }
        return card;
    }

    function renderMvVaultCards() {
        updateMvCounters();
        const hotStack = document.getElementById('mvHotCardsStack');
        const coldStack = document.getElementById('mvColdCardsStack');
        if (!hotStack || !coldStack) return;

        const filterFn = (item) => {
            const cat = (item.category || '').toLowerCase();
            if (mvCurrentCategory !== 'all' && cat !== mvCurrentCategory.toLowerCase()) {
                return false;
            }
            if (mvSearchQuery.trim() !== '') {
                const q = mvSearchQuery.toLowerCase().trim();
                const inContent = (item.content || '').toLowerCase().includes(q);
                const inCat = cat.includes(q);
                const inId = String(item.memory_id || item.id || '').toLowerCase().includes(q);
                if (!inContent && !inCat && !inId) return false;
            }
            return true;
        };

        const hotItems = mvMemories.filter(m => ((m.scope === 'cold') || ((m.memory_tier || m.type) === 'cold' && m.source_chat_id)) && filterFn(m));
        const coldItems = mvMemories.filter(m => ((m.scope === 'global') || ((m.memory_tier || m.type) === 'cold' && !m.source_chat_id)) && filterFn(m));
        const visibleGlobalIds = coldItems.map(item => String(item.memory_id || item.id));
        syncBulkToolbar('global', visibleGlobalIds);

        // Render Hot Items
        hotStack.innerHTML = '';
        if (hotItems.length === 0) {
            hotStack.innerHTML = `
                <div class="mv-empty-state">
                    <div class="mv-empty-title">No Older Chat Memories</div>
                    <div class="mv-empty-desc">Completed turns leaving the Hot five-turn window will be curated here.</div>
                </div>
            `;
        } else {
            hotItems.forEach(item => hotStack.appendChild(createMvCardElement(item)));
        }

        // Render Cold Items
        coldStack.innerHTML = '';
        if (coldItems.length === 0) {
            coldStack.innerHTML = `
                <div class="mv-empty-state">
                    <div class="mv-empty-title">No Global Memories Found</div>
                    <div class="mv-empty-desc">Stable facts and preferences classified by the curator appear here across chats.</div>
                </div>
            `;
        } else {
            coldItems.forEach(item => coldStack.appendChild(createMvCardElement(item)));
        }
    }

    function renderMvRecentMsgs() {
        const stack = document.getElementById('mvRecentMsgsStack');
        if (!stack) return;
        stack.innerHTML = '';

        if (!mvRecentMessages || mvRecentMessages.length === 0) {
            stack.innerHTML = `
                <div class="mv-empty-state">
                    <div class="mv-empty-title">No Recent Messages Found</div>
                    <div class="mv-empty-desc">Start a conversation in the Chat tab to populate the chronological 5-message context window.</div>
                </div>
            `;
            return;
        }

        mvRecentMessages.forEach(msg => {
            const role = (msg.role || 'user').toLowerCase();
            const roleClass = role === 'user' ? 'mv-role-user' : 'mv-role-assistant';
            const card = document.createElement('div');
            card.className = 'mv-recent-card';
            card.innerHTML = `
                <div class="mv-recent-head">
                    <span class="mv-role-badge ${roleClass}">${esc(role.toUpperCase())}</span>
                    <span class="mv-recent-time">${esc(formatTimeAgo(msg.created_at))}</span>
                </div>
                <div class="mv-recent-body">${esc(msg.content || '')}</div>
            `;
            stack.appendChild(card);
        });
    }

    function renderMvDerivedCards() {
        const stack = document.getElementById('mvDerivedCardsStack');
        if (!stack) return;
        stack.innerHTML = '';
        const coldMemories = mvMemories.filter(item => item.scope === 'cold' || ((item.memory_tier || item.type) === 'cold' && item.source_chat_id));

        if (coldMemories.length === 0) {
            stack.innerHTML = `
                <div class="mv-empty-state">
                    <div class="mv-empty-title">No Cold Memories Yet</div>
                    <div class="mv-empty-desc">Cold memory appears after a completed turn leaves this chat's five-turn Hot window.</div>
                </div>
            `;
            return;
        }

        coldMemories.forEach(item => {
            stack.appendChild(createMvCardElement(item));
        });
    }

    function renderMvActivities() {
        const stream = document.getElementById('mvActivityStream');
        if (!stream) return;
        stream.innerHTML = '';

        const visibleActivityIds = mvActivities.map(item => String(item.activity_id || item.id));
        syncBulkToolbar('activity', visibleActivityIds);
        if (!mvActivities || mvActivities.length === 0) {
            stream.innerHTML = `
                <div class="mv-empty-state">
                    <div class="mv-empty-title">No Activity Events Recorded</div>
                    <div class="mv-empty-desc">Memory store, search, update, delete, and promotion events are recorded here in real time.</div>
                </div>
            `;
            return;
        }

        mvActivities.forEach(act => {
            const card = document.createElement('div');
            card.className = 'mv-activity-card';
            const action = (act.action || 'EVENT').toUpperCase();
            const badgeClass = getActBadgeClass(action);
            card.innerHTML = `
                <label class="mv-activity-select" title="Select for permanent deletion"><input data-activity-id="${esc(String(act.activity_id || act.id))}" type="checkbox" ${mvSelectedActivityIds.has(String(act.activity_id || act.id)) ? 'checked' : ''} onchange="window.mvToggleActivitySelection('${esc(String(act.activity_id || act.id))}', this.checked)"></label>
                <div class="mv-activity-left">
                    <span class="mv-act-badge ${badgeClass}">${esc(action)}</span>
                    <span class="mv-act-details">${esc(act.details || '')}</span>
                </div>
                <span class="mv-act-time">${esc(formatTimeAgo(act.created_at))}</span>
            `;
            stream.appendChild(card);
        });
    }

    function renderCurrentView() {
        updateMvCounters();
        if (mvCurrentTab === 'vault') {
            renderMvVaultCards();
        } else if (mvCurrentTab === 'recent') {
            renderMvRecentMsgs();
        } else if (mvCurrentTab === 'derived') {
            renderMvDerivedCards();
        } else if (mvCurrentTab === 'activity') {
            renderMvActivities();
        }
    }

    function switchMvSubnav(viewName) {
        mvCurrentTab = viewName;
        // Update subnav buttons active state
        document.querySelectorAll('#mvSubnavPills .mv-subnav-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.view === viewName);
        });

        // Hide all views
        const views = {
            vault: document.getElementById('mvViewVault'),
            recent: document.getElementById('mvViewRecent'),
            derived: document.getElementById('mvViewDerived'),
            activity: document.getElementById('mvViewActivity'),
        };

        Object.entries(views).forEach(([name, el]) => {
            if (el) {
                if (name === 'vault') {
                    el.style.display = (name === viewName) ? 'grid' : 'none';
                } else {
                    el.style.display = (name === viewName) ? 'flex' : 'none';
                }
            }
        });

        renderCurrentView();
    }

    // ── Global Helper Methods for Inline Card Events ──
    window.mvStartEdit = function(id) {
        mvActiveEditingId = id;
        renderCurrentView();
        setTimeout(() => {
            const ta = document.getElementById(`mv_edit_ta_${id}`);
            if (ta) {
                ta.focus();
                ta.selectionStart = ta.selectionEnd = ta.value.length;
            }
        }, 20);
    };

    window.mvCancelEdit = function() {
        mvActiveEditingId = null;
        renderCurrentView();
    };

    window.mvSaveEdit = async function(id) {
        const ta = document.getElementById(`mv_edit_ta_${id}`);
        if (!ta) return;
        const newText = ta.value.trim();
        if (!newText) return;

        try {
            const res = await fetch(`/api/memory/${encodeURIComponent(id)}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ content: newText })
            });
            if (res.ok) {
                mvActiveEditingId = null;
                await refreshMemoryVaultData();
            } else {
                const err = await res.json().catch(() => ({ detail: 'Failed to update memory' }));
                alert(`Error updating memory: ${err.detail || 'Unknown error'}`);
            }
        } catch (e) {
            console.error('[MemoryVault] Save edit failed:', e);
            alert('Failed to save memory update.');
        }
    };

    window.mvPromote = async function(id) {
        try {
            const res = await fetch(`/api/memory/${encodeURIComponent(id)}/promote`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({})
            });
            if (res.ok) {
                await refreshMemoryVaultData();
            } else {
                const err = await res.json().catch(() => ({ detail: 'Failed to promote memory' }));
                alert(`Error promoting memory: ${err.detail || 'Unknown error'}`);
            }
        } catch (e) {
            console.error('[MemoryVault] Promotion failed:', e);
            alert('Failed to promote memory.');
        }
    };

    window.mvDelete = async function(id) {
        if (!confirm('Are you sure you want to delete this memory record?')) return;
        try {
            const res = await fetch(`/api/memory/${encodeURIComponent(id)}`, {
                method: 'DELETE'
            });
            if (res.ok) {
                if (mvActiveEditingId === id) mvActiveEditingId = null;
                await refreshMemoryVaultData();
            } else {
                const err = await res.json().catch(() => ({ detail: 'Failed to delete memory' }));
                alert(`Error deleting memory: ${err.detail || 'Unknown error'}`);
            }
        } catch (e) {
            console.error('[MemoryVault] Delete failed:', e);
            alert('Failed to delete memory.');
        }
    };

    function syncBulkToolbar(kind, visibleIds) {
        const isGlobal = kind === 'global';
        const selected = isGlobal ? mvSelectedGlobalIds : mvSelectedActivityIds;
        const prefix = isGlobal ? 'mvGlobal' : 'mvActivity';
        const valid = new Set(visibleIds);
        for (const id of [...selected]) {
            if (!valid.has(String(id))) selected.delete(id);
        }
        const selectAll = document.getElementById(`${prefix}SelectAll`);
        const count = document.getElementById(`${prefix}SelectedCount`);
        const deleteSelected = document.getElementById(`${prefix}DeleteSelectedBtn`);
        const deleteAll = document.getElementById(`${prefix}DeleteAllBtn`);
        if (selectAll) {
            selectAll.checked = visibleIds.length > 0 && visibleIds.every(id => selected.has(id));
            selectAll.indeterminate = selected.size > 0 && !selectAll.checked;
            selectAll.disabled = visibleIds.length === 0;
        }
        if (count) count.textContent = `${selected.size} selected`;
        if (deleteSelected) deleteSelected.disabled = selected.size === 0;
        if (deleteAll) deleteAll.disabled = visibleIds.length === 0;
    }

    window.mvToggleGlobalSelection = function(id, checked) {
        if (checked) mvSelectedGlobalIds.add(String(id)); else mvSelectedGlobalIds.delete(String(id));
        renderCurrentView();
    };

    window.mvToggleActivitySelection = function(id, checked) {
        if (checked) mvSelectedActivityIds.add(String(id)); else mvSelectedActivityIds.delete(String(id));
        renderCurrentView();
    };

    async function hardDeleteBulk(kind, ids) {
        const uniqueIds = [...new Set(ids.map(String).filter(Boolean))];
        if (!uniqueIds.length) return;
        const noun = kind === 'global' ? 'Global memory record' : 'activity-log entry';
        if (!confirm(`Permanently delete ${uniqueIds.length} ${noun}${uniqueIds.length === 1 ? '' : 's'}? This cannot be undone.`)) return;
        const endpoint = kind === 'global' ? '/api/memory/bulk-hard-delete' : '/api/memory/activity/bulk-hard-delete';
        const body = kind === 'global' ? { memory_ids: uniqueIds } : { activity_ids: uniqueIds };
        try {
            const res = await fetch(endpoint, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
            if (!res.ok) {
                const err = await res.json().catch(() => ({ detail: 'Permanent delete failed' }));
                throw new Error(err.detail || 'Permanent delete failed');
            }
            if (kind === 'global') mvSelectedGlobalIds.clear(); else mvSelectedActivityIds.clear();
            await refreshMemoryVaultData();
        } catch (error) {
            console.error('[MemoryVault] bulk hard delete failed:', error);
            alert(error.message || 'Permanent delete failed.');
        }
    }

    window.mvHardDeleteGlobal = function(ids) { return hardDeleteBulk('global', ids); };

    window.sageJumpToMemory = function(targetId, targetType) {
        // 1. Switch to Memory tab in top navbar
        const memTabBtn = document.querySelector('.tab-btn[data-tab="memory"]');
        if (memTabBtn) memTabBtn.click();

        // 2. Switch subnav if target is derived
        if (targetType === 'derived' || targetType === 'compacted') {
            switchMvSubnav('derived');
        } else {
            switchMvSubnav('vault');
        }

        // 3. Reset category filter and search so target card is visible
        if (mvCurrentCategory !== 'all') {
            const allChip = document.querySelector('#mvCategoryChips .mv-chip[data-cat="all"]');
            if (allChip) allChip.click();
        }
        const searchInput = document.getElementById('mvSearchInput');
        if (searchInput && searchInput.value !== '') {
            searchInput.value = '';
            mvSearchQuery = '';
            fetchMemories().then(renderCurrentView);
        }

        // 4. Smooth scroll target card into view and trigger highlight animation
        setTimeout(() => {
            const cardEl = document.getElementById(`mv_card_${targetId}`);
            if (cardEl) {
                cardEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
                cardEl.classList.remove('highlight-flash');
                void cardEl.offsetWidth; // force reflow
                cardEl.classList.add('highlight-flash');
                setTimeout(() => {
                    cardEl.classList.remove('highlight-flash');
                }, 1600);
            }
        }, 200);
    };

    function initMemoryVault() {
        if (!memoryVaultView) return;

        if (!mvInitialized) {
            mvInitialized = true;

            const searchInput = document.getElementById('mvSearchInput');
            const clearSearchBtn = document.getElementById('mvClearSearchBtn');
            const categoryChips = document.getElementById('mvCategoryChips');
            const closeBtn = document.getElementById('mvCloseBtn');
            const syncBtn = document.getElementById('mvSyncBtn');
            const subnavPills = document.getElementById('mvSubnavPills');

            const hotInput = document.getElementById('mvHotInput');
            const hotCategory = document.getElementById('mvHotCategory');
            const hotAddBtn = document.getElementById('mvHotAddBtn');

            const coldInput = document.getElementById('mvColdInput');
            const coldCategory = document.getElementById('mvColdCategory');
            const coldAddBtn = document.getElementById('mvColdAddBtn');
            const globalSelectAll = document.getElementById('mvGlobalSelectAll');
            const globalDeleteSelectedBtn = document.getElementById('mvGlobalDeleteSelectedBtn');
            const globalDeleteAllBtn = document.getElementById('mvGlobalDeleteAllBtn');
            const activitySelectAll = document.getElementById('mvActivitySelectAll');
            const activityDeleteSelectedBtn = document.getElementById('mvActivityDeleteSelectedBtn');
            const activityDeleteAllBtn = document.getElementById('mvActivityDeleteAllBtn');

            const visibleGlobalIds = () => [...document.querySelectorAll('#mvColdCardsStack input[data-memory-id]')]
                .map(input => String(input.dataset.memoryId || ''))
                .filter(Boolean);
            const visibleActivityIds = () => [...document.querySelectorAll('#mvActivityStream input[data-activity-id]')]
                .map(input => String(input.dataset.activityId || ''))
                .filter(Boolean);
            globalSelectAll?.addEventListener('change', () => {
                const ids = visibleGlobalIds();
                ids.forEach(id => globalSelectAll.checked ? mvSelectedGlobalIds.add(id) : mvSelectedGlobalIds.delete(id));
                renderCurrentView();
            });
            globalDeleteSelectedBtn?.addEventListener('click', () => hardDeleteBulk('global', [...mvSelectedGlobalIds]));
            globalDeleteAllBtn?.addEventListener('click', () => hardDeleteBulk('global', visibleGlobalIds()));
            activitySelectAll?.addEventListener('change', () => {
                const ids = visibleActivityIds();
                ids.forEach(id => activitySelectAll.checked ? mvSelectedActivityIds.add(id) : mvSelectedActivityIds.delete(id));
                renderCurrentView();
            });
            activityDeleteSelectedBtn?.addEventListener('click', () => hardDeleteBulk('activity', [...mvSelectedActivityIds]));
            activityDeleteAllBtn?.addEventListener('click', () => hardDeleteBulk('activity', visibleActivityIds()));

            // Subnav view switching
            subnavPills?.querySelectorAll('.mv-subnav-btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    const view = btn.dataset.view;
                    if (view) switchMvSubnav(view);
                });
            });

            // Live State Sync button
            syncBtn?.addEventListener('click', () => {
                refreshMemoryVaultData(true);
            });

            // Search (with backend search query fetch)
            let searchTimeout = null;
            searchInput?.addEventListener('input', (e) => {
                mvSearchQuery = e.target.value;
                if (clearSearchBtn) {
                    clearSearchBtn.style.display = mvSearchQuery ? 'block' : 'none';
                }
                clearTimeout(searchTimeout);
                searchTimeout = setTimeout(async () => {
                    await fetchMemories();
                    renderCurrentView();
                }, 250);
            });

            clearSearchBtn?.addEventListener('click', async () => {
                if (searchInput) searchInput.value = '';
                mvSearchQuery = '';
                clearSearchBtn.style.display = 'none';
                await fetchMemories();
                renderCurrentView();
            });

            // Category Filter Chips
            categoryChips?.querySelectorAll('.mv-chip').forEach(chip => {
                chip.addEventListener('click', async () => {
                    categoryChips.querySelectorAll('.mv-chip').forEach(c => c.classList.remove('active'));
                    chip.classList.add('active');
                    mvCurrentCategory = chip.dataset.cat || 'all';
                    await fetchMemories();
                    renderCurrentView();
                });
            });

            // Dedicated Hot Memory Submission (Real Backend POST)
            const addHotEntry = async () => {
                if (!hotInput) return;
                const text = hotInput.value.trim();
                if (!text) {
                    hotInput.focus();
                    return;
                }
                const cat = hotCategory ? hotCategory.value : 'project';
                const chatId = currentChatId || flashSessionId;

                try {
                    const res = await fetch('/api/memory', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            content: text,
                            category: cat,
                            tier: 'cold',
                            scope: 'cold',
                            chat_id: chatId
                        })
                    });
                    if (res.ok) {
                        hotInput.value = '';
                        await refreshMemoryVaultData();
                    } else {
                        const err = await res.json().catch(() => ({ detail: 'Failed to create memory' }));
                        alert(`Error creating chat memory: ${err.detail || 'Unknown error'}`);
                    }
                } catch (e) {
                    console.error('[MemoryVault] Chat memory creation error:', e);
                    alert('Failed to connect to backend memory API.');
                }
            };

            hotAddBtn?.addEventListener('click', addHotEntry);
            hotInput?.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    addHotEntry();
                }
            });

            // Dedicated Cold Memory Submission (Real Backend POST)
            const addColdEntry = async () => {
                if (!coldInput) return;
                const text = coldInput.value.trim();
                if (!text) {
                    coldInput.focus();
                    return;
                }
                const cat = coldCategory ? coldCategory.value : 'personal';

                try {
                    const res = await fetch('/api/memory', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            content: text,
                            category: cat,
                            tier: 'cold',
                            scope: 'global'
                        })
                    });
                    if (res.ok) {
                        coldInput.value = '';
                        await refreshMemoryVaultData();
                    } else {
                        const err = await res.json().catch(() => ({ detail: 'Failed to create memory' }));
                        alert(`Error creating global memory: ${err.detail || 'Unknown error'}`);
                    }
                } catch (e) {
                    console.error('[MemoryVault] Global memory creation error:', e);
                    alert('Failed to connect to backend memory API.');
                }
            };

            coldAddBtn?.addEventListener('click', addColdEntry);
            coldInput?.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    addColdEntry();
                }
            });

            // Close Button -> Switch to Chat Tab
            closeBtn?.addEventListener('click', () => {
                const chatTab = document.querySelector('.tab-btn[data-tab="chat"]');
                if (chatTab) chatTab.click();
            });

            // Global Escape listener for memory editing
            document.addEventListener('keydown', (e) => {
                if (e.key === 'Escape') {
                    if (mvActiveEditingId !== null) {
                        mvActiveEditingId = null;
                        renderCurrentView();
                    }
                }
            });
        }

        // Fetch fresh data whenever memory vault is opened
        refreshMemoryVaultData();
    }

    // ══════════════════════════════════════════════════
    // INITIAL IDLE STATE
    // ══════════════════════════════════════════════════
    clearExecTiles();
    rightPanelIdle();
});
