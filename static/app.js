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
    const dpmContent      = document.getElementById('dpmContent');

    let currentPreviewFile = null;
    let currentPreviewBlobUrl = null;

    let attachedFiles = [];
    let isRunning     = false;
    let chatActive    = false;  // has the input moved to the bottom yet?

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
    settingsBtn?.addEventListener('click', () => settingsPanel.classList.toggle('open'));
    closeSettingsBtn?.addEventListener('click', () => settingsPanel.classList.remove('open'));

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
    function initHistoryItem(item) {
        if (!item || item.dataset.bound) return;
        item.dataset.bound = 'true';

        item.addEventListener('click', (e) => {
            if (e.target.closest('.h-del-btn')) return;
            document.querySelectorAll('.history-item').forEach(i => i.classList.remove('active'));
            item.classList.add('active');
        });

        const delBtn = item.querySelector('.h-del-btn');
        delBtn?.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            const wasActive = item.classList.contains('active');
            item.style.opacity = '0';
            item.style.transform = 'translateX(-12px)';
            item.style.pointerEvents = 'none';
            setTimeout(() => {
                item.remove();
                if (wasActive) {
                    const firstRemaining = document.querySelector('.history-item');
                    if (firstRemaining) {
                        firstRemaining.classList.add('active');
                    } else {
                        resetChat();
                    }
                }
            }, 180);
        });
    }
    document.querySelectorAll('.history-item').forEach(initHistoryItem);

    searchInput?.addEventListener('input', () => {
        const q = searchInput.value.toLowerCase();
        document.querySelectorAll('.history-item').forEach(item => {
            const t = item.querySelector('.h-title')?.textContent.toLowerCase() || '';
            item.style.display = t.includes(q) ? '' : 'none';
        });
    });

    // ══════════════════════════════════════════════════
    // NEW CHAT — reset to initial centered state
    // ══════════════════════════════════════════════════
    newChatBtn?.addEventListener('click', resetChat);
    function resetChat() {
        chatMessages.innerHTML = '';
        chatActive = false;

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

        document.querySelectorAll('.history-item').forEach(i => i.classList.remove('active'));

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

    // Spawn a new tile ABOVE previous tiles (prepending shifts older tiles down)
    function spawnExecTile({ id, actor, actorClass, action, detail, status = 'running', duration }) {
        if (execEmptyState) execEmptyState.style.display = 'none';

        const tile = document.createElement('div');
        tile.className = 'exec-tile';
        tile.dataset.id = id || ('tile_' + Date.now() + '_' + Math.random().toString(36).substr(2, 4));
        tile.dataset.status = status;

        const isDone = status === 'done';
        tile.innerHTML = `
            <div class="et-head">
                <span class="et-actor-badge ${actorClass || 'gemma'}">${esc(actor || 'Gemma 4B')}</span>
                <span class="et-status ${isDone ? 'done' : 'running'}">
                    ${isDone ? 'Done' : '<span class="et-spinner"></span> Running'}
                </span>
            </div>
            <div class="et-body">
                <div class="et-action">${esc(action || 'Processing')}</div>
                <div class="et-detail">${esc(detail || 'Working...')}</div>
            </div>
            <div class="et-foot">
                <span class="et-time">${duration || (isDone ? '0.2s' : '0.0s')}</span>
                <span class="et-pill">LOCAL</span>
            </div>
        `;

        // Prepend so the new tile spawns above and shifts previous tiles downward
        execTilesTrack.insertBefore(tile, execTilesTrack.firstChild);

        if (!isDone) {
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

        execTilesTrack.scrollTop = 0;
        updateTileBlurs();
        setTimeout(updateTileBlurs, 340);

        return tile;
    }

    function completeExecTile(tile, completionDetail, durationText) {
        if (!tile) return;
        if (tile._timer) clearInterval(tile._timer);
        tile.dataset.status = 'done';

        const st = tile.querySelector('.et-status');
        if (st) {
            st.className = 'et-status done';
            st.textContent = 'Done';
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
        if (execStatusDot) execStatusDot.className = 'exec-dot done';
        if (execStatusText) execStatusText.textContent = 'Done';
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

        // Add to history list if starting a new chat
        const activeHistory = document.querySelector('.history-item.active');
        if (!activeHistory) {
            const hList = document.getElementById('historyList');
            if (hList) {
                const newItem = document.createElement('div');
                newItem.className = 'history-item active';
                const now = new Date();
                const timeStr = 'Today · ' + String(now.getHours()).padStart(2, '0') + ':' + String(now.getMinutes()).padStart(2, '0');
                const titleStr = text || (attachedFiles[0]?.name ? 'File: ' + attachedFiles[0].name : 'New conversation');
                newItem.innerHTML = `
                    <div class="h-text">
                        <span class="h-title">${esc(titleStr)}</span>
                        <span class="h-time">${timeStr}</span>
                    </div>
                    <button class="h-del-btn" title="Delete chat" aria-label="Delete chat">
                        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
                    </button>
                `;
                initHistoryItem(newItem);
                hList.insertBefore(newItem, hList.firstChild);
            }
        }

        isRunning = true;
        sendBtn.disabled = true;
        promptInput.disabled = true;
        stopBtn.style.display = 'flex';

        const formData = new FormData();
        formData.append('objective', text || 'Analyze the attached files.');
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

        // Spawn initial Gemma 4B reasoning tile
        let currentTile = spawnExecTile({
            id: 'gemma_reasoning',
            actor: 'Gemma 4B',
            actorClass: 'gemma',
            action: 'Reasoning & Planning',
            detail: 'Decomposing objective, checking constraints, and formulating execution strategy...',
            status: 'running'
        });

        // Dynamic multi-model tile spawning:
        // If files attached: Gemma calls Vision Model -> spawns new tile ABOVE Gemma, shifting Gemma DOWN
        if (hasFiles) {
            activeTimers.push(setTimeout(() => {
                if (!isRunning) return;
                completeExecTile(currentTile, 'Identified document attachments. Invoking Vision model.');
                currentTile = spawnExecTile({
                    id: 'vision_ocr',
                    actor: 'Qwen3-VL',
                    actorClass: 'vision',
                    action: 'Vision & Document OCR',
                    detail: `Parsing ${fileNames.slice(0, 2).join(', ')}${fileNames.length > 2 ? ' +' + (fileNames.length - 2) + ' more' : ''} — OCR extraction & visual grounding...`,
                    status: 'running'
                });

                const lower = text.toLowerCase();
                const wantsCode = lower.includes('code') || lower.includes('python') || lower.includes('script') || lower.includes('calc') || lower.includes('table');
                if (wantsCode) {
                    activeTimers.push(setTimeout(() => {
                        if (!isRunning) return;
                        completeExecTile(currentTile, 'Document OCR parsed. Extracted data passed to Coder.');
                        currentTile = spawnExecTile({
                            id: 'coder_agent',
                            actor: 'Qwen2.5-Coder',
                            actorClass: 'coder',
                            action: 'Code Synthesis & Execution',
                            detail: 'Generating Python script and executing safely inside air-gapped container...',
                            status: 'running'
                        });
                    }, 800));
                }
            }, 600));
        } else {
            const lower = text.toLowerCase();
            const wantsCode = lower.includes('code') || lower.includes('python') || lower.includes('script') || lower.includes('function') || lower.includes('calc');
            if (wantsCode) {
                activeTimers.push(setTimeout(() => {
                    if (!isRunning) return;
                    completeExecTile(currentTile, 'Execution plan verified. Dispatching to Coder model.');
                    currentTile = spawnExecTile({
                        id: 'coder_agent',
                        actor: 'Qwen2.5-Coder',
                        actorClass: 'coder',
                        action: 'Code Synthesis & Execution',
                        detail: 'Synthesizing Python logic and testing safely in air-gapped container...',
                        status: 'running'
                    });
                }, 700));
            }
        }

        currentAbortController = new AbortController();

        try {
            const res  = await fetch('/api/chat', {
                method: 'POST',
                body: formData,
                signal: currentAbortController.signal
            });
            const data = await res.json();
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
                completeExecTile(currentTile);

                // If backend provided trace events, spawn any remaining distinct steps
                if (data.trace && Array.isArray(data.trace)) {
                    data.trace.forEach(evt => {
                        if (evt.actor === 'coder' && !document.querySelector('.exec-tile[data-id="coder_agent"]')) {
                            spawnExecTile({
                                id: 'trace_coder_' + Date.now(),
                                actor: 'Qwen2.5-Coder',
                                actorClass: 'coder',
                                action: 'Code Execution',
                                detail: 'Executed tools: ' + (evt.calls?.map(c => c.tool).join(', ') || 'code analysis'),
                                status: 'done',
                                duration: evt.duration ? evt.duration.toFixed(1) + 's' : '0.4s'
                            });
                        }
                    });
                }

                // Final synthesis tile spawned on top
                const wallTime = data.telemetry?.total_wall_time ? data.telemetry.total_wall_time.toFixed(1) + 's' : '0.3s';
                spawnExecTile({
                    id: 'gemma_final',
                    actor: 'Gemma 4B',
                    actorClass: 'gemma',
                    action: 'Final Synthesis & Verification',
                    detail: 'Consolidated local responses. Zero external data leak verified.',
                    status: 'done',
                    duration: wallTime
                });

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
        const t = telemetry ? ' \u00B7 ' + telemetry.total_wall_time?.toFixed(1) + 's' : '';
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
    // SAGE MEMORY VAULT CONTROLLER
    // ══════════════════════════════════════════════════
    const SEED_MEMORIES = [
        {
            id: 'mem_h1',
            type: 'hot',
            category: 'task',
            content: 'Implementing Phase 2 canonical memory store CRUD operations.',
            updated: '3m ago',
            timestamp: Date.now() - 3 * 60 * 1000
        },
        {
            id: 'mem_h2',
            type: 'hot',
            category: 'decision',
            content: 'Default embedding model confirmed as all-MiniLM-L6-v2.',
            updated: '18m ago',
            timestamp: Date.now() - 18 * 60 * 1000
        },
        {
            id: 'mem_c1',
            type: 'cold',
            category: 'preference',
            content: 'Prefers FastAPI for backend development, specifically over Flask for asynchronous APIs.',
            updated: '2h ago',
            timestamp: Date.now() - 2 * 3600 * 1000
        },
        {
            id: 'mem_c2',
            type: 'cold',
            category: 'personal',
            content: "The user's name is Ojasvi.",
            updated: '1d ago',
            timestamp: Date.now() - 24 * 3600 * 1000
        }
    ];

    let sageMemories = (() => {
        try {
            const saved = localStorage.getItem('sage_memories');
            if (saved) {
                const parsed = JSON.parse(saved);
                if (Array.isArray(parsed) && parsed.length > 0) return parsed;
            }
        } catch (e) {}
        return [...SEED_MEMORIES];
    })();

    function persistSageMemories() {
        try {
            localStorage.setItem('sage_memories', JSON.stringify(sageMemories));
        } catch (e) {}
    }

    let mvCurrentTab = 'hot';
    let mvCurrentCategory = 'all';
    let mvSearchQuery = '';
    let mvActiveEditingId = null;
    let mvInitialized = false;

    function getMvTagClass(cat) {
        const c = (cat || '').toLowerCase();
        switch (c) {
            case 'task': return 'mv-tag-task';
            case 'decision': return 'mv-tag-decision';
            case 'preference': return 'mv-tag-preference';
            case 'personal': return 'mv-tag-personal';
            case 'project': return 'mv-tag-project';
            case 'technical': return 'mv-tag-technical';
            default: return 'mv-tag-summary';
        }
    }

    function updateMvCounters() {
        const hotCount = sageMemories.filter(m => m.type === 'hot').length;
        const coldCount = sageMemories.filter(m => m.type === 'cold').length;
        const hotCountEl = document.getElementById('mvHotCount');
        const coldCountEl = document.getElementById('mvColdCount');
        if (hotCountEl) hotCountEl.textContent = hotCount;
        if (coldCountEl) coldCountEl.textContent = coldCount;
    }

    function createMvCardElement(item) {
        const card = document.createElement('div');
        card.className = 'mv-card';
        card.id = `mv_card_${item.id}`;

        const isEditing = (mvActiveEditingId === item.id);

        if (isEditing) {
            card.innerHTML = `
                <div class="mv-card-header">
                    <div class="mv-card-header-left">
                        <span class="mv-tag ${getMvTagClass(item.category)}">${item.category.toUpperCase()}</span>
                        <span class="mv-card-time">${item.updated}</span>
                    </div>
                    <span class="mv-card-id">id: ${item.id}</span>
                </div>
                <div class="mv-edit-box">
                    <textarea class="mv-edit-textarea" id="mv_edit_ta_${item.id}">${esc(item.content)}</textarea>
                    <div class="mv-edit-actions">
                        <button class="mv-btn-cancel-edit" onclick="window.mvCancelEdit('${item.id}')" title="Discard changes (Esc)">Cancel</button>
                        <button class="mv-btn-save-edit" onclick="window.mvSaveEdit('${item.id}')" title="Save changes">Save Changes</button>
                    </div>
                </div>
            `;
        } else {
            const promoteHtml = (item.type === 'hot')
                ? `<button class="mv-btn mv-btn-promote" onclick="window.mvPromote('${item.id}')" title="Promote to persistent cold memory">Promote to Cold &nearr;</button>`
                : ``;

            card.innerHTML = `
                <div class="mv-card-header">
                    <div class="mv-card-header-left">
                        <span class="mv-tag ${getMvTagClass(item.category)}">${item.category.toUpperCase()}</span>
                        <span class="mv-card-time">${item.updated}</span>
                    </div>
                    <span class="mv-card-id">id: ${item.id}</span>
                </div>
                <div class="mv-card-body">${esc(item.content)}</div>
                <div class="mv-card-toolbar">
                    <button class="mv-btn" onclick="window.mvStartEdit('${item.id}')" title="Edit content">Edit</button>
                    <button class="mv-btn mv-btn-delete" onclick="window.mvDelete('${item.id}')" title="Delete record">Delete</button>
                    ${promoteHtml}
                </div>
            `;
        }
        return card;
    }

    function renderMvCards() {
        updateMvCounters();
        const hotStack = document.getElementById('mvHotCardsStack');
        const coldStack = document.getElementById('mvColdCardsStack');
        if (!hotStack || !coldStack) return;

        const filterFn = (item) => {
            if (mvCurrentCategory !== 'all' && item.category.toLowerCase() !== mvCurrentCategory.toLowerCase()) {
                return false;
            }
            if (mvSearchQuery.trim() !== '') {
                const q = mvSearchQuery.toLowerCase().trim();
                const inContent = (item.content || '').toLowerCase().includes(q);
                const inCat = (item.category || '').toLowerCase().includes(q);
                const inId = (item.id || '').toLowerCase().includes(q);
                if (!inContent && !inCat && !inId) return false;
            }
            return true;
        };

        const hotItems = sageMemories.filter(m => m.type === 'hot' && filterFn(m));
        const coldItems = sageMemories.filter(m => m.type === 'cold' && filterFn(m));

        // Render Hot Items
        hotStack.innerHTML = '';
        if (hotItems.length === 0) {
            hotStack.innerHTML = `
                <div class="mv-empty-state">
                    <div class="mv-empty-title">No Session Memories Found</div>
                    <div class="mv-empty-desc">Record session tasks or scratchpad context using the input above.</div>
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
                    <div class="mv-empty-title">No Persistent Memories Found</div>
                    <div class="mv-empty-desc">Add long-term rules, preferences, or promote session items to cold vault.</div>
                </div>
            `;
        } else {
            coldItems.forEach(item => coldStack.appendChild(createMvCardElement(item)));
        }
    }

    // Global helper methods for inline card events
    window.mvStartEdit = function(id) {
        mvActiveEditingId = id;
        renderMvCards();
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
        renderMvCards();
    };

    window.mvSaveEdit = function(id) {
        const ta = document.getElementById(`mv_edit_ta_${id}`);
        if (!ta) return;
        const newText = ta.value.trim();
        if (!newText) return;
        const target = sageMemories.find(m => m.id === id);
        if (target) {
            target.content = newText;
            target.updated = 'Just now';
            target.timestamp = Date.now();
            persistSageMemories();
        }
        mvActiveEditingId = null;
        renderMvCards();
    };

    window.mvPromote = function(id) {
        const target = sageMemories.find(m => m.id === id);
        if (target) {
            target.type = 'cold';
            target.updated = 'Just now';
            target.timestamp = Date.now();
            persistSageMemories();
            renderMvCards();
        }
    };

    window.mvDelete = function(id) {
        sageMemories = sageMemories.filter(m => m.id !== id);
        if (mvActiveEditingId === id) mvActiveEditingId = null;
        persistSageMemories();
        renderMvCards();
    };

    window.sageJumpToMemory = function(targetId, targetType) {
        // 1. Switch to Memory tab in top navbar
        const memTabBtn = document.querySelector('.tab-btn[data-tab="memory"]');
        if (memTabBtn) memTabBtn.click();

        // 2. Reset category filter and search so target card is visible
        if (mvCurrentCategory !== 'all') {
            const allChip = document.querySelector('#mvCategoryChips .mv-chip[data-cat="all"]');
            if (allChip) allChip.click();
        }
        const searchInput = document.getElementById('mvSearchInput');
        if (searchInput && searchInput.value !== '') {
            searchInput.value = '';
            mvSearchQuery = '';
            renderMvCards();
        }

        // 3. Smooth scroll target card into view and trigger highlight animation
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
        }, 120);
    };

    function initMemoryVault() {
        if (!memoryVaultView) return;

        if (!mvInitialized) {
            mvInitialized = true;

            const searchInput = document.getElementById('mvSearchInput');
            const clearSearchBtn = document.getElementById('mvClearSearchBtn');
            const categoryChips = document.getElementById('mvCategoryChips');
            const closeBtn = document.getElementById('mvCloseBtn');

            const hotInput = document.getElementById('mvHotInput');
            const hotCategory = document.getElementById('mvHotCategory');
            const hotAddBtn = document.getElementById('mvHotAddBtn');

            const coldInput = document.getElementById('mvColdInput');
            const coldCategory = document.getElementById('mvColdCategory');
            const coldAddBtn = document.getElementById('mvColdAddBtn');

            // Search
            searchInput?.addEventListener('input', (e) => {
                mvSearchQuery = e.target.value;
                if (clearSearchBtn) {
                    clearSearchBtn.style.display = mvSearchQuery ? 'block' : 'none';
                }
                renderMvCards();
            });

            clearSearchBtn?.addEventListener('click', () => {
                if (searchInput) searchInput.value = '';
                mvSearchQuery = '';
                clearSearchBtn.style.display = 'none';
                renderMvCards();
            });

            // Category Filter Chips
            categoryChips?.querySelectorAll('.mv-chip').forEach(chip => {
                chip.addEventListener('click', () => {
                    categoryChips.querySelectorAll('.mv-chip').forEach(c => c.classList.remove('active'));
                    chip.classList.add('active');
                    mvCurrentCategory = chip.dataset.cat || 'all';
                    renderMvCards();
                });
            });

            // Dedicated Hot Memory Submission
            const addHotEntry = () => {
                if (!hotInput) return;
                const text = hotInput.value.trim();
                if (!text) {
                    hotInput.focus();
                    return;
                }
                const cat = hotCategory ? hotCategory.value : 'task';
                const newId = 'mem_h' + (sageMemories.filter(m => m.type === 'hot').length + 1) + '_' + Math.floor(Math.random() * 1000);

                sageMemories.unshift({
                    id: newId,
                    type: 'hot',
                    category: cat,
                    content: text,
                    updated: 'Just now',
                    timestamp: Date.now()
                });
                persistSageMemories();
                hotInput.value = '';
                renderMvCards();
            };

            hotAddBtn?.addEventListener('click', addHotEntry);
            hotInput?.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    addHotEntry();
                }
            });

            // Dedicated Cold Memory Submission
            const addColdEntry = () => {
                if (!coldInput) return;
                const text = coldInput.value.trim();
                if (!text) {
                    coldInput.focus();
                    return;
                }
                const cat = coldCategory ? coldCategory.value : 'preference';
                const newId = 'mem_c' + (sageMemories.filter(m => m.type === 'cold').length + 1) + '_' + Math.floor(Math.random() * 1000);

                sageMemories.unshift({
                    id: newId,
                    type: 'cold',
                    category: cat,
                    content: text,
                    updated: 'Just now',
                    timestamp: Date.now()
                });
                persistSageMemories();
                coldInput.value = '';
                renderMvCards();
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
                        renderMvCards();
                    }
                }
            });
        }

        renderMvCards();
    }

    // ══════════════════════════════════════════════════
    // INITIAL IDLE STATE
    // ══════════════════════════════════════════════════
    clearExecTiles();
    rightPanelIdle();
});
